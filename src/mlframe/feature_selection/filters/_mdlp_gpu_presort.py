"""The device half of the supervised edge search: a block of columns sorted with its labels, the distinct values with their counts, and the deduplicated (value, label) pairs, from one composite-key sort.

The MDLP edge search of a column starts with three passes over its 500000 values that no decision depends on: ``sort_xy`` (the values in ascending order with their labels), ``sorted_unique_with_counts`` (the
distinct values and how often each occurs) and ``_dedupe_xy`` (the exact duplicate pairs collapsed, the labels of equal values in ascending order). Together they are about 35 percent of the CPU of a column
(18 of 49 ms) and run on every core in the first 35 seconds of a fit, where the cascades are also running: the cores are saturated and the GPU is idle. :func:`presort_block` does the three on the device for a
block of columns and hands the host arrays to the unchanged recursion, which stays on the CPU.

The values, the labels and the counts are those of the host passes bit for bit (the sign of a zero, the order of equal values, the rule that a column without exact duplicates keeps its arrays): the sort is
stable (a key is the order-preserving bits of the value, then the row), the runs are read off the sorted values, and the duplicate pairs are found by a second sort of ``(run, label)`` keys. The device is
used under a lock (one block at a time) and every failure returns ``None`` for the block: the host path then runs for those columns exactly as before.

Opt-in (``MLFRAME_MDLP_GPU_PRESORT=1``): the edges are equal and the process CPU falls by a third, but the edge search of an otherwise idle host is slower through the device (see :func:`mode`).
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from mlframe.utils.log_throttle import log_throttle

from ._kernel_types import render

logger = logging.getLogger(__name__)

#: Columns sorted together (the keys of a block are ``columns x rows`` 64-bit values, 32 MB for 8 columns of 500000 rows, and a sort needs about three times that).
BLOCK_COLUMNS = 8
#: Rows per edge-search worker from which the device presort is faster in wall than the host passes (stage A/B, 160 columns, 8 workers: 50000 rows 0.47 against 0.33 s, 100000 0.65 against 0.56, 200000 1.27 against
#: 1.19, 300000 1.75 against 1.88, 500000 2.87 against 3.16, 1000000 (80 columns) 2.98 against 3.31). The host passes run on every worker at once while the blocks go through the device one at a time, so the
#: more workers the host has, the later the device pays in wall; it saves 26 to 36 percent of the CPU at every size. 8 workers x 31250 = 250000 rows is the crossover measured on the 4-core / 8-thread development host.
MIN_ROWS_PER_WORKER = 31_250
#: Largest number of distinct label codes the pair keys can carry next to the run number (a run number is below 2 ** 23 for a block of 8 columns of 500000 rows).
_MAX_LABEL = (1 << 20) - 1

_DEVICE_LOCK = threading.Lock()


@dataclass(frozen=True)
class Presorted:
    """What the host passes would have produced for one column.

    ``xs`` / ``ys``: the finite values (double precision) in ascending order with their labels, ties in row order. ``uniq`` / ``counts``: the distinct values (in the column's dtype) and their counts.
    ``dedup``: ``(x, y)`` of the deduplicated pairs when the column has exact duplicate pairs, ``None`` when it has none (the host then keeps ``xs`` and ``ys`` as they are) or when the labels could not be keyed."""

    xs: np.ndarray
    ys: np.ndarray
    uniq: np.ndarray
    counts: np.ndarray
    dedup: "tuple[np.ndarray, np.ndarray] | None"


class SuppliedPresort(tuple):
    """``(xs, ys)`` as the host presort returns it, carrying what the device also computed for the column: ``uniq`` and ``counts`` (the distinct values and their counts) and ``dedup`` (see :class:`Presorted`)."""

    uniq: np.ndarray
    counts: np.ndarray
    dedup: "tuple[np.ndarray, np.ndarray] | None"

    def __new__(cls, presorted: Presorted) -> "SuppliedPresort":
        """The pair of arrays of ``presorted`` with its run facts attached."""
        self = super().__new__(cls, (presorted.xs, presorted.ys))
        self.uniq, self.counts, self.dedup = presorted.uniq, presorted.counts, presorted.dedup
        return self


#: The key of a (column, value, row) of a block, one thread a value: the column number above the order-preserving bits of the float32 value above the row number (a sort of the keys is a stable sort of every column).
_KEYS_SRC = r"""
extern "C" __global__
void mdlp_keys(const unsigned int* __restrict__ bits, unsigned long long* __restrict__ key, const long long n, const int row_bits) {
    const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const long long col = blockIdx.y;
    const unsigned int b = bits[col * n + r];
    const unsigned int ordered = b ^ ((b >> 31) ? 0xFFFFFFFFu : 0x80000000u);
    key[col * n + r] = ((unsigned long long)col << (32 + row_bits)) | ((unsigned long long)ordered << row_bits) | (unsigned long long)r;
}
"""

#: The sorted keys back to what the host passes hold: the value, its label, and whether it starts a run of equal values (a float compare, so -0.0 and +0.0 are one run as in the host's ``!=``).
_UNPACK_SRC = r"""
__device__ __forceinline__ {REAL} value_of(const unsigned long long k, const int row_bits) {
    const unsigned int top = (unsigned int)((k >> row_bits) & 0xFFFFFFFFull);
    return __uint_as_float((top & 0x80000000u) ? (top & 0x7FFFFFFFu) : ~top);
}
extern "C" __global__
void mdlp_unpack(const unsigned long long* __restrict__ sk, const {LAB}* __restrict__ labels, {REAL}* __restrict__ xs, {LAB}* __restrict__ ys,
                 unsigned char* __restrict__ start, const long long n, const int row_bits) {
    const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const long long idx = (long long)blockIdx.y * n + r;
    const unsigned long long k = sk[idx];
    const {REAL} x = value_of(k, row_bits);
    xs[idx] = x;
    ys[idx] = labels[(long long)(k & ((1ull << row_bits) - 1ull))];
    start[idx] = (r == 0) || (x != value_of(sk[idx - 1], row_bits));
}
"""

#: CUDA type of a label dtype (the unpack kernel is built once per label type).
_LABEL_CTYPE = {
    "int8": "signed char", "uint8": "unsigned char", "int16": "short", "uint16": "unsigned short",
    "int32": "int", "uint32": "unsigned int", "int64": "long long", "uint64": "unsigned long long",
}
_KERNELS: "dict[str, tuple]" = {}
_KERNELS_LOCK = threading.Lock()
#: Downloads of at least this many bytes go through page-locked memory (a pageable copy ran at 2.7 GB/s on the 1050 Ti, 32 MB in 11.9 ms).
_PINNED_MIN_BYTES = 1 << 20


def _kernels(cp: Any, label_dtype: np.dtype) -> tuple:
    """``(keys kernel, unpack kernel)`` for labels of this dtype, built once per process."""
    ctype = _LABEL_CTYPE[label_dtype.name]
    with _KERNELS_LOCK:
        if ctype not in _KERNELS:
            _KERNELS[ctype] = (
                cp.RawModule(code=_KEYS_SRC).get_function("mdlp_keys"),
                cp.RawModule(code=render(_UNPACK_SRC.replace("{LAB}", ctype), real=np.float32)).get_function("mdlp_unpack"),
            )
        return _KERNELS[ctype]


def _download(device_array: Any) -> np.ndarray:
    """``device_array`` on the host; a big one lands in page-locked memory (the copy then runs at the bus rate)."""
    if int(device_array.nbytes) < _PINNED_MIN_BYTES:
        return np.asarray(device_array.get())
    import cupyx

    out = cupyx.empty_pinned(device_array.shape, device_array.dtype)
    device_array.get(out=out)
    return np.asarray(out)


_SUPPLY = threading.local()


@contextlib.contextmanager
def supplied(column: np.ndarray, presorted: "Presorted | None") -> Iterator[None]:
    """Within the block, the edge search of ``column`` on THIS thread takes ``presorted`` instead of sorting (a no-op for ``None``)."""
    if presorted is None:
        yield
        return
    previous = getattr(_SUPPLY, "entry", None)
    _SUPPLY.entry = (column, SuppliedPresort(presorted))
    try:
        yield
    finally:
        _SUPPLY.entry = previous


def supplied_presort(column: np.ndarray) -> "SuppliedPresort | None":
    """The device presort supplied for ``column`` on this thread, if any (the object is the one passed to :func:`supplied`)."""
    entry = getattr(_SUPPLY, "entry", None)
    return entry[1] if entry is not None and entry[0] is column else None


def supplied_dedupe(x: np.ndarray, y: np.ndarray) -> "tuple[np.ndarray, np.ndarray] | None":
    """The deduplicated ``(x, y)`` the device computed for exactly these arrays (the supplied sorted pair), or ``None`` when they are not the supplied ones.

    For a column without exact duplicate pairs the host rule is to return its own arrays, which is what this returns."""
    entry = getattr(_SUPPLY, "entry", None)
    if entry is None or entry[1][0] is not x or entry[1][1] is not y:
        return None
    dedup = entry[1].dedup
    return (x, y) if dedup is None else dedup


def mode() -> str:
    """``MLFRAME_MDLP_GPU_PRESORT``: ``0`` off (the default), ``1`` on whenever the shape allows, ``auto`` on from ``MIN_ROWS_PER_WORKER`` rows for each edge-search worker.

    Off by default because of the whole fit, not the stage: on an idle device the fused version takes 2.87 s for 160 columns of 500000 rows against 3.16 s for the host passes (process CPU 11.0 against 15.8 s, equal
    edges), but a paired 500000 x 1000 ALL fit with the device presort took 628.5 s against 482.7 s without it (same selection; peak RSS growth +0.5 GB, the device out of memory on the first blocks while the cascades hold
    it), see audits/2026-11-23/_TRACKER.md, PLAN-SPEED-29."""
    return os.environ.get("MLFRAME_MDLP_GPU_PRESORT", "0").strip().lower()


def applicable(n_rows: int, labels: np.ndarray, workers: int = 8) -> bool:
    """Whether a block of ``n_rows`` rows with these labels should go through the device: integer labels in the keyable range, enough rows for ``workers`` edge-search workers, a usable device, not switched off."""
    setting = mode()
    if setting == "0" or (setting != "1" and n_rows < MIN_ROWS_PER_WORKER * max(1, int(workers))):
        return False
    if labels.dtype.kind not in "iu" or labels.size == 0 or int(labels.min()) < 0 or int(labels.max()) > _MAX_LABEL:
        return False
    try:
        from ._gpu_policy import cuda_available_for_run

        return bool(cuda_available_for_run())
    except Exception as exc:  # the policy module is advice: a failure means no device for this call
        log_throttle(logger, "mdlp_gpu_presort.policy", logging.INFO, "device presort of the edge search unavailable (%s: %s)", type(exc).__name__, exc)
        return False


def _eligible(column: np.ndarray) -> "np.ndarray | None":
    """The column as float32 when it is finite and float32 in value (a float32 column, or a double one that is exactly one), else ``None``."""
    if column.dtype == np.float32:
        x32 = np.ascontiguousarray(column)
    elif column.dtype == np.float64:  # Intentional float64: the test of the incoming column's dtype, no double array is made
        from .discretization._quantile_radix import _f64_as_f32_njit

        exact, x32 = _f64_as_f32_njit(np.ascontiguousarray(column))
        if not exact:
            return None
    else:
        return None
    return x32 if bool(np.isfinite(x32).all()) else None


def presort_block(columns: "list[np.ndarray]", labels: np.ndarray) -> "list[Presorted | None]":
    """One :class:`Presorted` per column of ``columns`` (``None`` for a column the device path does not take, and for the whole block on any device failure).

    ``labels`` are the encoded target codes of the rows (a non-negative integer vector, ``applicable`` checks); the columns share their row count."""
    out: "list[Presorted | None]" = [None] * len(columns)
    n = int(labels.shape[0])
    picked = [(i, x32) for i, c in enumerate(columns) if int(c.shape[0]) == n for x32 in [_eligible(c)] if x32 is not None]
    if not picked or n < 2:
        return out
    try:
        with _DEVICE_LOCK:
            block = _presort_on_device([x for _, x in picked], labels)
    except Exception as exc:  # an out-of-memory or a device fault: the block is sorted on the host, as before
        log_throttle(logger, "mdlp_gpu_presort.failed", logging.WARNING, "device presort of the edge search failed (%s: %s); the host sorts this block", type(exc).__name__, exc)
        return out
    results = _split(*block, labels.dtype)  # the host copies of the per-column arrays are made after the device is given back to the next block
    for (i, _), result in zip(picked, results):
        out[i] = _with_dtype(result, columns[i].dtype)
    return out


def _with_dtype(result: Presorted, dtype: np.dtype) -> Presorted:
    """The distinct values in the dtype of the column (the host rule: ``uniq.astype(finite.dtype)``)."""
    return Presorted(result.xs, result.ys, result.uniq.astype(dtype, copy=False), result.counts, result.dedup)


def _presort_on_device(x32s: "list[np.ndarray]", labels: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple | None]":
    """The sort, the runs and the duplicate pairs of the float32 columns ``x32s`` on the device (the caller holds the device lock): ``(xs, ys, uniq, counts, runs per column, dedup)`` as host arrays for :func:`_split`.

    Two fused kernels and one sort of 64-bit keys: ``mdlp_keys`` builds the keys, the sort orders every column at once (stable: the row is the last key bits), ``mdlp_unpack`` reads value, label and
    run start off the sorted keys; the runs, their counts and the number of runs of each column follow from the start flags."""
    import cupy as cp
    import cupyx
    from pyutilz.performance.gpu_hw_launch import block_size_for

    from ._device_sort_safe import device_sort

    k, n = len(x32s), int(labels.shape[0])
    row_bits = max(1, (n - 1).bit_length())
    col_bits = max(1, (k - 1).bit_length())
    if 32 + row_bits + col_bits > 64:
        raise ValueError(f"a block of {k} columns of {n} rows does not fit a 64-bit key")
    keys_kernel, unpack_kernel = _kernels(cp, labels.dtype)
    staged = cupyx.empty_pinned((k, n), np.float32)
    for r, x in enumerate(x32s):
        staged[r] = x
    bits = cp.asarray(staged).view(cp.uint32)
    key = cp.empty(k * n, dtype=cp.uint64)
    threads = block_size_for(keys_kernel, work_items=n, max_width=1024)
    keys_kernel((-(-n // threads), k), (threads,), (bits, key, np.int64(n), np.int32(row_bits)))
    del bits
    sk = device_sort(key)
    del key
    d_labels = cp.asarray(labels)
    xs32 = cp.empty(k * n, dtype=cp.float32)
    ys = cp.empty(k * n, dtype=d_labels.dtype)
    start = cp.empty(k * n, dtype=cp.uint8)
    threads = block_size_for(unpack_kernel, work_items=n, max_width=1024)
    unpack_kernel((-(-n // threads), k), (threads,), (sk, d_labels, xs32, ys, start, np.int64(n), np.int32(row_bits)))
    del sk
    flat_start = start.view(cp.bool_)
    run_starts = cp.flatnonzero(flat_start)
    counts = cp.diff(cp.concatenate([run_starts, cp.asarray([k * n], dtype=run_starts.dtype)])).astype(cp.int32)
    uniq32 = xs32[run_starts]
    runs_per_column = cp.count_nonzero(start.reshape(k, n), axis=1)
    dedup = _duplicate_pairs(cp, device_sort, flat_start, ys.reshape(k, n), run_starts, uniq32, k, n) if int(run_starts.size) < k * n else None
    xs_host = _download(xs32.astype(cp.float64).reshape(k, n))  # Intentional float64: the host passes hand the MDLP search the values in double precision
    ys_host = _download(ys.reshape(k, n))
    uniq_host, counts_host, per_column = _download(uniq32), _download(counts), _download(runs_per_column)
    del xs32, ys
    cp.get_default_memory_pool().free_all_blocks()
    return xs_host, ys_host, uniq_host, counts_host, per_column, dedup


def _duplicate_pairs(cp: Any, device_sort: Any, flat_start: Any, ys: Any, run_starts: Any, uniq32: Any, k: int, n: int) -> "tuple | None":
    """``(kept pairs: x (float32), y codes, pairs per column)`` of the columns' exact duplicate pairs collapsed, in (value, label) order; ``None`` when no column has a duplicate pair."""
    run_id = cp.cumsum(flat_start.astype(cp.int64)) - 1
    ybits = max(1, int(ys.max()).bit_length())
    key2 = (run_id.astype(cp.uint64) << np.uint64(ybits)) | ys.ravel().astype(cp.uint64)
    sk2 = device_sort(key2)
    del key2
    fresh = cp.ones(k * n, dtype=cp.bool_)
    fresh[1:] = sk2[1:] != sk2[:-1]
    kept_idx = cp.flatnonzero(fresh)
    if int(kept_idx.size) == k * n:
        return None
    kept = sk2[kept_idx]
    run_of_kept = (kept >> np.uint64(ybits)).astype(cp.int64)
    y_kept = (kept & np.uint64((1 << ybits) - 1)).astype(ys.dtype)
    x_kept = uniq32[run_of_kept]
    column_of_kept = run_starts[run_of_kept] // n
    pairs_per_column = cp.bincount(column_of_kept, minlength=k)
    return cp.asnumpy(x_kept), cp.asnumpy(y_kept), cp.asnumpy(pairs_per_column)


def _split(xs: np.ndarray, ys: np.ndarray, uniq: np.ndarray, counts: np.ndarray, per_column: np.ndarray, dedup: "tuple | None", label_dtype: np.dtype) -> "list[Presorted]":
    """The per-column :class:`Presorted` from the block arrays: the runs and the kept pairs are cut by the per-column counts."""
    out: "list[Presorted]" = []
    n = xs.shape[1]
    run_edges = np.concatenate([[0], np.cumsum(per_column)])
    pair_edges = None if dedup is None else np.concatenate([[0], np.cumsum(dedup[2])])
    for c in range(xs.shape[0]):
        pair = None
        if dedup is not None and pair_edges is not None and int(dedup[2][c]) != n:
            lo, hi = int(pair_edges[c]), int(pair_edges[c + 1])
            pair = (dedup[0][lo:hi].astype(np.float64), dedup[1][lo:hi].astype(label_dtype, copy=False))  # Intentional float64: the host dedupe returns the double-precision values
        out.append(Presorted(xs[c], ys[c].astype(label_dtype, copy=False), uniq[run_edges[c] : run_edges[c + 1]], counts[run_edges[c] : run_edges[c + 1]].astype(np.int64), pair))
    return out
