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

logger = logging.getLogger(__name__)

#: Columns sorted together (the keys of a block are ``columns x rows`` 64-bit values, 32 MB for 8 columns of 500000 rows, and a sort needs about three times that).
BLOCK_COLUMNS = 8
#: Rows from which the device wins against the eight-thread host (the block costs a transfer in and out).
MIN_ROWS = 100_000
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
    """``MLFRAME_MDLP_GPU_PRESORT``: ``0`` off (the default), ``1`` on whenever the shape allows, ``auto`` on from ``MIN_ROWS`` rows.

    Off by default: on 160 columns of 500000 rows the edge search took 3.82 s with the device presort against 3.04 s without it (all cores free, edges equal; process CPU 11.4 against 17.4 s), so
    what it saves is CPU on a saturated host, worth about 4 s of a whole 1000-column fit and not measurable against the 15 s noise of its wall (audits/2026-11-23/_TRACKER.md, PLAN-SPEED-28)."""
    return os.environ.get("MLFRAME_MDLP_GPU_PRESORT", "0").strip().lower()


def applicable(n_rows: int, labels: np.ndarray) -> bool:
    """Whether a block of ``n_rows`` rows with these labels should go through the device: integer labels in the keyable range, enough rows, a usable device, not switched off."""
    setting = mode()
    if setting == "0" or (setting != "1" and n_rows < MIN_ROWS):
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
            results = _presort_on_device([x for _, x in picked], labels)
    except Exception as exc:  # an out-of-memory or a device fault: the block is sorted on the host, as before
        log_throttle(logger, "mdlp_gpu_presort.failed", logging.WARNING, "device presort of the edge search failed (%s: %s); the host sorts this block", type(exc).__name__, exc)
        return out
    for (i, _), result in zip(picked, results):
        out[i] = _with_dtype(result, columns[i].dtype)
    return out


def _with_dtype(result: Presorted, dtype: np.dtype) -> Presorted:
    """The distinct values in the dtype of the column (the host rule: ``uniq.astype(finite.dtype)``)."""
    return Presorted(result.xs, result.ys, result.uniq.astype(dtype, copy=False), result.counts, result.dedup)


def _presort_on_device(x32s: "list[np.ndarray]", labels: np.ndarray) -> "list[Presorted]":
    """The sort, the runs and the duplicate pairs of the float32 columns ``x32s`` on the device (the caller holds the device lock)."""
    import cupy as cp

    from ._device_sort_safe import device_sort

    k, n = len(x32s), int(labels.shape[0])
    row_bits = max(1, (n - 1).bit_length())
    col_bits = max(1, (k - 1).bit_length())
    if 32 + row_bits + col_bits > 64:
        raise ValueError(f"a block of {k} columns of {n} rows does not fit a 64-bit key")
    matrix = np.empty((k, n), dtype=np.float32)
    for r, x in enumerate(x32s):
        matrix[r] = x
    bits = cp.asarray(matrix).view(cp.uint32)
    ordered = bits ^ ((bits >> 31) * np.uint32(0xFFFFFFFF) | np.uint32(0x80000000))  # unsigned order of the float order; -0.0 stays below +0.0 as in the host sort
    key = (cp.arange(k, dtype=cp.uint64)[:, None] << np.uint64(32 + row_bits)) | (ordered.astype(cp.uint64) << np.uint64(row_bits)) | cp.arange(n, dtype=cp.uint64)[None, :]
    del bits, ordered
    sk = device_sort(key.ravel())
    del key
    rows = (sk & np.uint64((1 << row_bits) - 1)).astype(cp.int64).reshape(k, n)
    top = ((sk >> np.uint64(row_bits)) & np.uint64(0xFFFFFFFF)).astype(cp.uint32).reshape(k, n)
    del sk
    restored = cp.where((top & np.uint32(0x80000000)) != 0, top & np.uint32(0x7FFFFFFF), ~top)
    xs32 = restored.view(cp.float32)
    ys = cp.asarray(labels)[rows]
    del rows, top, restored
    start = cp.ones((k, n), dtype=cp.bool_)
    start[:, 1:] = xs32[:, 1:] != xs32[:, :-1]
    flat_start = start.ravel()
    run_starts = cp.flatnonzero(flat_start)
    counts = cp.diff(cp.concatenate([run_starts, cp.asarray([k * n], dtype=run_starts.dtype)])).astype(cp.int32)
    uniq32 = xs32.ravel()[run_starts]
    runs_per_column = cp.bincount(run_starts // n, minlength=k)
    dedup = _duplicate_pairs(cp, device_sort, flat_start, ys, run_starts, uniq32, k, n) if int(run_starts.size) < k * n else None
    xs_host = cp.asnumpy(xs32.astype(cp.float64))  # Intentional float64: the host passes hand the MDLP search the values in double precision
    ys_host = cp.asnumpy(ys)
    uniq_host, counts_host, per_column = cp.asnumpy(uniq32), cp.asnumpy(counts), cp.asnumpy(runs_per_column)
    del xs32, ys
    cp.get_default_memory_pool().free_all_blocks()
    return _split(xs_host, ys_host, uniq_host, counts_host, per_column, dedup, labels.dtype)


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
