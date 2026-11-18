"""Row-wise ``argmax`` over a handful of resident operand columns in one launch, and a per-column finiteness memo.

``cp.argmax(cp.stack(columns, axis=1), axis=1)`` interleaves the columns into an ``(n, m)`` matrix, runs cupy's segmented cub reduction over ``n`` segments of ``m`` elements (about
15 ms per call at one million rows, for ``m = 3``) and the caller then casts the index to double. The kernel here reads the ``m`` columns once and writes the double index of the
first maximum, reading the columns in their own element type (``{REAL}``: the crit dtype is float32 by default, and a kernel that read float32 columns as ``double`` ran past the end of the stack and faulted with an illegal address); ``>`` is strict, so ties resolve to the lowest operand index exactly as ``argmax`` does. NaN rows must be excluded by the caller (the host check that already guards the
device path), since ``argmax`` and a comparison chain treat NaN differently.

The kernel reads the ``m`` resident columns in place through a device array of their pointers (no ``(m, n)`` stacked copy) and is row-streamed: a grid-stride loop over row blocks of at most 4096 blocks, so the only allocation is the ``(n,)`` index. ``row_argmax_host`` is the numba twin
(``prange`` over rows, a tuple of columns, no stack) and ``row_argmax_on_device`` the per-host choice between them (kernel_tuner ``row_argmax_backend``, a sweep over the row count on resident columns; ``MLFRAME_ROW_ARGMAX_DEVICE=0|1`` forces a side).

The finiteness of a host operand column is asked about once per triple that contains it; a column does not change between triples, so the answer is remembered per array object.
"""

from __future__ import annotations

import os
import threading
import weakref
from typing import Any, Sequence

import numpy as np
from numba import njit, prange

from pyutilz.performance.gpu_hw_launch import block_size_for

from mlframe.utils.log_throttle import log_throttle

from ._kernel_types import render
from ._ktc_fallback import host_device_sweep, rows_fallback
import logging

logger = logging.getLogger(__name__)

_SRC = r"""
extern "C" __global__ void row_argmax(const {REAL}* const* __restrict__ cols, const long long n, const int m, {ACC}* __restrict__ out) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        {REAL} best = cols[0][i];
        int bi = 0;
        for (int q = 1; q < m; ++q) {
            const {REAL} v = cols[q][i];
            if (v > best) { best = v; bi = q; }
        }
        out[i] = ({ACC})bi;
    }
}
"""


@njit(cache=True, parallel=True, nogil=True)
def _row_argmax_njit(cols: tuple, out: np.ndarray) -> None:
    """``out[i]`` = index of the first maximum of ``cols[0][i], cols[1][i], ...`` (``cols`` a tuple of equal-length same-dtype columns: no stacked copy; the host twin of the ``row_argmax`` kernel)."""
    m = len(cols)
    for i in prange(out.shape[0]):
        best = cols[0][i]
        bi = 0
        for q in range(1, m):
            if cols[q][i] > best:
                best = cols[q][i]
                bi = q
        out[i] = bi


def _read_only_view(a: np.ndarray) -> np.ndarray:
    """A read-only view of ``a``: numba types a read-only and a writable array differently, and a tuple of mixed ones cannot be indexed by a loop variable (``TypingError: getitem(Tuple(readonly array, ..., array), int64)``,
    which skipped the whole row-argmax FE of a fit whose columns were a mix of frame views and fresh arrays); read-only views give every column of the tuple one type."""
    v: np.ndarray = a.view()
    v.flags.writeable = False
    return v


def row_argmax_host(columns: Sequence[np.ndarray]) -> np.ndarray:
    """Host form: the parallel numba kernel over the columns (promoted to one common dtype); double precision ``(n,)`` index of the first maximum."""
    from mlframe._numba_parallel_guard import parallel_kernel_entry

    common = np.result_type(*[np.asarray(c).dtype for c in columns])
    cols = tuple(_read_only_view(np.ascontiguousarray(c, dtype=common)) for c in columns)
    out = np.empty(cols[0].shape[0], dtype=np.float64)  # Intentional float64: index output matches the caller's float64 cast contract
    with parallel_kernel_entry():
        _row_argmax_njit(cols, out)
    return out


DEVICE_FALLBACK_ROWS = 200_000
_SWEEP_ROWS = [20_000, 100_000, 500_000, 2_000_000]
_SWEEP_COLS = 4
_SALT = 1


def _sweep_inputs(dims: dict) -> tuple:
    """``(columns,)``: ``_SWEEP_COLS`` float32 columns of ``dims["n_samples"]`` rows."""
    rng = np.random.default_rng(0)
    n = int(dims["n_samples"])
    return (rng.normal(size=(_SWEEP_COLS, n)).astype(np.float32),)


def _sample(index: Any) -> Any:
    """Every 13th element of an index vector: the equivalence check of the sweep, taken on the side the vector lives so no full ``(n,)`` copy or reduction enters the timing (the unit test compares the whole vector)."""
    return index[::13]


def _sweep_host(columns: np.ndarray) -> np.ndarray:
    """The host form on the sweep inputs, sampled."""
    sampled: np.ndarray = _sample(row_argmax_host(list(columns)))
    return sampled


_SWEEP_LOCK = threading.Lock()
_SWEEP_RESIDENT: dict = {}  # id(host columns) -> (host columns, their device copies): the gate's operands are uploaded once and shared (``resident_operand``), so the device form is timed on resident columns


def _sweep_device(columns: np.ndarray) -> np.ndarray:
    """The device form on the sweep inputs: the kernel over columns already on the device (the upload is paid once per operand, not per triple); the index stays on the device (the gate keeps it there), its sample is copied back."""
    import cupy as cp

    with _SWEEP_LOCK:
        hit = _SWEEP_RESIDENT.get(id(columns))
        if hit is None or hit[0] is not columns:
            _SWEEP_RESIDENT.clear()
            hit = _SWEEP_RESIDENT[id(columns)] = (columns, [cp.asarray(c) for c in columns])
    sampled: np.ndarray = cp.asnumpy(_sample(row_argmax_cm(cp, hit[1])))
    return sampled


_fallback_choice = rows_fallback(__name__)

try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    _SPEC: Any = kernel_tuner(
        kernel_name="row_argmax_backend",
        variant_fns=(),  # the two backends are timed by the sweep callable
        tuner=host_device_sweep(__name__, _sweep_host, _sweep_device, rtol=0.0, atol=0.0),
        axes={"n_samples": list(_SWEEP_ROWS)},
        fallback=_fallback_choice,
        gpu_capable=True,
        salt=_SALT,
        cli_label="row_argmax_backend",
    )
except Exception as e:  # best-effort: without a tuner entry the row-count fallback decides
    logger.warning("kernel_tuner spec build for row_argmax_backend failed: %s", e)
    _SPEC = None


def row_argmax_on_device(n_rows: int) -> bool:
    """Whether the row argmax of host-resident columns of ``n_rows`` rows runs on the device: ``MLFRAME_ROW_ARGMAX_DEVICE=0|1`` forces a side, else the per-host kernel-tuning cache, else the row mark.

    Resident (already device) columns always take the device kernel: no transfer is paid there."""
    forced = os.environ.get("MLFRAME_ROW_ARGMAX_DEVICE", "").strip()
    if forced in ("0", "1"):
        return forced == "1"
    if _SPEC is None:
        return int(n_rows) >= DEVICE_FALLBACK_ROWS
    try:
        return str(_SPEC.choose(n_samples=int(n_rows))) == "device"
    except Exception as exc:
        log_throttle(logger, "row_argmax.choose_failed", logging.WARNING, "row_argmax_backend: KTC choose() failed, using the row-count fallback: %s", exc)
        return int(n_rows) >= DEVICE_FALLBACK_ROWS


_KERNELS: dict = {}  # element type name ("float" | "double") -> compiled kernel, one per type
_LOCK = threading.Lock()
_FINITE: dict = {}  # id(array) -> (weakref to the array, all finite)
_FINITE_MAX = 256


def kernel(cp: Any, real: Any) -> Any:
    """``row_argmax`` for resident columns of dtype ``real`` (float32 or double precision); compile point of :func:`row_argmax_cm` and of the register-pressure registry."""
    name = "float" if np.dtype(real) == np.float32 else "double"
    with _LOCK:
        ker = _KERNELS.get(name)
        if ker is None:
            ker = _KERNELS[name] = cp.RawKernel(render(_SRC, np.float32 if name == "float" else np.float64), "row_argmax")  # Intentional float64: mixed float types are promoted to double
    return ker


def row_argmax_cm(cp: Any, columns: Sequence[Any]) -> Any:
    """Double ``(n,)`` index of the first maximum across ``columns`` (resident float32 or double ``(n,)`` arrays; mixed types are promoted)."""
    n = int(columns[0].shape[0])
    dtypes = {c.dtype for c in columns}
    if len(dtypes) != 1 or next(iter(dtypes)) not in (cp.float32, cp.float64):  # Intentional float64: mixed float types are promoted to double
        columns = [c.astype(cp.float64) for c in columns]  # Intentional float64: mixed float types are promoted to double
    columns = [cp.ascontiguousarray(c) for c in columns]
    M = columns[0]
    ker = kernel(cp, M.dtype)
    out = cp.empty(n, dtype=cp.float64)  # Intentional float64: index output matches the caller's float64 cast contract
    threads = block_size_for(ker, work_items=n)
    blocks = max(1, min(4096, (n + threads - 1) // threads))
    ptrs = cp.asarray(np.array([int(c.data.ptr) for c in columns], dtype=np.uint64))  # the columns are read in place: no (m, n) stacked copy
    ker((blocks,), (threads,), (ptrs, np.int64(n), np.int32(len(columns)), out))
    return out


def all_finite_cached(arr: np.ndarray) -> bool:
    """``np.isfinite(arr).all()``, remembered per array object so a column shared by many triples is scanned once (the entry is dropped with the array)."""
    key = id(arr)
    with _LOCK:
        hit = _FINITE.get(key)
        if hit is not None and hit[0]() is arr:
            return bool(hit[1])
        flag = bool(np.all(np.isfinite(arr)))
        try:
            ref = weakref.ref(arr)
        except TypeError as _exc:  # not weak-referenceable: no memo
            logger.debug("all_finite_cached fell back to its default after %s", type(_exc).__name__)
            return flag
        if len(_FINITE) >= _FINITE_MAX:
            for stale in [k for k, (r, _f) in _FINITE.items() if r() is None]:
                del _FINITE[stale]
            if len(_FINITE) >= _FINITE_MAX:
                _FINITE.clear()
        _FINITE[key] = (ref, flag)
    return flag
