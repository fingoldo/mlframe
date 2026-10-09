"""Row-wise ``argmax`` over a handful of resident operand columns in one launch, and a per-column finiteness memo.

``cp.argmax(cp.stack(columns, axis=1), axis=1)`` interleaves the columns into an ``(n, m)`` matrix, runs cupy's segmented cub reduction over ``n`` segments of ``m`` elements (about
15 ms per call at one million rows, for ``m = 3``) and the caller then casts the index to float64. The kernel here reads the ``m`` columns once and writes the float64 index of the
first maximum; ``>`` is strict, so ties resolve to the lowest operand index exactly as ``argmax`` does. NaN rows must be excluded by the caller (the host check that already guards the
device path), since ``argmax`` and a comparison chain treat NaN differently.

The finiteness of a host operand column is asked about once per triple that contains it; a column does not change between triples, so the answer is remembered per array object.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any, Optional, Sequence

import numpy as np

_SRC = r"""
extern "C" __global__ void row_argmax(const double* __restrict__ M, const long long n, const int m, double* __restrict__ out) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        double best = M[i];
        int bi = 0;
        for (int q = 1; q < m; ++q) {
            const double v = M[(long long)q * n + i];
            if (v > best) { best = v; bi = q; }
        }
        out[i] = (double)bi;
    }
}
"""
_KERNEL: Optional[Any] = None
_LOCK = threading.Lock()
_FINITE: dict = {}  # id(array) -> (weakref to the array, all finite)
_FINITE_MAX = 256


def row_argmax_cm(cp: Any, columns: Sequence[Any]) -> Any:
    """Float64 ``(n,)`` index of the first maximum across ``columns`` (resident float64 ``(n,)`` arrays)."""
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(_SRC, "row_argmax")
    n = int(columns[0].shape[0])
    M = cp.ascontiguousarray(cp.stack(columns, axis=0))  # (m, n)
    out = cp.empty(n, dtype=cp.float64)
    blocks = max(1, min(4096, (n + 255) // 256))
    _KERNEL((blocks,), (256,), (M, np.int64(n), np.int32(len(columns)), out))
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
    except TypeError:  # not weak-referenceable: no memo
        return flag
    with _LOCK:
        if len(_FINITE) >= _FINITE_MAX:
            for stale in [k for k, (r, _f) in _FINITE.items() if r() is None]:
                del _FINITE[stale]
            if len(_FINITE) >= _FINITE_MAX:
                _FINITE.clear()
        _FINITE[key] = (ref, flag)
    return flag
