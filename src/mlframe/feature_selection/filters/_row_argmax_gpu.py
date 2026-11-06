"""Row-wise ``argmax`` over a handful of resident operand columns in one launch, and a per-column finiteness memo.

``cp.argmax(cp.stack(columns, axis=1), axis=1)`` interleaves the columns into an ``(n, m)`` matrix, runs cupy's segmented cub reduction over ``n`` segments of ``m`` elements (about
15 ms per call at one million rows, for ``m = 3``) and the caller then casts the index to double. The kernel here reads the ``m`` columns once and writes the double index of the
first maximum, reading the columns in their own element type (``{REAL}``: the crit dtype is float32 by default, and a kernel that read float32 columns as ``double`` ran past the end of the stack and faulted with an illegal address); ``>`` is strict, so ties resolve to the lowest operand index exactly as ``argmax`` does. NaN rows must be excluded by the caller (the host check that already guards the
device path), since ``argmax`` and a comparison chain treat NaN differently.

The finiteness of a host operand column is asked about once per triple that contains it; a column does not change between triples, so the answer is remembered per array object.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any, Sequence

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

from ._kernel_types import render
import logging

logger = logging.getLogger(__name__)

_SRC = r"""
extern "C" __global__ void row_argmax(const {REAL}* __restrict__ M, const long long n, const int m, {ACC}* __restrict__ out) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        {REAL} best = M[i];
        int bi = 0;
        for (int q = 1; q < m; ++q) {
            const {REAL} v = M[(long long)q * n + i];
            if (v > best) { best = v; bi = q; }
        }
        out[i] = ({ACC})bi;
    }
}
"""
_KERNELS: dict = {}  # element type name ("float" | "double") -> compiled kernel, one per type
_LOCK = threading.Lock()
_FINITE: dict = {}  # id(array) -> (weakref to the array, all finite)
_FINITE_MAX = 256


def row_argmax_cm(cp: Any, columns: Sequence[Any]) -> Any:
    """Double ``(n,)`` index of the first maximum across ``columns`` (resident float32 or double ``(n,)`` arrays; mixed types are promoted)."""
    n = int(columns[0].shape[0])
    M = cp.ascontiguousarray(cp.stack(columns, axis=0))  # (m, n)
    if M.dtype not in (cp.float32, cp.float64):  # Intentional float64: mixed float types are promoted to double
        M = M.astype(cp.float64)
    real = "float" if M.dtype == cp.float32 else "double"
    with _LOCK:
        kernel = _KERNELS.get(real)
        if kernel is None:
            kernel = _KERNELS[real] = cp.RawKernel(render(_SRC, np.float32 if real == "float" else np.float64), "row_argmax")  # Intentional float64: mixed float types are promoted to double
    out = cp.empty(n, dtype=cp.float64)  # Intentional float64: index output matches the caller's float64 cast contract
    threads = block_size_for(kernel, work_items=n)
    blocks = max(1, min(4096, (n + threads - 1) // threads))
    kernel((blocks,), (threads,), (M, np.int64(n), np.int32(len(columns)), out))
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
