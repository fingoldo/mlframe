"""Miller-Madow MI of a COLUMN-MAJOR candidate block, with the coalesced X read.

:func:`_fe_batched_mi.binned_mm_mi_from_values_gpu` takes an ``(n, K)`` row-major matrix and its kernel reads ``X[i*K + c]``: one block per column, consecutive threads stride by ``K``
doubles, so the dominant loop runs at a fraction of memory bandwidth (the plug-in MI kernels were moved to column-major for exactly this reason; the Miller-Madow ones were not).
This twin takes the ``(K, n)`` C-order block the column-major generators (``fused_gen_cm``) already produce, so no transpose is needed anywhere on the path - the radix edges read the
same buffer - and the kernel reads ``X[c*n + i]``. Edge dedup, the histogram and the MI arithmetic are the original kernel's text with only that index changed, so the result
is the same number.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._fe_batched_mi import (
    _MI_FROM_CODES_MAX_SHARED,
    _MI_MM_FROM_VALUES_NEK_F32_SRC,
    _MI_MM_FROM_VALUES_NEK_SRC,
    _get_dedup_edges_kernel,
)
from ._fe_batched_mi_guards import _assert_codes_in_range

_ROW_MAJOR_READ = "X[i * (long long)K + c]"
_COLUMN_MAJOR_READ = "X[(long long)c * n + i]"

_MM_NEK_CM_SRC = _MI_MM_FROM_VALUES_NEK_SRC.replace(_ROW_MAJOR_READ, _COLUMN_MAJOR_READ).replace("void mi_mm_from_values_nek(", "void mi_mm_from_values_nek_cm(")
_MM_NEK_CM_F32_SRC = _MI_MM_FROM_VALUES_NEK_F32_SRC.replace(_ROW_MAJOR_READ, _COLUMN_MAJOR_READ).replace("void mi_mm_from_values_nek_f32(", "void mi_mm_from_values_nek_cm_f32(")
_KERNELS: dict = {}


def _kernel(cp, f32: bool):
    """Lazily compiled column-major Miller-Madow kernel (f64 or f32 X; edges are always f64)."""
    key = "f32" if f32 else "f64"
    ker = _KERNELS.get(key)
    if ker is None:
        ker = cp.RawKernel(_MM_NEK_CM_F32_SRC if f32 else _MM_NEK_CM_SRC, "mi_mm_from_values_nek_cm_f32" if f32 else "mi_mm_from_values_nek_cm")
        _KERNELS[key] = ker
    return ker


def binned_mm_mi_from_cm_gpu(
    x_cm: Any, interior_edges: Any, y_codes: Any, nbins: int, ky: int, h_y: float, k_y: int, codes_trusted: bool = False, return_device: bool = False
) -> Any:
    """Miller-Madow marginal MI of each row of the ``(K, n)`` C-order block ``x_cm`` (column ``c`` of the candidate matrix is row ``c``) against ``y_codes``.

    Same contract as ``binned_mm_mi_from_values_gpu`` for the edges (``(nbins-1, K)``), ``ky`` / ``h_y`` / ``k_y`` and the return (a host ``(K,)`` array, or the resident one with
    ``return_device``); ``None`` when the ``nbins * ky`` histogram does not fit in shared memory."""
    import cupy as cp

    is_f32 = getattr(x_cm, "dtype", None) == cp.float32
    Xc = cp.ascontiguousarray(x_cm if is_f32 else x_cm.astype(cp.float64, copy=False))
    K, n = int(Xc.shape[0]), int(Xc.shape[1])
    Ky = int(ky)
    if int(nbins) * Ky * 4 > _MI_FROM_CODES_MAX_SHARED:
        return None
    E = cp.ascontiguousarray(interior_edges.astype(cp.float64, copy=False))
    if isinstance(y_codes, cp.ndarray):
        yv = y_codes.astype(cp.int64, copy=False).ravel()
    else:
        from ._fe_resident_operands import resident_operand

        yv = resident_operand(np.asarray(y_codes).ravel(), "binned_mm_ycodes", dtype=np.int64)
    _assert_codes_in_range(yv, Ky, "binned_mm_mi_from_cm_gpu y codes", codes_trusted)
    ne = int(E.shape[0])
    cmin = cp.ascontiguousarray(Xc.min(axis=1).astype(cp.float64, copy=False))
    cmax = cp.ascontiguousarray(Xc.max(axis=1).astype(cp.float64, copy=False))
    Ec = cp.empty((ne + 1, K), dtype=cp.float64)  # the dedup kernel's transient trailing row, as in the row-major twin
    ne_k = cp.empty(K, dtype=cp.int32)
    threads = 256
    _get_dedup_edges_kernel()(((K + threads - 1) // threads,), (threads,), (E, cmin, cmax, np.int32(ne), np.int32(K), Ec, ne_k))
    mi_out = cp.empty(K, dtype=cp.float64)
    _kernel(cp, is_f32)(
        (K,), (256,),
        (Xc, Ec, yv, np.int64(n), np.int32(K), np.int32(int(nbins)), np.int32(Ky), np.float64(1.0 / float(max(1, n))), np.float64(float(h_y)), np.int32(int(k_y)), ne_k, mi_out),
        shared_mem=int(nbins) * Ky * 4,
    )
    return mi_out if return_device else cp.asnumpy(mi_out)
