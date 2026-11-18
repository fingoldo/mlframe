"""Compile points of the basis-evaluation kernels of ``_gpu_resident_basis`` (clenshaw, robust scale, heavy-tail statistics, preprocessing, column statistics, absolute correlation).

The lazy getters of that module (which is at the module-size limit) call these factories, so the register-pressure registry checks exactly the code a fit compiles.
"""

from __future__ import annotations

import threading

from typing import Any

import numpy as np

from ._kernel_types import render

_ABS_CORR_SRC = render(r"""
#define REAL_T {REAL}
// coalescing-ok: the scalar path below 32 columns, measured faster than the tiled one at this width (tracker GC11-40: width 4 0.218 against 0.583 ms, width 16 0.314 against 0.574 ms at 100000 rows); the dispatch already takes the tiled kernel from 32 columns
extern "C" __global__
void abs_corr(const REAL_T* __restrict__ cand, const REAL_T* __restrict__ yc, const REAL_T* __restrict__ mu, const long long n,
              const int m, const {ACC} Syy, {ACC}* __restrict__ out) {
    // Centred accumulation: yc is the centred target and mu the per-column means, so the sums below are sum((v-mean)^2) and
    // sum((v-mean)*yc) directly. The raw-power-sum form n*sum(v^2) - (sum v)^2 cancels to rounding noise once a column's mean is far
    // above its spread, and then reported a corrupted |corr| or the -1.0 skip sentinel for a perfectly usable column.
    int c = blockIdx.x;
    if (c >= m) return;
    __shared__ {ACC} sh[2];                       // sum_cc, sum_cy
    int tid = threadIdx.x, nt = blockDim.x;
    if (tid < 2) sh[tid] = 0.0;
    __syncthreads();
    {ACC} pcc = 0.0, pcy = 0.0;
    {ACC} muc = ({ACC})mu[c];
    for (long long i = tid; i < n; i += nt) {
        {ACC} d = ({ACC})cand[i * (long long)m + c] - muc;
        pcc += d * d; pcy += d * ({ACC})yc[i];
    }
    atomicAdd(&sh[0], pcc); atomicAdd(&sh[1], pcy);
    __syncthreads();
    if (tid == 0) {
        {ACC} scc = sh[0], scy = sh[1];
        {ACC} den = sqrt(scc * Syy);
        {ACC} corr = den > 1e-300 ? fabs(scy / den) : 0.0;
        bool ok = isfinite(corr) && (sqrt(scc) > 1e-12);   // same degenerate test as the cupy chain: cn = sqrt(sum cc^2) > 1e-12
        out[c] = ok ? corr : -1.0;
    }
}
""")
_ABS_CORR_KERNELS: dict = {}
_ABS_CORR_KERNELS_LOCK = threading.RLock()  # serialises the get-or-build of the cache above (one build per key even with concurrent fits)


def _get_abs_corr_kernel(real_t: str = "double"):
    """Lazily compile-and-cache the ``abs_corr`` RawKernel for the input element type ``real_t`` (``"float"`` or ``"double"``); the sums accumulate in double either way."""
    with _ABS_CORR_KERNELS_LOCK:
        with _basis()._KERNEL_CACHE_LOCK:
            kern = _ABS_CORR_KERNELS.get(real_t)
            if kern is None:
                import cupy as cp
                kern = kernel_abs_corr(cp, np.float32 if real_t == "float" else np.float64)  # Intentional float64: the device kernel is built in the data's own precision: float64 data keeps double
                _ABS_CORR_KERNELS[real_t] = kern
        return kern


def _basis() -> Any:
    """The basis module (imported on first use: it imports this module for its carved abs_corr kernel)."""
    from . import _gpu_resident_basis

    return _gpu_resident_basis


def _src(source: str, real: Any) -> str:
    """``source`` with the element type of the registry dtype ``real`` filled in."""
    return source.replace("{REAL}", "float" if np.dtype(real) == np.float32 else "double")


def kernel_clenshaw(cp: Any, real: Any) -> Any:
    """``clenshaw_basis`` for values of dtype ``real``; compile point of the getter and of the register-pressure registry."""
    return cp.RawKernel(_src(_basis()._CLENSHAW_SRC, real), "clenshaw_basis")


def kernel_robust_scale(cp: Any, real: Any) -> Any:
    """``robust_scale_combine`` for values of dtype ``real``; compile point of the getter and of the register-pressure registry."""
    return cp.RawKernel(_src(_basis()._ROBUST_SCALE_COMBINE_SRC, real), "robust_scale_combine")


def kernel_heavy_tail(cp: Any, real: Any) -> Any:
    """``heavy_tail_stats_tiled`` for values of dtype ``real``; compile point of the getter and of the register-pressure registry."""
    return cp.RawKernel(_src(_basis()._HEAVY_TAIL_STATS_TILED_SRC, real), "heavy_tail_stats_tiled")


def kernel_heavy_tail_finish(cp: Any, real: Any) -> Any:
    """``heavy_tail_finish`` for values of dtype ``real``; compile point of the getter and of the register-pressure registry."""
    return cp.RawKernel(_src(_basis()._HEAVY_TAIL_STATS_TILED_SRC, real), "heavy_tail_finish")


def kernel_preprocess(cp: Any, real: Any) -> Any:
    """``basis_preprocess`` for values of dtype ``real``; compile point of the getter and of the register-pressure registry."""
    return cp.RawKernel(_src(_basis()._BASIS_PREPROCESS_SRC, real), "basis_preprocess")


def kernel_preprocess_nr(cp: Any, real: Any) -> Any:
    """``basis_preprocess_nr`` for values of dtype ``real``; compile point of the getter and of the register-pressure registry."""
    return cp.RawKernel(_src(_basis()._BASIS_PREPROCESS_NR_SRC, real), "basis_preprocess_nr")


def kernel_meanstd(cp: Any, real: Any) -> Any:
    """``col_meanstd_tiled`` for values of dtype ``real``; compile point of the getter and of the register-pressure registry."""
    return cp.RawKernel(_src(_basis()._COL_MEANSTD_TILED_SRC, real), "col_meanstd_tiled")


def kernel_minmax(cp: Any, real: Any) -> Any:
    """``col_minmax_tiled`` for values of dtype ``real``; compile point of the getter and of the register-pressure registry."""
    return cp.RawKernel(_src(_basis()._COL_MINMAX_TILED_SRC, real), "col_minmax_tiled")


def kernel_abs_corr(cp: Any, real: Any) -> Any:
    """``abs_corr`` for values of dtype ``real``; compile point of the getter and of the register-pressure registry."""
    return cp.RawKernel(_src(_ABS_CORR_SRC, real), "abs_corr")
