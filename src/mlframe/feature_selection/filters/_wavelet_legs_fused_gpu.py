"""Dyadic-Haar leg code matrices for the wavelet leg selection, built in two kernel launches.

The per-leg cupy build (:func:`_wavelet_basis_fe_batched._select_wavelet_legs_batched_device`) made, for each of the ``2**(max_scale+1) - 1`` candidate legs, a nested ``where``,
two support counts, ``+1``, an ``astype`` and two row gathers: ~16 launches per leg, ~366 per call, dozens of calls per fit. A leg ``psi_{j,k}`` is ``+1`` on ``[k/2^j,
(k+1/2)/2^j)``, ``-1`` on ``[(k+1/2)/2^j, (k+1)/2^j)`` and ``0`` elsewhere, and ``z * 2^j`` is exact (a power of two), so the cell and the half are read off ``floor`` and the
fractional part with exactly the comparisons the ``where`` chain makes.

* ``leg_supports``: per leg, how many rows are ``+1`` and how many ``-1`` (the eligibility counts).
* ``leg_codes``: for the eligible legs only, ``leg + 1`` written straight into the train and validation code matrices (the ``i % 3 == 0`` rows are validation), already in the
  row order and ``(rows, legs)`` layout the batched MI consumes.
"""

from __future__ import annotations

from pyutilz.performance.gpu_hw_launch import block_size_for

from typing import Any, Optional

from mlframe.feature_selection.filters._raw_module_cache import RawModuleCache
from mlframe.feature_selection.filters._fe_gpu_batch._devices import crit_float_dtype

_SRC = r"""
// Leg value of row value z for level j, cell k: +1 left half, -1 right half, 0 outside the cell (also for z outside [0, 1) and NaN).
__device__ __forceinline__ int leg_value(const {ACC} z, const int j, const int k) {
    if (!(z >= 0.0 && z < 1.0)) return 0;
    const {ACC} t = z * ({ACC})(1LL << j);
    const {ACC} c = floor(t);
    if ((int)c != k) return 0;
    return (t - c < 0.5) ? 1 : -1;
}
extern "C" __global__ void leg_supports(const {REAL}* __restrict__ z, const long long n, const int max_scale, int* __restrict__ counts) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const {ACC} zi = z[i];
        if (!(zi >= 0.0 && zi < 1.0)) continue;
        for (int j = 0; j <= max_scale; ++j) {
            const {ACC} t = zi * ({ACC})(1LL << j);
            const {ACC} c = floor(t);
            const int leg = (1 << j) - 1 + (int)c;  // flat leg id: levels laid out one after the other
            const int sign_slot = (t - c < 0.5) ? 0 : 1;
            atomicAdd(&counts[2 * leg + sign_slot], 1);
        }
    }
}
extern "C" __global__ void leg_codes(const {REAL}* __restrict__ z, const long long n, const int* __restrict__ eligible_j, const int* __restrict__ eligible_k, const int n_legs,
                                     long long* __restrict__ tr, long long* __restrict__ va) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const {ACC} zi = z[i];
        const bool is_val = (i % 3) == 0;
        // position of the row within its split: validation rows are 0, 3, 6, ...; training rows are the rest, in order
        const long long pos = is_val ? i / 3 : i - (i / 3 + 1);
        long long* out = (is_val ? va : tr) + pos * n_legs;
        for (int e = 0; e < n_legs; ++e) out[e] = (long long)(leg_value(zi, eligible_j[e], eligible_k[e]) + 1);
    }
}
"""


_SRC_ACC = _SRC.replace("{ACC}", "double")  # Intentional float64: the leg cell boundaries are compared in double for every row type, matching the host reference
_MODULES_BY_REAL = {"double": RawModuleCache(_SRC_ACC.replace("{REAL}", "double")), "float": RawModuleCache(_SRC_ACC.replace("{REAL}", "float"))}


def leg_code_matrices(cp: Any, z_g: Any, max_scale: int, min_half_rows: int) -> Optional[tuple]:
    """``(metas, tr_mat, va_mat)`` for the legs with at least ``min_half_rows`` rows on each sign, or ``None`` when the fused path does not apply.

    ``metas`` is the list of ``(j, k)`` in the order the per-leg loop produced (level by level, cell by cell); ``tr_mat`` / ``va_mat`` are resident C-contiguous int64 ``(rows, legs)``
    code matrices (``leg + 1``) of the training rows and of the ``i % 3 == 0`` validation rows. ``([], None, None)`` when no leg is eligible."""
    import numpy as np

    if max_scale < 0 or max_scale > 12:
        return None
    n = int(z_g.shape[0])
    z = cp.ascontiguousarray(z_g, dtype=crit_float_dtype())
    n_legs_all = (1 << (int(max_scale) + 1)) - 1
    mod = _MODULES_BY_REAL["float" if z.dtype.itemsize == 4 else "double"].get(cp)  # the kernels read ``z`` at its own dtype and compare in double precision
    counts = cp.zeros(2 * n_legs_all, dtype=cp.int32)
    supports = mod.get_function("leg_supports")
    support_threads = block_size_for(supports, work_items=n)
    blocks = max(1, min(4096, (n + support_threads - 1) // support_threads))
    supports((blocks,), (support_threads,), (z, cp.int64(n), cp.int32(max_scale), counts))
    host = cp.asnumpy(counts).reshape(n_legs_all, 2)
    elig = np.flatnonzero((host[:, 0] >= min_half_rows) & (host[:, 1] >= min_half_rows))
    if elig.size == 0:
        return [], None, None
    levels = np.floor(np.log2(elig + 1)).astype(np.int64)
    cells = elig - ((1 << levels) - 1)
    n_va = (n + 2) // 3
    n_tr = n - n_va
    # Cushion-check BEFORE the tr/va code matrices: without it, a device pool already pinned at its limit
    # fails this call's allocation on EVERY call with the SAME OutOfMemoryError (the GPU11-13/14 pattern).
    # The caller's own try/except (_wavelet_basis_fe_batched.py) returns None (host fallback) on ANY
    # exception here, so declining early is a pure latency win, not a new failure mode.
    from ._fe_gpu_vram import fe_gpu_has_vram_cushion

    if not fe_gpu_has_vram_cushion((n_tr + n_va) * int(elig.size) * 8):
        return None
    tr = cp.empty((n_tr, int(elig.size)), dtype=cp.int64)
    va = cp.empty((n_va, int(elig.size)), dtype=cp.int64)
    ej = cp.asarray(levels.astype(np.int32))
    ek = cp.asarray(cells.astype(np.int32))
    codes_kernel = mod.get_function("leg_codes")
    code_threads = block_size_for(codes_kernel, work_items=n)
    codes_kernel((max(1, min(4096, (n + code_threads - 1) // code_threads)),), (code_threads,), (z, cp.int64(n), ej, ek, cp.int32(elig.size), tr, va))
    return [(int(j), int(k)) for j, k in zip(levels, cells)], tr, va
