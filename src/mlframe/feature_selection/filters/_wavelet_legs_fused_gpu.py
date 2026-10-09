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

import threading
from typing import Any, Optional

_LOCK = threading.Lock()
_MODULE: Optional[Any] = None

_SRC = r"""
// Leg value of row value z for level j, cell k: +1 left half, -1 right half, 0 outside the cell (also for z outside [0, 1) and NaN).
__device__ __forceinline__ int leg_value(const double z, const int j, const int k) {
    if (!(z >= 0.0 && z < 1.0)) return 0;
    const double t = z * (double)(1LL << j);
    const double c = floor(t);
    if ((int)c != k) return 0;
    return (t - c < 0.5) ? 1 : -1;
}
extern "C" __global__ void leg_supports(const double* __restrict__ z, const long long n, const int max_scale, int* __restrict__ counts) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const double zi = z[i];
        if (!(zi >= 0.0 && zi < 1.0)) continue;
        for (int j = 0; j <= max_scale; ++j) {
            const double t = zi * (double)(1LL << j);
            const double c = floor(t);
            const int leg = (1 << j) - 1 + (int)c;  // flat leg id: levels laid out one after the other
            const int sign_slot = (t - c < 0.5) ? 0 : 1;
            atomicAdd(&counts[2 * leg + sign_slot], 1);
        }
    }
}
extern "C" __global__ void leg_codes(const double* __restrict__ z, const long long n, const int* __restrict__ eligible_j, const int* __restrict__ eligible_k, const int n_legs,
                                     long long* __restrict__ tr, long long* __restrict__ va) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const double zi = z[i];
        const bool is_val = (i % 3) == 0;
        // position of the row within its split: validation rows are 0, 3, 6, ...; training rows are the rest, in order
        const long long pos = is_val ? i / 3 : i - (i / 3 + 1);
        long long* out = (is_val ? va : tr) + pos * n_legs;
        for (int e = 0; e < n_legs; ++e) out[e] = (long long)(leg_value(zi, eligible_j[e], eligible_k[e]) + 1);
    }
}
"""


def _module(cp):
    """Compile the kernels once."""
    global _MODULE
    with _LOCK:
        if _MODULE is None:
            _MODULE = cp.RawModule(code=_SRC, options=("-std=c++14",))
        return _MODULE


def leg_code_matrices(cp: Any, z_g: Any, max_scale: int, min_half_rows: int) -> Optional[tuple]:
    """``(metas, tr_mat, va_mat)`` for the legs with at least ``min_half_rows`` rows on each sign, or ``None`` when the fused path does not apply.

    ``metas`` is the list of ``(j, k)`` in the order the per-leg loop produced (level by level, cell by cell); ``tr_mat`` / ``va_mat`` are resident C-contiguous int64 ``(rows, legs)``
    code matrices (``leg + 1``) of the training rows and of the ``i % 3 == 0`` validation rows. ``([], None, None)`` when no leg is eligible."""
    import numpy as np

    if max_scale < 0 or max_scale > 12:
        return None
    n = int(z_g.shape[0])
    z = cp.ascontiguousarray(z_g, dtype=cp.float64)
    n_legs_all = (1 << (int(max_scale) + 1)) - 1
    mod = _module(cp)
    counts = cp.zeros(2 * n_legs_all, dtype=cp.int32)
    blocks = max(1, min(4096, (n + 255) // 256))
    mod.get_function("leg_supports")((blocks,), (256,), (z, cp.int64(n), cp.int32(max_scale), counts))
    host = cp.asnumpy(counts).reshape(n_legs_all, 2)
    elig = np.flatnonzero((host[:, 0] >= min_half_rows) & (host[:, 1] >= min_half_rows))
    if elig.size == 0:
        return [], None, None
    levels = np.floor(np.log2(elig + 1)).astype(np.int64)
    cells = elig - ((1 << levels) - 1)
    n_va = (n + 2) // 3
    n_tr = n - n_va
    tr = cp.empty((n_tr, int(elig.size)), dtype=cp.int64)
    va = cp.empty((n_va, int(elig.size)), dtype=cp.int64)
    ej = cp.asarray(levels.astype(np.int32))
    ek = cp.asarray(cells.astype(np.int32))
    mod.get_function("leg_codes")((blocks,), (256,), (z, cp.int64(n), ej, ek, cp.int32(elig.size), tr, va))
    return [(int(j), int(k)) for j, k in zip(levels, cells)], tr, va
