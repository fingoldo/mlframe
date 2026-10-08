"""Analytic large-n noise gate fed from device-resident candidate codes.

At large n the pair search's permutation gate is replaced by the analytic G-test: the gate needs each candidate column's observed plug-in MI and its occupied
bin count. The dispatcher used to get both on the host - it copied the (n, K) codes back from the device, ran the CPU observed-MI kernel over them (about 1.5 s
per call at n=30000, K~2000, and 27 of the 36 s of a 100k-row strict fit) and counted occupied bins with an njit pass. When the producer kept the codes
resident, both reductions are one pass each on the device, and only the (K,) MI vector and the (K,) occupied counts cross the bus.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Codes per device block (n * k elements) of the generic fallback: the MI kernel widens a block to int64, so this bounds that transient to ~256 MB.
_BLOCK_ELEMENTS = 1 << 25

# Shared-memory budget per block for the fused kernel's joint histograms (bytes); 48 KB is available on every supported card.
_SHARED_BUDGET = 48 * 1024

# FUSED observed-MI + occupied-bin kernel over a resident int8 (n, K) code matrix. One block owns TILE adjacent columns: its threads are laid out as
# (row group, column) so a warp reads TILE consecutive code bytes of one row (coalesced) and the joint histograms live in shared memory, so the matrix is
# read once, as int8, and nothing wider than the (K,) results is written. The plug-in MI formula and its terms are those of ``mi_from_codes``
# (``_fe_batched_mi``); the occupied-bin count is the number of non-empty x rows of the histogram, which equals ``_occupied_bins_per_col``.
_FUSED_SRC = r"""
extern "C" __global__
void analytic_obs_mi_bins_i8(const signed char* __restrict__ codes, const int* __restrict__ y, const long long n, const int K,
                             const int Kx, const int Ky, const int tile, const double inv_n,
                             double* __restrict__ mi_out, int* __restrict__ bins_out) {
    extern __shared__ int sh[];                       // (tile, Kx*Ky) joint histograms
    const int M = Kx * Ky;
    const long long col0 = (long long)blockIdx.x * tile;
    const int cj = threadIdx.x % tile;                // column within the tile
    const int rg = threadIdx.x / tile;                // row group
    const int ngroups = blockDim.x / tile;
    for (int s = threadIdx.x; s < tile * M; s += blockDim.x) sh[s] = 0;
    __syncthreads();
    const long long col = col0 + cj;
    if (col < K) {
        for (long long i = rg; i < n; i += ngroups) {
            int cx = (int)codes[i * (long long)K + col];
            int cy = y[i];
            atomicAdd(&sh[cj * M + cx * Ky + cy], 1);
        }
    }
    __syncthreads();
    if (rg == 0 && col < K) {
        const int* h = sh + cj * M;
        double mi = 0.0;
        int occupied = 0;
        for (int xx = 0; xx < Kx; ++xx) {
            long long rx = 0;
            for (int yy = 0; yy < Ky; ++yy) rx += h[xx * Ky + yy];
            if (rx == 0) continue;
            occupied += 1;
            double px = (double)rx * inv_n;
            for (int yy = 0; yy < Ky; ++yy) {
                long long nxy = h[xx * Ky + yy];
                if (nxy == 0) continue;
                long long ry = 0;
                for (int xx2 = 0; xx2 < Kx; ++xx2) ry += h[xx2 * Ky + yy];
                double pxy = (double)nxy * inv_n;
                double py = (double)ry * inv_n;
                mi += pxy * log(pxy / (px * py));
            }
        }
        mi_out[col] = mi > 0.0 ? mi : 0.0;
        bins_out[col] = occupied;
    }
}
"""

_FUSED_KERNEL = None


def _fused_kernel():
    """Lazily compiled fused kernel."""
    global _FUSED_KERNEL
    if _FUSED_KERNEL is None:
        import cupy as cp

        _FUSED_KERNEL = cp.RawKernel(_FUSED_SRC, "analytic_obs_mi_bins_i8")
    return _FUSED_KERNEL


def _tile_for(m_cells: int) -> int:
    """Columns per block so the tile's histograms fit the shared budget: the largest of 32/16/8/4 that fits, or 0 when even 4 columns do not."""
    for tile in (32, 16, 8, 4):
        if tile * m_cells * 4 <= _SHARED_BUDGET:
            return tile
    return 0


def _fused_observed_mi_and_bins(device_codes: Any, yc: np.ndarray, ky: int) -> "Optional[tuple[np.ndarray, np.ndarray]]":
    """Fused single-pass observed MI + occupied bins over an int8 ``(n, K)`` resident matrix, or ``None`` when the shape / dtype is outside what the kernel
    covers (the caller then takes the generic path)."""
    import cupy as cp

    if device_codes.dtype != cp.int8 or device_codes.ndim != 2:
        return None
    n, k_total = int(device_codes.shape[0]), int(device_codes.shape[1])
    lo, hi = (int(v) for v in cp.stack([device_codes.min(), device_codes.max()]).get())  # one read: every code must sit inside its histogram
    if lo < 0:
        return None
    kx = hi + 1
    ky_eff = max(int(ky), int(yc.max()) + 1 if yc.size else 1)
    tile = _tile_for(kx * ky_eff)
    if tile == 0:
        return None
    codes = cp.ascontiguousarray(device_codes)
    from mlframe.feature_selection.filters._fe_resident_operands import resident_operand

    y32 = resident_operand(yc.astype(np.int32), "analytic_gate_y32", dtype=np.int32).ravel()
    mi = cp.empty(k_total, dtype=cp.float64)
    bins = cp.empty(k_total, dtype=cp.int32)
    threads = tile * max(1, 256 // tile)
    blocks = (k_total + tile - 1) // tile
    _fused_kernel()(
        (blocks,), (threads,),
        (codes, y32, np.int64(n), np.int32(k_total), np.int32(kx), np.int32(ky_eff), np.int32(tile), np.float64(1.0 / float(max(1, n))), mi, bins),
        shared_mem=tile * kx * ky_eff * 4,
    )
    return np.asarray(mi.get(), dtype=np.float64), np.asarray(bins.get(), dtype=np.int64)


def resident_observed_mi_and_bins(device_codes: Any, classes_y: np.ndarray, by: int) -> "tuple[np.ndarray, np.ndarray]":
    """Observed plug-in MI (nats) and occupied-bin count of every column of the resident ``(n, K)`` code matrix, against the target codes ``classes_y``.

    An int8 matrix goes through the fused single-pass kernel; anything else (or a histogram too large for shared memory) takes the generic block-wise path.
    Both return host ``(K,)`` float64 / int64 arrays - the same quantities the CPU ``npermutations=0`` kernel and ``_occupied_bins_per_col`` give.
    """
    import cupy as cp

    from mlframe.feature_selection.filters._fe_batched_mi import binned_mi_from_codes_gpu

    yc = np.ascontiguousarray(classes_y, dtype=np.int64).ravel()
    try:
        fused = _fused_observed_mi_and_bins(device_codes, yc, int(by))
    except Exception as e:  # best-effort: the generic block-wise path computes the same observed MI and occupied bins
        logger.debug("fused observed-MI kernel failed, using the generic path: %s", e)
        fused = None
    if fused is not None:
        return fused
    n, k_total = int(device_codes.shape[0]), int(device_codes.shape[1])
    step = max(1, min(k_total, _BLOCK_ELEMENTS // max(1, n)))
    observed = np.empty(k_total, dtype=np.float64)
    bins = np.empty(k_total, dtype=np.int64)
    for start in range(0, k_total, step):
        block = device_codes[:, start : start + step]
        width = int(block.shape[1])
        observed[start : start + width] = binned_mi_from_codes_gpu(block, yc, ky=int(by), codes_trusted=True)
        wide = block.astype(cp.int64, copy=False)
        m = int(wide.max()) + 1 if wide.size else 1
        counts = cp.bincount((wide * width + cp.arange(width, dtype=cp.int64)[None, :]).ravel(), minlength=m * width)
        bins[start : start + width] = (counts.reshape(m, width) > 0).sum(axis=0).get().astype(np.int64)
    return observed, bins


def resident_analytic_gate(device_codes: Any, classes_y: np.ndarray, by: int, n_rows: int, min_nonzero_confidence: float) -> Optional[np.ndarray]:
    """``fe_mi[K]`` of the analytic noise gate computed from resident codes, or ``None`` on any device fault so the caller keeps the host path."""
    try:
        from .._analytic_mi_null import analytic_batch_noise_gate

        observed, bins = resident_observed_mi_and_bins(device_codes, classes_y, by)
        return analytic_batch_noise_gate(None, observed, classes_y, int(n_rows), float(min_nonzero_confidence), bx_per_col=bins, by=int(by))
    except Exception as e:
        logger.debug("resident analytic noise gate failed, using the host path: %s", e)
        return None
