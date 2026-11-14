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
from .._kernel_types import render

logger = logging.getLogger(__name__)

# Codes per device block (n * k elements) of the generic fallback: the MI kernel widens a block to int64, so this bounds that transient to ~256 MB.
_BLOCK_ELEMENTS = 1 << 25

# Threads per block of the fused kernel (rounded to whole tiles): 128 measured best with four loads in flight, 20.4 to 19.2 ms at 100000 x 6000.
_FUSED_THREADS = 128

# Shared-memory budget per block for the fused kernel's joint histograms (bytes); 48 KB is available on every supported card.
_SHARED_BUDGET = 48 * 1024

# FUSED observed-MI + occupied-bin kernel over a resident int8 (n, K) code matrix. One block owns TILE adjacent columns: its threads are laid out as
# (row group, column) so a warp reads TILE consecutive code bytes of one row (coalesced) and the joint histograms live in shared memory, so the matrix is
# read once, as int8, and nothing wider than the (K,) results is written. The plug-in MI formula and its terms are those of ``mi_from_codes``
# (``_fe_batched_mi``); the occupied-bin count is the number of non-empty x rows of the histogram, which equals ``_occupied_bins_per_col``.
_FUSED_SRC = render(r"""
extern "C" __global__
void analytic_obs_mi_bins_i8(const signed char* __restrict__ codes, const int* __restrict__ y, const long long n, const int K,
                             const int Kx, const int Ky, const int tile, const {ACC} inv_n,
                             {ACC}* __restrict__ mi_out, int* __restrict__ bins_out) {
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
        // four independent loads in flight per thread: one load per iteration left the card at 15 to 19 GB/s of 112 (latency bound), four give 27 to 31
        long long i = rg;
        for (; i + 3LL * ngroups < n; i += 4LL * ngroups) {
            const int cx0 = (int)codes[i * (long long)K + col];
            const int cy0 = y[i];
            const int cx1 = (int)codes[(i + ngroups) * (long long)K + col];
            const int cy1 = y[i + ngroups];
            const int cx2 = (int)codes[(i + 2LL * ngroups) * (long long)K + col];
            const int cy2 = y[i + 2LL * ngroups];
            const int cx3 = (int)codes[(i + 3LL * ngroups) * (long long)K + col];
            const int cy3 = y[i + 3LL * ngroups];
            atomicAdd(&sh[cj * M + cx0 * Ky + cy0], 1);
            atomicAdd(&sh[cj * M + cx1 * Ky + cy1], 1);
            atomicAdd(&sh[cj * M + cx2 * Ky + cy2], 1);
            atomicAdd(&sh[cj * M + cx3 * Ky + cy3], 1);
        }
        for (; i < n; i += ngroups) {
            const int cx = (int)codes[i * (long long)K + col];
            const int cy = y[i];
            atomicAdd(&sh[cj * M + cx * Ky + cy], 1);
        }
    }
    __syncthreads();
    if (rg == 0 && col < K) {
        const int* h = sh + cj * M;
        {ACC} mi = 0.0;
        int occupied = 0;
        for (int xx = 0; xx < Kx; ++xx) {
            long long rx = 0;
            for (int yy = 0; yy < Ky; ++yy) rx += h[xx * Ky + yy];
            if (rx == 0) continue;
            occupied += 1;
            {ACC} px = ({ACC})rx * inv_n;
            for (int yy = 0; yy < Ky; ++yy) {
                long long nxy = h[xx * Ky + yy];
                if (nxy == 0) continue;
                long long ry = 0;
                for (int xx2 = 0; xx2 < Kx; ++xx2) ry += h[xx2 * Ky + yy];
                {ACC} pxy = ({ACC})nxy * inv_n;
                {ACC} py = ({ACC})ry * inv_n;
                mi += pxy * log(pxy / (px * py));
            }
        }
        mi_out[col] = mi > 0.0 ? mi : 0.0;
        bins_out[col] = occupied;
    }
}
""")

# One pass over the int8 matrix for its smallest and largest code (the fused kernel needs the level count and a non-negativity check before it launches): cupy's min() and max() are two passes over
# the whole matrix, 10 ms of a 44 ms call at 100000 x 6000. Each thread reads four bytes at a time (an int), folds them, and one atomic per warp merges the results.
_MINMAX_SRC = r"""
extern "C" __global__
void minmax_i8(const signed char* __restrict__ a, const long long total, int* __restrict__ out) {
    int lo = 127, hi = -128;
    const long long total4 = total >> 2;
    const int* __restrict__ a4 = (const int*)a;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total4; i += stride) {
        const int w = a4[i];
        const int b0 = (signed char)(w & 0xff), b1 = (signed char)((w >> 8) & 0xff), b2 = (signed char)((w >> 16) & 0xff), b3 = (signed char)((w >> 24) & 0xff);
        lo = min(lo, min(min(b0, b1), min(b2, b3)));
        hi = max(hi, max(max(b0, b1), max(b2, b3)));
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (long long i = total4 << 2; i < total; ++i) {
            const int v = (int)a[i];
            lo = min(lo, v);
            hi = max(hi, v);
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        lo = min(lo, __shfl_down_sync(0xffffffffu, lo, off));
        hi = max(hi, __shfl_down_sync(0xffffffffu, hi, off));
    }
    if ((threadIdx.x & 31) == 0) {
        atomicMin(&out[0], lo);
        atomicMax(&out[1], hi);
    }
}
"""

_MINMAX_KERNEL = None


def _minmax_i8(cp: Any, codes: Any) -> "tuple[int, int]":
    """``(smallest, largest)`` code of a contiguous int8 matrix in one device pass and one 8-byte read."""
    global _MINMAX_KERNEL
    from pyutilz.performance.gpu_hw_launch import block_size_for

    if _MINMAX_KERNEL is None:
        _MINMAX_KERNEL = cp.RawKernel(_MINMAX_SRC, "minmax_i8")
    out = cp.asarray(np.array([127, -128], dtype=np.int32))
    total = int(codes.size)
    threads = block_size_for(_MINMAX_KERNEL, work_items=max(1, total // 4))
    blocks = max(1, min(4096, (total // 4 + threads - 1) // threads))
    _MINMAX_KERNEL((blocks,), (threads,), (codes, np.int64(total), out))
    lo, hi = (int(v) for v in out.get())
    return lo, hi


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


# A wide block reads 128 adjacent code bytes of a row, the width of a memory line, where a 32-column tile asks for a 32-byte piece of each row from DRAM pages 1500 bytes apart. Measured on the
# card with equal outputs, 100000 rows x 1500 columns x 20 cells: 4.58 -> 3.89 ms (1.18x); x 6000 columns 19.1 -> 18.8 ms; narrower matrices keep the 32-column tile.
_WIDE_TILE = 128
_WIDE_TILE_MIN_COLUMNS = 512
_WIDE_TILE_THREADS = 1024


def _launch_shape(k_total: int, m_cells: int) -> "tuple[int, int]":
    """``(columns per block, threads per block)`` of the fused kernel: the 128-column tile for a wide matrix whose histograms fit, else the 32-or-narrower tile of ``_tile_for``."""
    if k_total >= _WIDE_TILE_MIN_COLUMNS and _WIDE_TILE * m_cells * 4 <= _SHARED_BUDGET:
        return _WIDE_TILE, _WIDE_TILE_THREADS
    tile = _tile_for(m_cells)
    return tile, tile * max(1, _FUSED_THREADS // tile) if tile else 0


def _fused_observed_mi_and_bins(device_codes: Any, yc: np.ndarray, ky: int) -> "Optional[tuple[np.ndarray, np.ndarray]]":
    """Fused single-pass observed MI + occupied bins over an int8 ``(n, K)`` resident matrix, or ``None`` when the shape / dtype is outside what the kernel
    covers (the caller then takes the generic path)."""
    import cupy as cp

    if device_codes.dtype != cp.int8 or device_codes.ndim != 2:
        return None
    n, k_total = int(device_codes.shape[0]), int(device_codes.shape[1])
    codes = cp.ascontiguousarray(device_codes)
    lo, hi = _minmax_i8(cp, codes) if codes.size else (0, 0)  # one pass, one read: every code must sit inside its histogram
    if lo < 0:
        return None
    kx = hi + 1
    if yc.size and int(yc.min()) < 0:
        # A negative target code would index the shared histogram below zero (into a neighbouring column's cells); the generic path handles it.
        return None
    ky_eff = max(int(ky), int(yc.max()) + 1 if yc.size else 1)
    tile, threads = _launch_shape(k_total, kx * ky_eff)
    if tile == 0:
        return None
    from mlframe.feature_selection.filters._fe_resident_operands import resident_operand

    y32 = resident_operand(yc.astype(np.int32), "analytic_gate_y32", dtype=np.int32).ravel()
    mi = cp.empty(k_total, dtype=cp.float64)  # Intentional float64: MI statistic output compared at full precision
    bins = cp.empty(k_total, dtype=cp.int32)
    blocks = (k_total + tile - 1) // tile
    _fused_kernel()(
        (blocks,), (threads,),
        (codes, y32, np.int64(n), np.int32(k_total), np.int32(kx), np.int32(ky_eff), np.int32(tile), np.float64(1.0 / float(max(1, n))), mi, bins),  # Intentional float64: the CUDA kernel signature declares double, so this must stay float64
        shared_mem=tile * kx * ky_eff * 4,
    )
    # one device-to-host copy for both outputs (each blocking read is a round trip of a few hundred microseconds on WDDM, and this runs once per pair); bin counts are exact in double
    both = cp.concatenate([mi, bins.astype(cp.float64)]).get()  # Intentional float64: bins are small integers, exact in double; one buffer with the MI values
    return np.asarray(both[:k_total], dtype=np.float64), both[k_total:].astype(np.int64)  # Intentional float64: MI statistic output compared at full precision


def resident_observed_mi_and_bins(device_codes: Any, classes_y: np.ndarray, by: int) -> "tuple[np.ndarray, np.ndarray]":
    """Observed plug-in MI (nats) and occupied-bin count of every column of the resident ``(n, K)`` code matrix, against the target codes ``classes_y``.

    An int8 matrix goes through the fused single-pass kernel; anything else (or a histogram too large for shared memory) takes the generic block-wise path.
    Both return host ``(K,)`` double / int64 arrays - the same quantities the CPU ``npermutations=0`` kernel and ``_occupied_bins_per_col`` give.
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
    observed = np.empty(k_total, dtype=np.float64)  # Intentional float64: MI statistic output compared at full precision
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
        from .._analytic_mi_null import analytic_batch_noise_gate, pair_gate_min_n

        observed, bins = resident_observed_mi_and_bins(device_codes, classes_y, by)
        return analytic_batch_noise_gate(None, observed, classes_y, int(n_rows), float(min_nonzero_confidence), bx_per_col=bins, by=int(by), min_n=pair_gate_min_n())
    except Exception as e:
        logger.warning("resident analytic noise gate failed, using the host path: %s", e)
        return None
