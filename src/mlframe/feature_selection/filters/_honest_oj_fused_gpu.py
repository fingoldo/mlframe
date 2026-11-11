"""Device winner pick of the fused optimal_joint honest floor (``_honest_oj_fused``): the step that is ~90% of the CPU kernel.

Per permutation the CPU kernel scores every (fold, re-binned column, candidate count) cell with a plug-in MI of the cell's fixed validation codes against
the permuted fold target, then picks per column the candidate with the largest mean fold MI. Here the joint histogram of every (cell, permutation) is built
on the device and its MI summed with the loops of ``_plug_in_mi_njit`` (same order, double precision; only the device ``log`` may differ from the host's in
the last ulp, so a pick can flip only on a tie to the ulp). The mean over folds and the first-max pick run on the device in the CPU order; the pool's
corrected MI under the picked codes stays on the CPU (``_pool_block``, a tenth of the kernel).

Two histogram routes:

* binary target (the common case): every bin of a cell is a bitmask over its validation rows (built once, the codes do not move) and the permuted target
  is a bitmask too; the ones of a bin are ``popcount(mask AND target)``, its zeros the fixed bin count minus the ones. 32 rows per instruction, no atomics.
* any other target: one CUDA block per (cell, permutation) scatters the rows into per-warp shared histograms with atomics.

Off when no usable GPU, when ``MLFRAME_HONEST_OJ_GPU=0``, when the scorer's tuned choice is the host, or when the codes do not fit the VRAM budget; any
device fault returns ``None`` and the CPU kernel runs.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for
from numba import njit, prange

from ._numba_utils_cross_module import _permuted_codes, _pool_max
from ._kernel_types import render
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

_KERNEL = render(r"""
__device__ {ACC} cell_mi(const int* hist, int n, int k_x, int k_y)
{
    // the loops of _plug_in_mi_njit (miller_madow=False), same order, {ACC} precision
    {ACC} mi = 0.0;
    if (n > 0 && k_y >= 1) {
        {ACC} px[64]; {ACC} py[64];
        for (int a = 0; a < k_x; ++a) px[a] = 0.0;
        for (int q = 0; q < k_y; ++q) py[q] = 0.0;
        for (int a = 0; a < k_x; ++a) for (int q = 0; q < k_y; ++q) { {ACC} v = ({ACC})hist[a * k_y + q]; px[a] += v; py[q] += v; }
        const {ACC} nf = ({ACC})n;
        for (int a = 0; a < k_x; ++a) {
            if (px[a] <= 0.0) continue;
            for (int q = 0; q < k_y; ++q) {
                {ACC} v = ({ACC})hist[a * k_y + q];
                if (v <= 0.0 || py[q] <= 0.0) continue;
                {ACC} p = v / nf;
                mi += p * log(p * nf / (px[a] * py[q] / nf));
            }
        }
    }
    return mi;
}

// one block per (cell, permutation); per-warp shared histograms with atomics
extern "C" __global__ void oj_cell_mi(const signed char* __restrict__ xb, const int* __restrict__ vy, const long long* __restrict__ kx,
                                      const long long* __restrict__ nv, const int* __restrict__ ky, int F, int PM, int NV, int KXMAX, int KYMAX,
                                      {ACC}* __restrict__ out)
{
    extern __shared__ unsigned char smem[];
    int* hist = (int*)smem;
    const int c = blockIdx.x;
    const int b = blockIdx.y;
    const int f = c / PM;
    const int k_x = (int)kx[c];
    const int k_y = ky[b * F + f];
    if (k_x == 0) { if (threadIdx.x == 0) out[(long long)b * F * PM + c] = nan(""); return; }
    const int H = k_x * k_y;
    const int W = blockDim.x >> 5;
    for (int t = threadIdx.x; t < W * H; t += blockDim.x) hist[t] = 0;
    __syncthreads();
    const int n = (int)nv[f];
    const signed char* x = xb + (long long)c * NV;
    const int* y = vy + ((long long)b * F + f) * NV;
    int* mine = hist + (threadIdx.x >> 5) * H;
    for (int i = threadIdx.x; i < n; i += blockDim.x) atomicAdd(&mine[(int)x[i] * k_y + y[i]], 1);
    __syncthreads();
    for (int t = threadIdx.x; t < H; t += blockDim.x) { int s = 0; for (int w = 0; w < W; ++w) s += hist[w * H + t]; hist[t] = s; }
    __syncthreads();
    if (threadIdx.x != 0) return;
    out[(long long)b * F * PM + c] = cell_mi(hist, n, k_x, k_y);
}

// private 16-bit counters per thread, laid out [bin][thread]: no atomics; valid while k_x * k_y <= 64 and every thread sees at most 65535 rows
extern "C" __global__ void oj_cell_mi_private(const signed char* __restrict__ xb, const int* __restrict__ vy, const long long* __restrict__ kx,
                                              const long long* __restrict__ nv, const int* __restrict__ ky, int F, int PM, int NV, int KXMAX, int KYMAX,
                                              {ACC}* __restrict__ out)
{
    extern __shared__ unsigned char smem[];
    unsigned short* cnt = (unsigned short*)smem;
    __shared__ int hist[64];
    const int c = blockIdx.x;
    const int b = blockIdx.y;
    const int f = c / PM;
    const int k_x = (int)kx[c];
    const int k_y = ky[b * F + f];
    const int T = blockDim.x;
    if (k_x == 0) { if (threadIdx.x == 0) out[(long long)b * F * PM + c] = nan(""); return; }
    const int H = k_x * k_y;
    for (int t = 0; t < H; ++t) cnt[t * T + threadIdx.x] = 0;
    const int n = (int)nv[f];
    const signed char* x = xb + (long long)c * NV;
    const int* y = vy + ((long long)b * F + f) * NV;
    for (int i = threadIdx.x; i < n; i += T) cnt[((int)x[i] * k_y + y[i]) * T + threadIdx.x] += 1;
    __syncthreads();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, W = T >> 5;
    for (int t = warp; t < H; t += W) {
        int s = 0;
        for (int q = lane; q < T; q += 32) s += cnt[t * T + q];
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
        if (lane == 0) hist[t] = s;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    out[(long long)b * F * PM + c] = cell_mi(hist, n, k_x, k_y);
}

// bit i of masks[(off[c] + x) * NW + i / 32]: row i of cell c's validation fold holds code x (built once: the codes do not move)
extern "C" __global__ void oj_build_masks(const signed char* __restrict__ xb, const long long* __restrict__ off, const long long* __restrict__ kx,
                                          const long long* __restrict__ nv, int C, int PM, int NV, int NW, unsigned int* __restrict__ masks)
{
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)C * NV) return;
    const int c = (int)(g / NV), i = (int)(g % NV);
    if (kx[c] == 0 || i >= (int)nv[c / PM]) return;
    atomicOr(&masks[(off[c] + (int)xb[(long long)c * NV + i]) * NW + (i >> 5)], 1u << (i & 31));
}

// bit i of ybits[(b * F + f) * NW + i / 32]: the permuted fold target is 1 at validation row i (binary targets)
extern "C" __global__ void oj_build_ybits(const int* __restrict__ vy, const long long* __restrict__ nv, int B, int F, int NV, int NW,
                                          unsigned int* __restrict__ ybits)
{
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)B * F * NW) return;
    const int w = (int)(g % NW);
    const long long bf = g / NW;
    const int f = (int)(bf % F);
    const int* y = vy + bf * NV;
    unsigned int m = 0u;
    for (int k = 0; k < 32; ++k) {
        const int i = (w << 5) + k;
        if (i < (int)nv[f] && y[i] == 1) m |= 1u << k;
    }
    ybits[g] = m;
}

// c1[(b * C + c) * KXMAX + x] = rows of bin x of cell c where the permuted target is 1: popcounts of mask AND target bits, PB permutations per block
#define PB 8
extern "C" __global__ void oj_count_ones(const unsigned int* __restrict__ masks, const unsigned int* __restrict__ ybits, const long long* __restrict__ off,
                                         const long long* __restrict__ kx, int B, int C, int F, int PM, int NW, int KXMAX, int* __restrict__ c1)
{
    __shared__ int sacc[PB];
    const int c = blockIdx.x;
    const int b0 = blockIdx.y * PB;
    const int f = c / PM;
    const int k_x = (int)kx[c];
    const int nb = min(PB, B - b0);
    for (int x = 0; x < k_x; ++x) {
        if (threadIdx.x < PB) sacc[threadIdx.x] = 0;
        __syncthreads();
        int acc[PB];
        for (int b = 0; b < PB; ++b) acc[b] = 0;
        const unsigned int* mrow = masks + (off[c] + x) * NW;
        for (int w = threadIdx.x; w < NW; w += blockDim.x) {
            const unsigned int m = mrow[w];
            if (m == 0u) continue;
            for (int b = 0; b < PB; ++b) if (b < nb) acc[b] += __popc(m & ybits[((long long)(b0 + b) * F + f) * NW + w]);
        }
        for (int b = 0; b < PB; ++b) {
            int v = acc[b];
            for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
            if ((threadIdx.x & 31) == 0 && b < nb) atomicAdd(&sacc[b], v);
        }
        __syncthreads();
        if (threadIdx.x < nb) c1[((long long)(b0 + threadIdx.x) * C + c) * KXMAX + x] = sacc[threadIdx.x];
        __syncthreads();
    }
}

// the cell MI of every (permutation, cell) from the ones and the fixed bin counts: hist[x, 1] = ones, hist[x, 0] = px - ones (exact integers)
extern "C" __global__ void oj_mi_from_ones(const int* __restrict__ c1, const int* __restrict__ px, const long long* __restrict__ kx,
                                           const long long* __restrict__ nv, const int* __restrict__ ky, int B, int C, int F, int PM, int KXMAX,
                                           {ACC}* __restrict__ out)
{
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)B * C) return;
    const int c = (int)(g % C), b = (int)(g / C);
    const int f = c / PM;
    const int k_x = (int)kx[c];
    if (k_x == 0) { out[g] = nan(""); return; }
    const int k_y = ky[b * F + f];
    int hist[64];
    for (int x = 0; x < k_x; ++x) {
        const int ones = c1[g * KXMAX + x];
        const int all = px[(long long)c * KXMAX + x];
        if (k_y >= 2) { hist[x * k_y] = all - ones; hist[x * k_y + 1] = ones; } else { hist[x * k_y] = all; }
    }
    out[g] = cell_mi(hist, (int)nv[f], k_x, k_y);
}
// pool MI of every (permutation, pool column, candidate count) cell, the loops of _mi_codes_njit with integer counts: ones from the popcounts,
// zeros = fixed bin count - ones; py = (rows - ones total, ones total) of the permutation
extern "C" __global__ void oj_pool_mi(const int* __restrict__ c1, const int* __restrict__ px, const long long* __restrict__ kx, const int* __restrict__ ones,
                                      int B, int C, int KXMAX, long long n, {ACC}* __restrict__ out)
{
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)B * C) return;
    const int c = (int)(g % C), b = (int)(g / C);
    const int k_x = (int)kx[c];
    if (k_x == 0) { out[g] = 0.0; return; }
    const long long py1 = ones[b], py0 = n - py1;
    {ACC} mi = 0.0;
    for (int a = 0; a < k_x; ++a) {
        const long long all = px[(long long)c * KXMAX + a];
        if (all == 0) continue;
        const long long n1 = c1[g * KXMAX + a];
        const long long n0 = all - n1;
        if (n0 != 0) mi += (({ACC})n0 / ({ACC})n) * log(({ACC})(n0 * n) / ({ACC})(all * py0));
        if (n1 != 0) mi += (({ACC})n1 / ({ACC})n) * log(({ACC})(n1 * n) / ({ACC})(all * py1));
    }
    out[g] = mi;
}
""")
_MAX_BINS = 64  # the kernels' per-thread marginal arrays; wider codes or targets stay on the CPU
_cache: dict = {}
_cache_lock = threading.Lock()


def _kernel(cp: Any, name: str = "oj_cell_mi") -> Any:
    """The compiled kernel ``name`` of :data:`_KERNEL`, once per process."""
    with _cache_lock:
        k = _cache.get(name)
        if k is None:
            mod = _cache.get("module")
            if mod is None:
                mod = cp.RawModule(code=_KERNEL)
                _cache["module"] = mod
            k = mod.get_function(name)
            _cache[name] = k
    return k


@njit(cache=True, parallel=True, nogil=True)
def _pool_block(perms, picks, yc, rows, pool_dep, pool_rc, pool_nb, pool_kxe, n_eff, out):
    """The pool's largest corrected MI under each permutation with the winners ``picks[b]`` (``_fused_block``'s pool step)."""
    for b in prange(perms.shape[0]):
        ycp, ny, ky_eff = _permuted_codes(perms[b], yc, rows)
        out[b] = _pool_max(picks[b], ycp, ny, ky_eff, pool_dep, pool_rc, pool_nb, pool_kxe, n_eff)


def _launch_flat(cp: Any, name: str, n_work: int, args: tuple) -> None:
    """Launch the one-thread-per-item kernel ``name`` over ``n_work`` items with the block size the device and the kernel's register use call for."""
    kernel = _kernel(cp, name)
    threads = block_size_for(kernel, work_items=n_work)
    kernel(((n_work + threads - 1) // threads,), (threads,), args)


class DeviceWinners:
    """The packed plan codes resident on the device; :meth:`picks` returns the winners of a block of permutations."""

    def __init__(self, cp: Any, xb: np.ndarray, kx: np.ndarray, nv: np.ndarray, val_idx: np.ndarray, y_raw: np.ndarray, px: np.ndarray) -> None:
        self.cp = cp
        self.F, self.P, self.M, self.NV = (int(v) for v in xb.shape)
        self.C = self.F * self.P * self.M
        self.xb = cp.asarray(np.ascontiguousarray(xb, dtype=np.int8))
        self.kx_h = np.ascontiguousarray(kx, dtype=np.int64)
        self.kx = cp.asarray(self.kx_h)
        self.nv = cp.asarray(np.ascontiguousarray(nv, dtype=np.int64))
        self.val_idx = cp.asarray(np.ascontiguousarray(val_idx, dtype=np.int64))
        self.valid = cp.asarray(np.arange(self.NV)[None, :] < np.asarray(nv)[:, None])  # (F, NV)
        self.y_raw = cp.asarray(np.ascontiguousarray(y_raw, dtype=np.int32))
        self.kxmax = max(1, int(self.kx_h.max()))
        self.px_h = px
        self.masks: Any = None  # the per-bin row bitmasks, built on the first binary-target block

    def _build_masks(self) -> None:
        """Per-bin row bitmasks of every cell, the cells' bin offsets and their fixed bin counts (once)."""
        cp = self.cp
        kxc = self.kx_h.ravel()
        off = np.zeros(kxc.size, dtype=np.int64)
        off[1:] = np.cumsum(kxc)[:-1]
        self.NW = (self.NV + 31) // 32
        self.off = cp.asarray(off)
        self.masks = cp.zeros((int(kxc.sum()) * self.NW,), dtype=cp.uint32)
        total = self.C * self.NV
        args = (self.xb, self.off, self.kx, self.nv, np.int32(self.C), np.int32(self.P * self.M), np.int32(self.NV), np.int32(self.NW), self.masks)
        _launch_flat(cp, "oj_build_masks", total, args)
        self.px = cp.asarray(np.ascontiguousarray(self.px_h.reshape(self.C, -1)[:, : self.kxmax], dtype=np.int32))

    def _cell_mi_bitmask(self, vy: Any, ky: Any, b_n: int) -> Any:
        """``(B, F, P, M)`` cell MI of a binary-target block through the bitmask popcounts."""
        cp = self.cp
        if self.masks is None:
            self._build_masks()
        n_y = b_n * self.F * self.NW
        ybits = cp.empty((n_y,), dtype=cp.uint32)
        _launch_flat(cp, "oj_build_ybits", n_y, (vy, self.nv, np.int32(b_n), np.int32(self.F), np.int32(self.NV), np.int32(self.NW), ybits))
        c1 = cp.zeros((b_n * self.C * self.kxmax,), dtype=cp.int32)
        pm = np.int32(self.P * self.M)
        args = (self.masks, ybits, self.off, self.kx, np.int32(b_n), np.int32(self.C), np.int32(self.F), pm, np.int32(self.NW), np.int32(self.kxmax), c1)
        _kernel(cp, "oj_count_ones")((self.C, (b_n + 7) // 8), (128,), args)
        out = cp.empty((b_n, self.F, self.P, self.M), dtype=cp.float64)  # Intentional float64: MI statistic
        n_o = b_n * self.C
        args2 = (c1, self.px, self.kx, self.nv, ky, np.int32(b_n), np.int32(self.C), np.int32(self.F), pm, np.int32(self.kxmax), out)
        _kernel(cp, "oj_mi_from_ones")(((n_o + 127) // 128,), (128,), args2)
        return out

    def _cell_mi_scatter(self, vy: Any, ky: Any, b_n: int, kymax: int) -> Any:
        """``(B, F, P, M)`` cell MI through shared-memory histograms (any target)."""
        cp = self.cp
        out = cp.empty((b_n, self.F, self.P, self.M), dtype=cp.float64)  # Intentional float64: MI statistic
        # per-thread counters: bench-attempt-rejected (2026-11-03) on a GTX 1050 Ti, 0.92 s vs 0.41 s per 64 permutations for the atomics (32 KB of
        # shared memory per block leaves one block per SM); MLFRAME_HONEST_OJ_GPU_PRIVATE=1 turns it on
        private = self.kxmax * kymax <= 64 and self.NV <= 256 * 65535 and os.environ.get("MLFRAME_HONEST_OJ_GPU_PRIVATE", "0").strip() == "1"
        args = (self.xb, vy, self.kx, self.nv, ky, np.int32(self.F), np.int32(self.P * self.M), np.int32(self.NV), np.int32(self.kxmax), np.int32(kymax), out)
        smem = (2 * 64 * 256) if private else 4 * 8 * self.kxmax * kymax
        _kernel(cp, "oj_cell_mi_private" if private else "oj_cell_mi")((self.C, b_n), (256,), args, shared_mem=smem)
        return out

    def picks(self, perms: np.ndarray) -> Optional[np.ndarray]:
        """``(B, P)`` winner index per permutation and column (``-1`` where no candidate was scored), or ``None`` for targets too wide for the kernels."""
        cp = self.cp
        b_n = int(perms.shape[0])
        pg = cp.asarray(perms.astype(np.int32, copy=False))
        vy = cp.ascontiguousarray(cp.where(self.valid[None], self.y_raw[pg[:, self.val_idx]], 0).astype(cp.int32))  # (B, F, NV)
        ky = cp.ascontiguousarray((cp.where(self.valid[None], vy, -1).max(axis=2) + 1).astype(cp.int32))  # (B, F)
        kymax = int(ky.max())
        if kymax > _MAX_BINS or self.kxmax > _MAX_BINS:
            return None
        if kymax <= 2 and self.kxmax <= 32 and os.environ.get("MLFRAME_HONEST_OJ_GPU_BITMASK", "1").strip() != "0":
            out = self._cell_mi_bitmask(vy, ky, b_n)
        else:
            out = self._cell_mi_scatter(vy, ky, b_n, kymax)
        # mean over the scored folds, summed in fold order as the CPU does, then the first strict maximum over the candidates
        total = cp.zeros((b_n, self.P, self.M), dtype=cp.float64)  # Intentional float64: MI statistic
        cells = cp.zeros((self.P, self.M), dtype=cp.float64)  # Intentional float64: fold count divisor
        for f in range(self.F):
            ok = cp.asarray(self.kx_h[f] > 0)
            total = total + cp.where(ok[None], out[:, f], 0.0)
            cells = cells + ok
        mean = cp.where(cells[None] > 0, total / cp.maximum(cells[None], 1.0), -cp.inf)
        pick = cp.argmax(mean, axis=2)
        pick = cp.where(cp.isneginf(mean.max(axis=2)), -1, pick)
        return np.asarray(cp.asnumpy(pick), dtype=np.int64)


def _synthetic_winner_inputs(dims: dict, seed: int = 0) -> tuple:
    """Packed-plan inputs of the winner pick for the sweep: 3 folds over ``n_rows`` rows, ``n_cols`` cells as (column, 4 candidates), binary target,
    one 64-permutation block."""
    rng = np.random.default_rng(seed)
    n, cells = int(dims["n_rows"]), int(dims["n_cols"])
    n_f, n_m = 3, 4
    p = max(1, cells // (n_f * n_m))
    y = rng.integers(0, 2, size=n).astype(np.int64)
    fold = rng.permutation(n) % n_f
    nv = np.array([(fold == f).sum() for f in range(n_f)], dtype=np.int64)
    val_idx = np.zeros((n_f, int(nv.max())), dtype=np.int64)
    for f in range(n_f):
        val_idx[f, : nv[f]] = np.flatnonzero(fold == f)
    kx = np.tile(np.array([4, 8, 16, 32], dtype=np.int64), (n_f, p, 1))
    xb = (rng.random((n_f, p, n_m, int(nv.max()))) * kx[..., None]).astype(np.int8)
    perms = np.stack([rng.permutation(n) for _ in range(64)]).astype(np.int64)
    return xb, kx, nv, val_idx, y, perms


def _sweep_winners() -> list:
    """Host-vs-device sweep of one winner-pick block for the ``honest_oj_winners`` kernel-tuning entry."""
    from ._honest_oj_fused import _fold_targets, _px_table, _winners_minority
    from ._scorer_device_dispatch import _FORCED, sweep_scorer

    @njit(cache=True, parallel=True, nogil=True)
    def _host_picks(perms, y, nv, val_idx, xb, kx, px, out):
        """CPU winner pick of every permutation of the block."""
        for b in prange(perms.shape[0]):
            vy, ky = _fold_targets(perms[b], y, nv, val_idx)
            out[b] = _winners_minority(xb, kx, nv, vy, ky, px)

    def run(inp: Any) -> np.ndarray:
        """The winner pick of the block with the forced backend."""
        xb, kx, nv, val_idx, y, perms = inp
        px = _px_table(xb, kx, nv)
        if _FORCED.get() == "device":
            import cupy as cp

            picks = DeviceWinners(cp, xb, kx, nv, val_idx, y, px).picks(perms)
            if picks is not None:
                return picks
        out = np.empty((perms.shape[0], kx.shape[1]), dtype=np.int64)
        _host_picks(perms, y, nv, val_idx, xb, kx, px, out)
        return out

    return sweep_scorer(run, _synthetic_winner_inputs)


def device_winners(xb: np.ndarray, kx: np.ndarray, nv: np.ndarray, val_idx: np.ndarray, y_raw: np.ndarray, px: np.ndarray, n_rows: int) -> Optional[DeviceWinners]:
    """A :class:`DeviceWinners` when the device path applies (see the module docstring), else ``None``."""
    if os.environ.get("MLFRAME_HONEST_OJ_GPU", "1").strip() == "0":
        return None
    try:
        from ._gpu_policy import cuda_available_for_run

        if not cuda_available_for_run():
            return None
        from ._scorer_device_dispatch import scorer_device_choice

        if not scorer_device_choice("honest_oj_winners", int(n_rows), int(np.prod(kx.shape)), sweep=_sweep_winners):
            return None
        import cupy as cp

        # codes, fold rows, target, and the bitmasks (one bit per row and bin)
        need = int(xb.size) + 8 * int(val_idx.size) + 4 * int(n_rows) + int(np.asarray(kx).sum()) * ((int(xb.shape[-1]) + 31) // 32) * 4
        free, _total = cp.cuda.runtime.memGetInfo()
        if need > 0.5 * free:
            logger.debug("honest optimal_joint winners stay on the CPU: %d bytes needed, %d free", need, free)
            return None
        return DeviceWinners(cp, xb, kx, nv, val_idx, y_raw, px)
    except Exception as exc:
        log_throttle(logger, "_honest_oj_fused_gpu.except_421", logging.WARNING, "honest optimal_joint device winners unavailable (%s: %s); CPU", type(exc).__name__, exc)
        return None


class DevicePool:
    """The pool codes at the evaluation rows as per-bin bitmasks on the device; :meth:`maxima` is ``_pool_max`` of a block for a binary target."""

    def __init__(self, cp: Any, pool_dep: np.ndarray, pool_rc: np.ndarray, pool_nb: np.ndarray, pool_kxe: np.ndarray, yc: np.ndarray, rows: np.ndarray) -> None:
        self.cp = cp
        n_pool, n_m, r = (int(v) for v in pool_rc.shape)
        self.T, self.M, self.R = n_pool, n_m, r
        self.C = n_pool * n_m
        kx = np.ascontiguousarray(pool_nb, dtype=np.int64).copy()
        kx[pool_dep < 0, 1:] = 0  # a fixed column has candidate 0 only
        self.kx_h = kx.ravel()
        self.kxmax = max(1, int(self.kx_h.max()))
        self.NW = (r + 31) // 32
        off = np.zeros(self.C, dtype=np.int64)
        off[1:] = np.cumsum(self.kx_h)[:-1]
        self.off, self.kx = cp.asarray(off), cp.asarray(self.kx_h)
        xb = cp.asarray(np.ascontiguousarray(pool_rc.reshape(self.C, r), dtype=np.int8))
        self.nv = cp.asarray(np.array([r], dtype=np.int64))
        self.masks = cp.zeros((int(self.kx_h.sum()) * self.NW,), dtype=cp.uint32)
        args = (xb, self.off, self.kx, self.nv, np.int32(self.C), np.int32(self.C), np.int32(r), np.int32(self.NW), self.masks)
        _kernel(cp, "oj_build_masks")(((self.C * r + 255) // 256,), (256,), args)
        px = np.zeros((self.C, self.kxmax), dtype=np.int32)
        flat = pool_rc.reshape(self.C, r)
        for c in range(self.C):
            if self.kx_h[c] > 0:
                px[c] = np.bincount(flat[c], minlength=self.kxmax)[: self.kxmax]
        self.px = cp.asarray(px)
        self.dep = pool_dep
        self.kxe = cp.asarray(np.ascontiguousarray(pool_kxe, dtype=np.float64))  # Intentional float64: Miller-Madow support term
        self.yc = cp.asarray(np.ascontiguousarray(yc, dtype=np.int32))
        self.rows = cp.asarray(np.ascontiguousarray(rows, dtype=np.int64))

    def maxima(self, perms: np.ndarray, picks: np.ndarray, n_eff: float) -> np.ndarray:
        """``_pool_max`` of every permutation of the block (``-inf`` where a re-binned column has no winner)."""
        cp = self.cp
        b_n = int(perms.shape[0])
        pg = cp.asarray(perms.astype(np.int32, copy=False))
        vy = cp.ascontiguousarray(self.yc[pg[:, self.rows]])  # (B, R)
        ones = vy.sum(axis=1).astype(cp.int32)
        n_y = b_n * self.NW
        ybits = cp.empty((n_y,), dtype=cp.uint32)
        _launch_flat(cp, "oj_build_ybits", n_y, (vy, self.nv, np.int32(b_n), np.int32(1), np.int32(self.R), np.int32(self.NW), ybits))
        c1 = cp.zeros((b_n * self.C * self.kxmax,), dtype=cp.int32)
        args = (self.masks, ybits, self.off, self.kx, np.int32(b_n), np.int32(self.C), np.int32(1), np.int32(self.C), np.int32(self.NW), np.int32(self.kxmax), c1)
        _kernel(cp, "oj_count_ones")((self.C, (b_n + 7) // 8), (128,), args)
        mi = cp.empty((b_n, self.T, self.M), dtype=cp.float64)  # Intentional float64: MI statistic
        n_o = b_n * self.C
        _launch_flat(cp, "oj_pool_mi", n_o, (c1, self.px, self.kx, ones, np.int32(b_n), np.int32(self.C), np.int32(self.kxmax), np.int64(self.R), mi))
        # the candidate of every pool column under every permutation: the column's winner, 0 for a fixed column
        dep = self.dep
        m_idx = np.zeros((b_n, self.T), dtype=np.int64)
        has = dep >= 0
        m_idx[:, has] = picks[:, dep[has]]
        bad = (m_idx < 0).any(axis=1)
        m_g = cp.asarray(np.maximum(m_idx, 0))
        sel = cp.take_along_axis(mi, m_g[:, :, None], axis=2)[:, :, 0]
        kxe = self.kxe[cp.arange(self.T)[None, :], m_g]
        ky_eff = ((ones > 0).astype(cp.float64) + (ones < self.R).astype(cp.float64))[:, None]  # Intentional float64: the class counts are exact integers in double, as in the CPU kernel
        corr = sel - (kxe - 1.0) * (ky_eff - 1.0) / (2.0 * float(n_eff))
        res = cp.maximum(corr.max(axis=1), 0.0)
        out = np.asarray(cp.asnumpy(res), dtype=np.float64)  # Intentional float64: statistic vector
        out[bad] = -np.inf
        self.last_mi = np.asarray(cp.asnumpy(sel), dtype=np.float64)  # Intentional float64: (B, T) plain MI of the block, kept for the candidates' null
        return out


def device_pool(dev: Optional[DeviceWinners], pool_dep: np.ndarray, pool_rc: np.ndarray, pool_nb: np.ndarray, pool_kxe: np.ndarray, yc: np.ndarray,
                rows: np.ndarray) -> Optional[DevicePool]:
    """A :class:`DevicePool` when the winners run on the device, the target codes are binary and the pool codes fit the bitmask kernels, else ``None``."""
    if dev is None or os.environ.get("MLFRAME_HONEST_OJ_GPU_POOL", "1").strip() == "0":
        return None
    try:
        y_at = np.asarray(yc)[np.asarray(rows)]
        if y_at.size == 0 or int(y_at.min()) < 0 or int(y_at.max()) > 1 or int(np.max(pool_nb)) > 127:
            return None
        cp = dev.cp
        need = int(pool_rc.size) + int(np.sum(pool_nb)) * ((int(pool_rc.shape[-1]) + 31) // 32) * 4
        free, _total = cp.cuda.runtime.memGetInfo()
        if need > 0.5 * free:
            return None
        return DevicePool(cp, pool_dep, pool_rc, pool_nb, pool_kxe, yc, rows)
    except Exception as exc:
        log_throttle(logger, "_honest_oj_fused_gpu.except_507", logging.WARNING, "honest optimal_joint device pool unavailable (%s: %s); CPU", type(exc).__name__, exc)
        return None
