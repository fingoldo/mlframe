"""The rank-1 ALS alternating sweep in about twenty kernel launches instead of three hundred.

The resident sweep (:func:`_hermite_prewarp_gpu_resident._als_sweep_gpu`) is launch-bound: per half-iteration it runs a matvec, ``std`` and ``abs().max()`` reductions, a guarded
divide, a Gram kernel, a partial-sum reduction, two gathers and a ``cupy.linalg.solve`` (a handful of cusolver kernels) on a matrix no bigger than 9x9 - about 330 launches per
call for ~3 ms of GPU work. Here one half-iteration is three launches:

* ``matvec_stats``: ``f = B c`` and per-block ``(sum, sum of squares, max|f|)`` of ``f`` (shifted by ``f[0]`` so the one-pass variance keeps its precision);
* ``gram_scaled``: ``A'A`` and ``A'y`` for ``A = B_other * (f / scale)`` with ``scale`` derived in the kernel prologue from those block statistics exactly as
  ``f / guarded_scale(std(f), max|f|)`` does, so the normalised weight is never written out;
* ``solve_small``: one block sums the Gram partials in fixed order and solves the ``d x d`` normal equations by Gaussian elimination with partial pivoting.

A singular normal matrix yields NaN coefficients; the caller then runs the original path (which falls back to ``lstsq``), so the result contract is unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from mlframe.feature_selection.filters._kernel_types import render
from mlframe.feature_selection.filters._raw_module_cache import RawModuleCache
from mlframe.feature_selection.filters._safe_scale import _REL_TOL

_STAT_BLOCKS = 64  # fixed grid of the statistics kernel, so a consumer block reduces only this many partials

_SRC = r"""
extern "C" {
__device__ __forceinline__ {ACC} block_sum({ACC} v, {ACC}* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    {ACC} r = 0.0;
    if (threadIdx.x == 0) { for (int w = 0; w < 8; ++w) r += sh[w]; }
    return r;
}
__device__ __forceinline__ {ACC} block_max({ACC} v, {ACC}* sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    {ACC} r = 0.0;
    if (threadIdx.x == 0) { for (int w = 0; w < 8; ++w) r = fmax(r, sh[w]); }
    return r;
}
__global__ void matvec_stats(const {REAL}* B, const {ACC}* c, {REAL}* f, {ACC}* stats, const long long n) {
    {ACC} cc[D];
    #pragma unroll
    for (int k = 0; k < D; ++k) cc[k] = c[k];
    {ACC} f0 = 0.0;
    #pragma unroll
    for (int k = 0; k < D; ++k) f0 += B[k] * cc[k];
    {ACC} s1 = 0.0, s2 = 0.0, mx = 0.0;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const {REAL}* r = B + i * D;
        {ACC} s = 0.0;
        #pragma unroll
        for (int k = 0; k < D; ++k) s += r[k] * cc[k];
        f[i] = s;
        const {ACC} d = s - f0;
        s1 += d; s2 += d * d; mx = fmax(mx, fabs(s));
    }
    __shared__ {ACC} sh[8];
    {ACC} r1 = block_sum(s1, sh); __syncthreads();
    {ACC} r2 = block_sum(s2, sh); __syncthreads();
    {ACC} r3 = block_max(mx, sh);
    if (threadIdx.x == 0) {
        stats[blockIdx.x * 4 + 0] = r1; stats[blockIdx.x * 4 + 1] = r2; stats[blockIdx.x * 4 + 2] = r3; stats[blockIdx.x * 4 + 3] = f0;
    }
}
__global__ void gram_scaled(const {REAL}* B, const {REAL}* w, const {REAL}* y, const {ACC}* stats, {ACC}* part, const long long n, const int nstat, const int use_w, const {ACC} rel_tol) {
    __shared__ {ACC} scale_sh;
    if (threadIdx.x == 0) {
        {ACC} s1 = 0.0, s2 = 0.0, mx = 0.0;
        for (int b = 0; b < nstat; ++b) { s1 += stats[b * 4]; s2 += stats[b * 4 + 1]; mx = fmax(mx, stats[b * 4 + 2]); }
        const {ACC} mean = s1 / ({ACC})n;
        const {ACC} var = fmax(s2 / ({ACC})n - mean * mean, 0.0);
        const {ACC} sd = sqrt(var);
        scale_sh = (sd > rel_tol * mx) ? sd : 1.0;
    }
    __syncthreads();
    const {ACC} scale = scale_sh;
    {ACC} acc[NACC];
    #pragma unroll
    for (int t = 0; t < NACC; ++t) acc[t] = 0.0;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const {REAL}* r = B + i * D;
        const {ACC} wi = use_w ? w[i] / scale : 1.0;
        {ACC} a[D];
        #pragma unroll
        for (int k = 0; k < D; ++k) a[k] = r[k] * wi;
        const {ACC} yi = y[i];
        int t = 0;
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            #pragma unroll
            for (int k = j; k < D; ++k) { acc[t] += a[j] * a[k]; ++t; }
            acc[t] += a[j] * yi; ++t;
        }
    }
    __shared__ {ACC} warp_sums[8][NACC];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int t = 0; t < NACC; ++t) {
        {ACC} v = acc[t];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) warp_sums[warp][t] = v;
    }
    __syncthreads();
    if (threadIdx.x < NACC) {
        {ACC} v = 0.0;
        for (int wv = 0; wv < 8; ++wv) v += warp_sums[wv][threadIdx.x];
        part[(long long)blockIdx.x * NACC + threadIdx.x] = v;
    }
}
__global__ void solve_small(const {ACC}* part, const int nblocks, {ACC}* coef) {
    __shared__ {ACC} tot[NACC];
    if (threadIdx.x < NACC) {
        // Neumaier-compensated sum: thousands of partials added serially would otherwise cost ~nblocks*eps, which a badly conditioned Gram matrix amplifies.
        {ACC} s = 0.0, comp = 0.0;
        for (int b = 0; b < nblocks; ++b) {
            const {ACC} x = part[(long long)b * NACC + threadIdx.x];
            const {ACC} t = s + x;
            if (fabs(s) >= fabs(x)) comp += (s - t) + x; else comp += (x - t) + s;
            s = t;
        }
        tot[threadIdx.x] = s + comp;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    {ACC} M[D][D + 1];
    int t = 0;
    for (int j = 0; j < D; ++j) {
        for (int k = j; k < D; ++k) { M[j][k] = tot[t]; M[k][j] = tot[t]; ++t; }
        M[j][D] = tot[t]; ++t;
    }
    bool bad = false;
    for (int col = 0; col < D && !bad; ++col) {
        int piv = col;
        {ACC} best = fabs(M[col][col]);
        for (int r = col + 1; r < D; ++r) { if (fabs(M[r][col]) > best) { best = fabs(M[r][col]); piv = r; } }
        if (!(best > 0.0) || !isfinite(best)) { bad = true; break; }
        if (piv != col) { for (int k = 0; k <= D; ++k) { {ACC} tmp = M[col][k]; M[col][k] = M[piv][k]; M[piv][k] = tmp; } }
        for (int r = col + 1; r < D; ++r) {
            const {ACC} fct = M[r][col] / M[col][col];
            for (int k = col; k <= D; ++k) M[r][k] -= fct * M[col][k];
        }
    }
    if (bad) { for (int k = 0; k < D; ++k) coef[k] = nan(""); return; }
    for (int r = D - 1; r >= 0; --r) {
        {ACC} s = M[r][D];
        for (int k = r + 1; k < D; ++k) s -= M[r][k] * coef[k];
        coef[r] = s / M[r][r];
    }
}
}
"""


_MODULES_BY_REAL = {np.dtype(t): RawModuleCache(render(_SRC, real=t)) for t in (np.float32, np.float64)}  # Intentional float64: one module per design element type


def als_sweep_fused(cp: Any, Ba: Any, Bb: Any, yc: Any, iters: int) -> Optional[tuple]:
    """Both coefficient vectors of the alternating sweep as host arrays, or ``None`` when the fused path cannot give a finite answer (the caller runs the original path).

    ``Ba``/``Bb`` are C-contiguous ``(n, d)`` float32 or double designs (read in their own type) with the same ``d <= 12``, ``yc`` the centred target."""
    n, d = Ba.shape
    if Bb.shape != Ba.shape or d > 12 or n < 2:
        return None
    real = np.dtype(np.float32) if Ba.dtype == cp.float32 else np.dtype(np.float64)  # Intentional float64: the designs are read in their own type; the sums, the Gram and the coefficients stay float64
    ba = cp.ascontiguousarray(Ba, dtype=real)
    bb = cp.ascontiguousarray(Bb, dtype=real)
    y = cp.ascontiguousarray(yc, dtype=real)
    mod = _MODULES_BY_REAL[real].get(cp, d)
    matvec_stats = mod.get_function("matvec_stats")
    gram_scaled = mod.get_function("gram_scaled")
    solve_small = mod.get_function("solve_small")
    nacc = d * (d + 1) // 2 + d
    gram_blocks = int(min(4096, max(1, (n + 255) // 256)))
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    part = cp.empty((gram_blocks, nacc), dtype=cp.float64)
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    coef = cp.empty(2 * d, dtype=cp.float64)  # [ca | cb]
    ca, cb = coef[:d], coef[d:]
    f = cp.empty(n, dtype=real)
    g = cp.empty(n, dtype=real)
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    stats_f = cp.empty((_STAT_BLOCKS, 4), dtype=cp.float64)
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    stats_g = cp.empty((_STAT_BLOCKS, 4), dtype=cp.float64)
    nn = cp.int64(n)
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    rel = cp.float64(_REL_TOL)

    def gram(B, w, stats):
        """Launch the Gram kernel (``w=None`` for the unweighted first solve) and the small solve that follows it."""
        gram_scaled((gram_blocks,), (256,), (B, w if w is not None else y, y, stats, part, nn, cp.int32(_STAT_BLOCKS), cp.int32(0 if w is None else 1), rel))  # launch-ok: the kernels keep one partial per warp in shared arrays of 8 rows, so the block is exactly 8 warps

    def solve(out):
        """Reduce the Gram partials and solve the normal equations into ``out``."""
        solve_small((1,), (256,), (part, cp.int32(gram_blocks), out))  # launch-ok: the kernels keep one partial per warp in shared arrays of 8 rows, so the block is exactly 8 warps

    gram(bb, None, stats_g)
    solve(cb)
    matvec_stats((_STAT_BLOCKS,), (256,), (bb, cb, g, stats_g, nn))  # launch-ok: the kernels keep one partial per warp in shared arrays of 8 rows, so the block is exactly 8 warps
    for _ in range(max(1, int(iters))):
        gram(ba, g, stats_g)
        solve(ca)
        matvec_stats((_STAT_BLOCKS,), (256,), (ba, ca, f, stats_f, nn))  # launch-ok: the kernels keep one partial per warp in shared arrays of 8 rows, so the block is exactly 8 warps
        gram(bb, f, stats_f)
        solve(cb)
        matvec_stats((_STAT_BLOCKS,), (256,), (bb, cb, g, stats_g, nn))  # launch-ok: the kernels keep one partial per warp in shared arrays of 8 rows, so the block is exactly 8 warps
    host = cp.asnumpy(coef)
    if not np.all(np.isfinite(host)):
        return None
    # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
    return np.ascontiguousarray(host[:d], dtype=np.float64), np.ascontiguousarray(host[d:], dtype=np.float64)
