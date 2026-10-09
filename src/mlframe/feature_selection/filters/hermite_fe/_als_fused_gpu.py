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

import threading
from typing import Any, Optional

from mlframe.feature_selection.filters._safe_scale import _REL_TOL

_MODULES: dict = {}
_LOCK = threading.Lock()
_STAT_BLOCKS = 64  # fixed grid of the statistics kernel, so a consumer block reduces only this many partials

_SRC = r"""
extern "C" {
__device__ __forceinline__ double block_sum(double v, double* sh) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    double r = 0.0;
    if (threadIdx.x == 0) { for (int w = 0; w < 8; ++w) r += sh[w]; }
    return r;
}
__device__ __forceinline__ double block_max(double v, double* sh) {
    for (int o = 16; o > 0; o >>= 1) v = fmax(v, __shfl_down_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    double r = 0.0;
    if (threadIdx.x == 0) { for (int w = 0; w < 8; ++w) r = fmax(r, sh[w]); }
    return r;
}
__global__ void matvec_stats(const double* B, const double* c, double* f, double* stats, const long long n) {
    double cc[D];
    #pragma unroll
    for (int k = 0; k < D; ++k) cc[k] = c[k];
    double f0 = 0.0;
    #pragma unroll
    for (int k = 0; k < D; ++k) f0 += B[k] * cc[k];
    double s1 = 0.0, s2 = 0.0, mx = 0.0;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const double* r = B + i * D;
        double s = 0.0;
        #pragma unroll
        for (int k = 0; k < D; ++k) s += r[k] * cc[k];
        f[i] = s;
        const double d = s - f0;
        s1 += d; s2 += d * d; mx = fmax(mx, fabs(s));
    }
    __shared__ double sh[8];
    double r1 = block_sum(s1, sh); __syncthreads();
    double r2 = block_sum(s2, sh); __syncthreads();
    double r3 = block_max(mx, sh);
    if (threadIdx.x == 0) {
        stats[blockIdx.x * 4 + 0] = r1; stats[blockIdx.x * 4 + 1] = r2; stats[blockIdx.x * 4 + 2] = r3; stats[blockIdx.x * 4 + 3] = f0;
    }
}
__global__ void gram_scaled(const double* B, const double* w, const double* y, const double* stats, double* part, const long long n, const int nstat, const int use_w, const double rel_tol) {
    __shared__ double scale_sh;
    if (threadIdx.x == 0) {
        double s1 = 0.0, s2 = 0.0, mx = 0.0;
        for (int b = 0; b < nstat; ++b) { s1 += stats[b * 4]; s2 += stats[b * 4 + 1]; mx = fmax(mx, stats[b * 4 + 2]); }
        const double mean = s1 / (double)n;
        const double var = fmax(s2 / (double)n - mean * mean, 0.0);
        const double sd = sqrt(var);
        scale_sh = (sd > rel_tol * mx) ? sd : 1.0;
    }
    __syncthreads();
    const double scale = scale_sh;
    double acc[NACC];
    #pragma unroll
    for (int t = 0; t < NACC; ++t) acc[t] = 0.0;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const double* r = B + i * D;
        const double wi = use_w ? w[i] / scale : 1.0;
        double a[D];
        #pragma unroll
        for (int k = 0; k < D; ++k) a[k] = r[k] * wi;
        const double yi = y[i];
        int t = 0;
        #pragma unroll
        for (int j = 0; j < D; ++j) {
            #pragma unroll
            for (int k = j; k < D; ++k) { acc[t] += a[j] * a[k]; ++t; }
            acc[t] += a[j] * yi; ++t;
        }
    }
    __shared__ double warp_sums[8][NACC];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    #pragma unroll
    for (int t = 0; t < NACC; ++t) {
        double v = acc[t];
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) warp_sums[warp][t] = v;
    }
    __syncthreads();
    if (threadIdx.x < NACC) {
        double v = 0.0;
        for (int wv = 0; wv < 8; ++wv) v += warp_sums[wv][threadIdx.x];
        part[(long long)blockIdx.x * NACC + threadIdx.x] = v;
    }
}
__global__ void solve_small(const double* part, const int nblocks, double* coef) {
    __shared__ double tot[NACC];
    if (threadIdx.x < NACC) {
        // Neumaier-compensated sum: thousands of partials added serially would otherwise cost ~nblocks*eps, which a badly conditioned Gram matrix amplifies.
        double s = 0.0, comp = 0.0;
        for (int b = 0; b < nblocks; ++b) {
            const double x = part[(long long)b * NACC + threadIdx.x];
            const double t = s + x;
            if (fabs(s) >= fabs(x)) comp += (s - t) + x; else comp += (x - t) + s;
            s = t;
        }
        tot[threadIdx.x] = s + comp;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    double M[D][D + 1];
    int t = 0;
    for (int j = 0; j < D; ++j) {
        for (int k = j; k < D; ++k) { M[j][k] = tot[t]; M[k][j] = tot[t]; ++t; }
        M[j][D] = tot[t]; ++t;
    }
    bool bad = false;
    for (int col = 0; col < D && !bad; ++col) {
        int piv = col;
        double best = fabs(M[col][col]);
        for (int r = col + 1; r < D; ++r) { if (fabs(M[r][col]) > best) { best = fabs(M[r][col]); piv = r; } }
        if (!(best > 0.0) || !isfinite(best)) { bad = true; break; }
        if (piv != col) { for (int k = 0; k <= D; ++k) { double tmp = M[col][k]; M[col][k] = M[piv][k]; M[piv][k] = tmp; } }
        for (int r = col + 1; r < D; ++r) {
            const double fct = M[r][col] / M[col][col];
            for (int k = col; k <= D; ++k) M[r][k] -= fct * M[col][k];
        }
    }
    if (bad) { for (int k = 0; k < D; ++k) coef[k] = nan(""); return; }
    for (int r = D - 1; r >= 0; --r) {
        double s = M[r][D];
        for (int k = r + 1; k < D; ++k) s -= M[r][k] * coef[k];
        coef[r] = s / M[r][r];
    }
}
}
"""


def _module(cp, d: int):
    """Kernel module specialised (unrolled) for design width ``d``, built once per width."""
    with _LOCK:
        mod = _MODULES.get(d)
        if mod is None:
            nacc = d * (d + 1) // 2 + d
            mod = cp.RawModule(code=_SRC, options=("-std=c++14", f"-DD={d}", f"-DNACC={nacc}"))
            _MODULES[d] = mod
        return mod


def als_sweep_fused(cp: Any, Ba: Any, Bb: Any, yc: Any, iters: int) -> Optional[tuple]:
    """Both coefficient vectors of the alternating sweep as host arrays, or ``None`` when the fused path cannot give a finite answer (the caller runs the original path).

    ``Ba``/``Bb`` are C-contiguous float64 ``(n, d)`` designs with the same ``d <= 12``, ``yc`` the centred target."""
    import numpy as np

    n, d = Ba.shape
    if Bb.shape != Ba.shape or d > 12 or n < 2:
        return None
    ba = cp.ascontiguousarray(Ba, dtype=cp.float64)
    bb = cp.ascontiguousarray(Bb, dtype=cp.float64)
    y = cp.ascontiguousarray(yc, dtype=cp.float64)
    mod = _module(cp, d)
    matvec_stats = mod.get_function("matvec_stats")
    gram_scaled = mod.get_function("gram_scaled")
    solve_small = mod.get_function("solve_small")
    nacc = d * (d + 1) // 2 + d
    gram_blocks = int(min(4096, max(1, (n + 255) // 256)))
    part = cp.empty((gram_blocks, nacc), dtype=cp.float64)
    coef = cp.empty(2 * d, dtype=cp.float64)  # [ca | cb]
    ca, cb = coef[:d], coef[d:]
    f = cp.empty(n, dtype=cp.float64)
    g = cp.empty(n, dtype=cp.float64)
    stats_f = cp.empty((_STAT_BLOCKS, 4), dtype=cp.float64)
    stats_g = cp.empty((_STAT_BLOCKS, 4), dtype=cp.float64)
    nn = cp.int64(n)
    rel = cp.float64(_REL_TOL)

    def gram(B, w, stats):
        """Launch the Gram kernel (``w=None`` for the unweighted first solve) and the small solve that follows it."""
        gram_scaled((gram_blocks,), (256,), (B, w if w is not None else y, y, stats, part, nn, cp.int32(_STAT_BLOCKS), cp.int32(0 if w is None else 1), rel))

    def solve(out):
        """Reduce the Gram partials and solve the normal equations into ``out``."""
        solve_small((1,), (256,), (part, cp.int32(gram_blocks), out))

    gram(bb, None, stats_g)
    solve(cb)
    matvec_stats((_STAT_BLOCKS,), (256,), (bb, cb, g, stats_g, nn))
    for _ in range(max(1, int(iters))):
        gram(ba, g, stats_g)
        solve(ca)
        matvec_stats((_STAT_BLOCKS,), (256,), (ba, ca, f, stats_f, nn))
        gram(bb, f, stats_f)
        solve(cb)
        matvec_stats((_STAT_BLOCKS,), (256,), (bb, cb, g, stats_g, nn))
    host = cp.asnumpy(coef)
    if not np.all(np.isfinite(host)):
        return None
    return np.ascontiguousarray(host[:d], dtype=np.float64), np.ascontiguousarray(host[d:], dtype=np.float64)
