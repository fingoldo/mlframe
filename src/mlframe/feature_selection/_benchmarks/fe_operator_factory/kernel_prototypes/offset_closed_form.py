"""Closed-form shift estimators, O(n), no sort of candidates.
 (ii) interaction_ols_shift : y_rank ~ [1,u,v,uv] one-pass 4x4 normal equations -> t = b_v/b_uv (centered: minus mean(u))
 (i)  zero_cross_shift      : per u-quantile-bin slope of y on v is linear in bin-mean(u); its zero gives -t.
Deterministic: NBLK fixed row blocks (independent of thread count), Neumaier merge of block partials.
The CUDA twin of (ii) at the bottom (``_GRAM4_SRC``) was NEVER RUN ON A GPU; only ``emulate_gram4_cuda``, a pure-numpy emulation of its arithmetic, is verified (tests/feature_selection/fe/factory).
"""

import numpy as np
from numba import njit, prange

NBLK = 64  # fixed grid of the statistics kernel (same as hermite_fe/_als_fused_gpu._STAT_BLOCKS)
NACC = 14  # 10 upper-triangle Gram entries of [1,x,w,xw] + 4 right-hand sides
SING_REL_TOL = 1e-12
CUDA_THREADS = 256


@njit(cache=True, nogil=True)
def _neumaier_merge(part, out):
    """part (nblk, nacc) -> out (nacc,), fixed order, compensated."""
    nb, na = part.shape
    for j in range(na):
        s = 0.0
        comp = 0.0
        for b in range(nb):
            x = part[b, j]
            t = s + x
            if abs(s) >= abs(x):
                comp += (s - t) + x
            else:
                comp += (x - t) + s
            s = t
        out[j] = s + comp


@njit(cache=True, nogil=True)
def _solve4(tot, coef):
    """Gaussian elimination with partial pivoting on the 4x4 normal equations; NaN on singular (caller falls back)."""
    M = np.empty((4, 5))
    t = 0
    for j in range(4):
        for k in range(j, 4):
            M[j, k] = tot[t]
            M[k, j] = tot[t]
            t += 1
    for j in range(4):
        M[j, 4] = tot[10 + j]
    scale = 0.0
    for j in range(4):
        scale = max(scale, abs(M[j, j]))
    for col in range(4):
        piv = col
        best = abs(M[col, col])
        for r in range(col + 1, 4):
            if abs(M[r, col]) > best:
                best = abs(M[r, col])
                piv = r
        if not (best > SING_REL_TOL * scale) or not np.isfinite(best):
            for k in range(4):
                coef[k] = np.nan
            return
        if piv != col:
            for k in range(5):
                tmp = M[col, k]
                M[col, k] = M[piv, k]
                M[piv, k] = tmp
        for r in range(col + 1, 4):
            f = M[r, col] / M[col, col]
            for k in range(col, 5):
                M[r, k] -= f * M[col, k]
    for r in range(3, -1, -1):
        s = M[r, 4]
        for k in range(r + 1, 4):
            s -= M[r, k] * coef[k]
        coef[r] = s / M[r, r]


@njit(parallel=True, nogil=True, cache=True)
def interaction_ols_shift(U, V, yr):
    """U, V (K, n) float64; yr (n,) float64 (rank-transformed target in [0,1]).  Returns (K,) t estimates
    (NaN where singular / b_uv ~ 0).  Pass 0 centers (means), pass 1 accumulates NACC sums per row block."""
    K, n = U.shape
    t_out = np.empty(K)
    part = np.zeros((K, NBLK, NACC))
    mu = np.zeros((K, 2))
    my = 0.0
    for i in range(n):
        my += yr[i]
    my /= n
    for k in prange(K):
        su = 0.0
        sv = 0.0
        for i in range(n):
            su += U[k, i]
            sv += V[k, i]
        mu[k, 0] = su / n
        mu[k, 1] = sv / n
    blk = (n + NBLK - 1) // NBLK
    for kb in prange(K * NBLK):
        k = kb // NBLK
        b = kb % NBLK
        mu_u = mu[k, 0]
        mu_v = mu[k, 1]
        acc = np.zeros(NACC)
        for i in range(b * blk, min(n, (b + 1) * blk)):
            x = U[k, i] - mu_u
            w = V[k, i] - mu_v
            yy = yr[i] - my
            f3 = x * w
            acc[0] += 1.0
            acc[1] += x
            acc[2] += w
            acc[3] += f3
            acc[4] += x * x
            acc[5] += x * w
            acc[6] += x * f3
            acc[7] += w * w
            acc[8] += w * f3
            acc[9] += f3 * f3
            acc[10] += yy
            acc[11] += x * yy
            acc[12] += w * yy
            acc[13] += f3 * yy
        for j in range(NACC):
            part[k, b, j] = acc[j]
    for k in prange(K):
        tot = np.empty(NACC)
        coef = np.empty(4)
        _neumaier_merge(part[k], tot)
        _solve4(tot, coef)
        if np.isfinite(coef[3]) and abs(coef[3]) > 0.0:
            t_out[k] = coef[2] / coef[3] - mu[k, 0]
        else:
            t_out[k] = np.nan
    return t_out


@njit(cache=True, nogil=True)
def quantile_edges(x, nbins):
    """Interior edges (nbins-1), repo semantics (anchors + lerp)."""
    n = x.shape[0]
    nq = nbins + 1
    kths = np.empty(2 * (nbins - 1), dtype=np.int64)
    fr = np.empty(nbins - 1)
    los = np.empty(nbins - 1, dtype=np.int64)
    m = 0
    for k in range(1, nbins):
        pos = (k / (nq - 1)) * (n - 1)
        lo = int(np.floor(pos))
        hi = lo + 1 if lo < n - 1 else lo
        los[k - 1] = lo
        fr[k - 1] = pos - lo
        kths[m] = lo
        kths[m + 1] = hi
        m += 2
    part = np.partition(x, kths)
    e = np.empty(nbins - 1)
    for k in range(1, nbins):
        lo = los[k - 1]
        hi = lo + 1 if lo < n - 1 else lo
        e[k - 1] = part[lo] + (part[hi] - part[lo]) * fr[k - 1]
    return e


@njit(parallel=True, nogil=True, cache=True)
def zero_cross_shift(U, V, yr, nbins):
    """Per form: bin u by its own quantile edges, per bin slope s_j = cov(v,y)/var(v), WLS of s_j on bin-mean(u_j)
    (weights = bin counts): s = alpha + beta*ubar ; slope vanishes at ubar = -t  =>  t = alpha / beta.  Returns (K,)."""
    K, n = U.shape
    NS = 6
    t_out = np.empty(K)
    edges = np.empty((K, nbins - 1))
    for k in prange(K):
        edges[k] = quantile_edges(U[k], nbins)
    part = np.zeros((K, NBLK, nbins * NS))
    blk = (n + NBLK - 1) // NBLK
    for kb in prange(K * NBLK):
        k = kb // NBLK
        b = kb % NBLK
        for i in range(b * blk, min(n, (b + 1) * blk)):
            u = U[k, i]
            v = V[k, i]
            y = yr[i]
            lo = 0
            hi = nbins - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if u < edges[k, mid]:
                    hi = mid
                else:
                    lo = mid + 1
            o = lo * NS
            part[k, b, o] += 1.0
            part[k, b, o + 1] += u
            part[k, b, o + 2] += v
            part[k, b, o + 3] += y
            part[k, b, o + 4] += v * y
            part[k, b, o + 5] += v * v
    for k in prange(K):
        tot = np.empty(nbins * NS)
        _neumaier_merge(part[k], tot)
        sw = 0.0
        sx = 0.0
        sy = 0.0
        sxx = 0.0
        sxy = 0.0
        for j in range(nbins):
            o = j * NS
            cn = tot[o]
            if cn < 2.0:
                continue
            mv = tot[o + 2] / cn
            vv = tot[o + 5] / cn - mv * mv
            if not (vv > 0.0):
                continue
            s = (tot[o + 4] / cn - mv * tot[o + 3] / cn) / vv
            ub = tot[o + 1] / cn
            sw += cn
            sx += cn * ub
            sy += cn * s
            sxx += cn * ub * ub
            sxy += cn * ub * s
        den = sw * sxx - sx * sx
        if sw > 0.0 and abs(den) > 0.0:
            beta = (sw * sxy - sx * sy) / den
            alpha = (sy - beta * sx) / sw
            t_out[k] = alpha / beta if beta != 0.0 else np.nan
        else:
            t_out[k] = np.nan
    return t_out


# ------------------------------------------------------------------------------------------------ CUDA (not run here)
_GRAM4_SRC = r"""
#define NACC 14
extern "C" {
// grid = (NBLK, K); block = 256.  Thread (b,tid) walks i = b*256+tid, += NBLK*256 (fixed order, no atomics).
__global__ void gram4_partials(const double* U, const double* V, const double* yr, const double* mu /*K x 3: mu_u, mu_v, mu_y*/,
                               const long long n, double* part /*K x NBLK x NACC*/) {
    const int k = blockIdx.y;
    const double mu_u = mu[k * 3], mu_v = mu[k * 3 + 1], mu_y = mu[k * 3 + 2];
    double acc[NACC];
    #pragma unroll
    for (int t = 0; t < NACC; ++t) acc[t] = 0.0;
    const double* u = U + (long long)k * n;
    const double* v = V + (long long)k * n;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        const double f1 = u[i] - mu_u, f2 = v[i] - mu_v, f3 = f1 * f2, yy = yr[i] - mu_y;
        acc[0] += 1.0; acc[1] += f1; acc[2] += f2; acc[3] += f3;
        acc[4] += f1 * f1; acc[5] += f1 * f2; acc[6] += f1 * f3; acc[7] += f2 * f2; acc[8] += f2 * f3; acc[9] += f3 * f3;
        acc[10] += yy; acc[11] += f1 * yy; acc[12] += f2 * yy; acc[13] += f3 * yy;
    }
    __shared__ double warp_sums[8][NACC];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int t = 0; t < NACC; ++t) {
        double s = acc[t];
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        if (lane == 0) warp_sums[warp][t] = s;
    }
    __syncthreads();
    if (threadIdx.x < NACC) {
        double s = 0.0;
        for (int w = 0; w < 8; ++w) s += warp_sums[w][threadIdx.x];
        part[((long long)k * gridDim.x + blockIdx.x) * NACC + threadIdx.x] = s;
    }
}
// grid = K, block = 32: lanes 0..13 Neumaier-merge the NBLK partials, lane 0 solves the 4x4 (partial pivoting), writes t.
__global__ void solve4_shift(const double* part, const int nblk, const double* mu, const double sing_rel_tol, double* t_out) {
    __shared__ double tot[NACC];
    const int k = blockIdx.x;
    if (threadIdx.x < NACC) {
        double s = 0.0, comp = 0.0;
        for (int b = 0; b < nblk; ++b) {
            const double x = part[((long long)k * nblk + b) * NACC + threadIdx.x];
            const double t = s + x;
            if (fabs(s) >= fabs(x)) comp += (s - t) + x; else comp += (x - t) + s;
            s = t;
        }
        tot[threadIdx.x] = s + comp;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    double M[4][5], coef[4];
    int t = 0;
    for (int j = 0; j < 4; ++j) for (int c = j; c < 4; ++c) { M[j][c] = tot[t]; M[c][j] = tot[t]; ++t; }
    for (int j = 0; j < 4; ++j) M[j][4] = tot[10 + j];
    double scale = 0.0;
    for (int j = 0; j < 4; ++j) scale = fmax(scale, fabs(M[j][j]));
    for (int col = 0; col < 4; ++col) {
        int piv = col; double best = fabs(M[col][col]);
        for (int r = col + 1; r < 4; ++r) if (fabs(M[r][col]) > best) { best = fabs(M[r][col]); piv = r; }
        if (!(best > sing_rel_tol * scale) || !isfinite(best)) { t_out[k] = nan(""); return; }
        if (piv != col) for (int c = 0; c < 5; ++c) { double tmp = M[col][c]; M[col][c] = M[piv][c]; M[piv][c] = tmp; }
        for (int r = col + 1; r < 4; ++r) { const double f = M[r][col] / M[col][col]; for (int c = col; c < 5; ++c) M[r][c] -= f * M[col][c]; }
    }
    for (int r = 3; r >= 0; --r) { double s = M[r][4]; for (int c = r + 1; c < 4; ++c) s -= M[r][c] * coef[c]; coef[r] = s / M[r][r]; }
    t_out[k] = (coef[3] != 0.0) ? coef[2] / coef[3] - mu[k * 3] : nan("");
}
// replay / candidate generation, column-major (C = K*G rows): out[c*n+i] = scrub((U[k]+T[c])*V[k]), k = c / G
__global__ void offset_gen_cm(const double* U, const double* V, const double* T, const long long n, const int G, const int C, double* out) {
    const long long total = (long long)C * n;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long long)gridDim.x * blockDim.x) {
        const int c = (int)(idx / n); const long long i = idx - (long long)c * n; const int k = c / G;
        double r = (U[(long long)k * n + i] + T[c]) * V[(long long)k * n + i];
        out[idx] = isfinite(r) ? r : 0.0;
    }
}
}
"""


def emulate_gram4_cuda(U, V, yr, nblk=NBLK, threads=CUDA_THREADS):
    """numpy emulation of gram4_partials + solve4_shift with the SAME summation order (strided lanes, shfl_down tree,
    in-order warp sum, Neumaier merge, partial-pivot elimination).  Returns (K,) t."""
    K, n = U.shape
    my = float(np.mean(yr))
    out = np.empty(K)
    for k in range(K):
        mu_u, mu_v = float(np.mean(U[k])), float(np.mean(V[k]))
        f1 = U[k] - mu_u
        f2 = V[k] - mu_v
        f3 = f1 * f2
        yy = yr - my
        feats = [np.ones(n), f1, f2, f3, f1 * f1, f1 * f2, f1 * f3, f2 * f2, f2 * f3, f3 * f3, yy, f1 * yy, f2 * yy, f3 * yy]
        part = np.zeros((nblk, 14))
        stride = nblk * threads
        for b in range(nblk):
            acc = np.zeros((threads, 14))
            for s in range(0, n, stride):
                idx = b * threads + np.arange(threads) + s
                ok = idx < n
                for a in range(14):
                    acc[ok, a] += feats[a][idx[ok]]
            for a in range(14):
                w = acc[:, a].reshape(8, 32).copy()
                for o in (16, 8, 4, 2, 1):
                    sh = w.copy()
                    sh[:, : 32 - o] = w[:, o:]  # lanes >= 32-o keep their own value (shfl_down semantics)
                    w = w + sh
                s_ = 0.0
                for wi in range(8):
                    s_ += w[wi, 0]
                part[b, a] = s_
        tot = np.empty(14)
        for a in range(14):
            s = 0.0
            comp = 0.0
            for b in range(nblk):
                x = part[b, a]
                t = s + x
                comp += ((s - t) + x) if abs(s) >= abs(x) else ((x - t) + s)
                s = t
            tot[a] = s + comp
        M = np.zeros((4, 5))
        t = 0
        for j in range(4):
            for c in range(j, 4):
                M[j, c] = M[c, j] = tot[t]
                t += 1
        M[:, 4] = tot[10:14]
        scale = max(abs(M[j, j]) for j in range(4))
        bad = False
        for col in range(4):
            piv = col
            best = abs(M[col, col])
            for r in range(col + 1, 4):
                if abs(M[r, col]) > best:
                    best = abs(M[r, col])
                    piv = r
            if not (best > SING_REL_TOL * scale) or not np.isfinite(best):
                bad = True
                break
            if piv != col:
                M[[col, piv]] = M[[piv, col]]
            for r in range(col + 1, 4):
                f = M[r, col] / M[col, col]
                M[r, col:] -= f * M[col, col:]
        if bad:
            out[k] = np.nan
            continue
        coef = np.zeros(4)
        for r in range(3, -1, -1):
            s = M[r, 4]
            for c in range(r + 1, 4):
                s -= M[r, c] * coef[c]
            coef[r] = s / M[r, r]
        out[k] = coef[2] / coef[3] - mu_u if coef[3] != 0 else np.nan
    return out
