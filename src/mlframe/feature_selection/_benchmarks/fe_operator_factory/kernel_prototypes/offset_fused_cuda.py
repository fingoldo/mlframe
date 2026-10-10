"""Fused recompute-instead-of-store CUDA kernels for the offset-product family + numpy emulation.

NEVER RUN ON A GPU: the CUDA source below was designed and reviewed but never compiled or executed on a device; only the pure-numpy emulation (``emulate_offset_fused``,
``emulate_null``) is exercised, in tests/feature_selection/fe/factory. Treat the CUDA source as a design, not a validated kernel.

Family:  MODE_SHIFT (0): cand = (u + p0) * v                         (p = (t, 0, 0))   <- bit-parity with the CPU reference
         MODE_MIX   (1): cand = ((alpha*u)*v + beta*v) + gamma*u      (p = (alpha, beta, gamma)), rounded ops (no fma contraction)
Both scrub non-finite -> 0.0.  One block per candidate c = k*G + g (k = c / G picks the (u_k, v_k) column pair).
Pass 1: 8 byte-radix passes on the order-preserving u64 key of the REGENERATED candidate (same transform as
        _gpu_resident_select_kernels.radix_select_interp_f64_v3); integer smem atomics only -> deterministic.
Edges : a + (b-a)*frac from the exact order statistics at the host-provided ranks/fracs (same arithmetic as the CPU reference).
Pass 2: regenerate, code = #edges <= x (upper bound), integer atomicAdd joint histogram in smem; thread 0 sums the MI in the CPU loop order.
Output: mi[c] (+ optional edges[c, nbins-1]).  No (n x K*G) matrix anywhere; only per-block smem.
"""

import numpy as np

MODE_SHIFT = 0
MODE_MIX = 1
MAXR = 64  # max ranks = 2*(nbins-1)
MAXW = 64  # max distinct prefix windows

_OFFSET_FUSED_SRC = r"""
#define MAXR 64
#define MAXW 64
__device__ __forceinline__ double cand_val(const double* u, const double* v, const long long i, const double p0, const double p1, const double p2, const int mode) {
    double r;
    if (mode == 0) r = __dmul_rn(__dadd_rn(u[i], p0), v[i]);
    else r = __dadd_rn(__dadd_rn(__dmul_rn(__dmul_rn(p0, u[i]), v[i]), __dmul_rn(p1, v[i])), __dmul_rn(p2, u[i]));
    return isfinite(r) ? r : 0.0;
}
__device__ __forceinline__ unsigned long long key_of(const double d) {
    unsigned long long k = (unsigned long long)__double_as_longlong(d);
    return (k & 0x8000000000000000ULL) ? ~k : (k | 0x8000000000000000ULL);
}
__device__ __forceinline__ double val_of(unsigned long long k) {
    k = (k & 0x8000000000000000ULL) ? (k & 0x7FFFFFFFFFFFFFFFULL) : ~k;
    return __longlong_as_double((long long)k);
}
extern "C" __global__
void offset_fused_mi(const double* __restrict__ U, const double* __restrict__ V, const double* __restrict__ P /*C x 3*/,
                     const int G, const long long n, const int* __restrict__ y, const int ncls, const int nbins,
                     const long long* __restrict__ ranks /*R=2*(nbins-1), ascending*/, const double* __restrict__ fr /*nbins-1*/,
                     const int mode, double* __restrict__ mi_out, double* __restrict__ edges_out /*C x (nbins-1) or null*/) {
    const int c = blockIdx.x, tid = threadIdx.x, nt = blockDim.x, k = c / G;
    const int ne = nbins - 1, R = 2 * ne;
    const double p0 = P[3 * c], p1 = P[3 * c + 1], p2 = P[3 * c + 2];
    const double* u = U + (long long)k * n;
    const double* v = V + (long long)k * n;
    extern __shared__ unsigned int sh[];                    // pass 1: W*256 digit counters ; pass 2: nbins*ncls joint + ncls + nbins counters
    __shared__ unsigned long long prefix[MAXR], below[MAXR], wpref[MAXW];
    __shared__ int rank2w[MAXR];
    __shared__ int W;
    __shared__ double osv[MAXR];
    __shared__ double edges[MAXR];
    if (tid < R) { prefix[tid] = 0ULL; below[tid] = 0ULL; }
    __syncthreads();
    for (int byte = 7; byte >= 0; --byte) {
        const int shift = byte * 8;
        const unsigned long long hmask = (byte == 7) ? 0ULL : (0xFFFFFFFFFFFFFFFFULL << ((byte + 1) * 8));
        if (tid == 0) {
            int w_ = 0;
            for (int r = 0; r < R; ++r) {
                const unsigned long long p = prefix[r] & hmask; int f = -1;
                for (int q = 0; q < w_; ++q) if (wpref[q] == p) { f = q; break; }
                if (f < 0) { wpref[w_] = p; rank2w[r] = w_; ++w_; } else rank2w[r] = f;
            }
            W = w_;
        }
        __syncthreads();
        const int Wl = W;
        for (int s = tid; s < Wl * 256; s += nt) sh[s] = 0u;
        __syncthreads();
        for (long long i = tid; i < n; i += nt) {                 // REGENERATE the candidate; nothing is stored
            const unsigned long long kk = key_of(cand_val(u, v, i, p0, p1, p2, mode));
            const unsigned long long pm = kk & hmask;
            for (int w = 0; w < Wl; ++w) if (wpref[w] == pm) { atomicAdd(&sh[w * 256 + (int)((kk >> shift) & 0xFFULL)], 1u); break; }
        }
        __syncthreads();
        if (tid < R) {
            const int w2 = rank2w[tid]; unsigned long long acc = below[tid]; int chosen = 0; const long long want = ranks[tid];
            for (int b = 0; b < 256; ++b) { const unsigned long long cn = sh[w2 * 256 + b]; if (acc + cn > (unsigned long long)want) { chosen = b; break; } acc += cn; }
            prefix[tid] |= ((unsigned long long)chosen) << shift; below[tid] = acc;
        }
        __syncthreads();
    }
    if (tid < R) osv[tid] = val_of(prefix[tid]);
    __syncthreads();
    if (tid < ne) {
        const double a = osv[2 * tid], b = osv[2 * tid + 1];
        edges[tid] = a + (b - a) * fr[tid];                      // no fma: compile with --fmad=false (cp.RawKernel(options=("--fmad=false",)))
        if (edges_out) edges_out[(long long)c * ne + tid] = edges[tid];
    }
    __syncthreads();
    unsigned int* hxy = sh;                                      // nbins*ncls
    unsigned int* hy = sh + nbins * ncls;                        // ncls
    unsigned int* hx = hy + ncls;                                // nbins
    for (int s = tid; s < nbins * ncls + ncls + nbins; s += nt) sh[s] = 0u;
    __syncthreads();
    for (long long i = tid; i < n; i += nt) {
        const double x = cand_val(u, v, i, p0, p1, p2, mode);
        int lo = 0, hi = ne;
        while (lo < hi) { const int mid = (lo + hi) >> 1; if (x < edges[mid]) hi = mid; else lo = mid + 1; }
        const int yy = y[i];
        atomicAdd(&hxy[lo * ncls + yy], 1u); atomicAdd(&hx[lo], 1u); atomicAdd(&hy[yy], 1u);
    }
    __syncthreads();
    if (tid == 0) {                                              // identical loop order to the CPU reference -> same rounding
        const double dn = (double)n, log_n = log(dn); double mi = 0.0;
        for (int b = 0; b < nbins; ++b) {
            if (hx[b] == 0u) continue;
            const double log_hx = log((double)hx[b]);
            for (int cl = 0; cl < ncls; ++cl) {
                const unsigned int nxy = hxy[b * ncls + cl];
                if (nxy == 0u || hy[cl] == 0u) continue;
                mi += ((double)nxy / dn) * (log((double)nxy) + log_n - log_hx - log((double)hy[cl]));
            }
        }
        mi_out[c] = mi > 0.0 ? mi : 0.0;
    }
}
// deterministic first-max over C candidates (one block): winner index and value
extern "C" __global__ void argmax_first(const double* mi, const int C, int* best_idx, double* best_val) {
    __shared__ double sv[256]; __shared__ int si[256];
    double bv = -1.0; int bi = 0;
    for (int c = threadIdx.x; c < C; c += blockDim.x) if (mi[c] > bv) { bv = mi[c]; bi = c; }   // strided scan keeps the lowest index per lane
    sv[threadIdx.x] = bv; si[threadIdx.x] = bi; __syncthreads();
    if (threadIdx.x == 0) { double v0 = sv[0]; int i0 = si[0];
        for (int t = 1; t < blockDim.x; ++t) if (sv[t] > v0 || (sv[t] == v0 && si[t] < i0)) { v0 = sv[t]; i0 = si[t]; }
        *best_idx = i0; *best_val = v0; }
}
// permutation null, recompute variant: grid = (C, B). Block (c, b) re-bins candidate c against shuffled y row b; edges_in from offset_fused_mi.
extern "C" __global__
void offset_fused_null(const double* __restrict__ U, const double* __restrict__ V, const double* __restrict__ P, const int G, const long long n,
                       const signed char* __restrict__ Yperm /*B x n*/, const int ncls, const int nbins, const double* __restrict__ edges_in,
                       const int mode, double* __restrict__ mi_perm /*B x C*/) {
    const int c = blockIdx.x, b = blockIdx.y, C = gridDim.x, tid = threadIdx.x, nt = blockDim.x, k = c / G, ne = nbins - 1;
    const double p0 = P[3 * c], p1 = P[3 * c + 1], p2 = P[3 * c + 2];
    const double* u = U + (long long)k * n; const double* v = V + (long long)k * n;
    const signed char* yp = Yperm + (long long)b * n;
    extern __shared__ unsigned int sh[];
    __shared__ double edges[MAXR];
    if (tid < ne) edges[tid] = edges_in[(long long)c * ne + tid];
    unsigned int* hxy = sh; unsigned int* hy = sh + nbins * ncls; unsigned int* hx = hy + ncls;
    for (int s = tid; s < nbins * ncls + ncls + nbins; s += nt) sh[s] = 0u;
    __syncthreads();
    for (long long i = tid; i < n; i += nt) {
        const double x = cand_val(u, v, i, p0, p1, p2, mode);
        int lo = 0, hi = ne; while (lo < hi) { const int mid = (lo + hi) >> 1; if (x < edges[mid]) hi = mid; else lo = mid + 1; }
        const int yy = yp[i]; atomicAdd(&hxy[lo * ncls + yy], 1u); atomicAdd(&hx[lo], 1u); atomicAdd(&hy[yy], 1u);
    }
    __syncthreads();
    if (tid == 0) {
        const double dn = (double)n, log_n = log(dn); double mi = 0.0;
        for (int bb = 0; bb < nbins; ++bb) { if (hx[bb] == 0u) continue; const double lhx = log((double)hx[bb]);
            for (int cl = 0; cl < ncls; ++cl) { const unsigned int nxy = hxy[bb * ncls + cl]; if (nxy == 0u || hy[cl] == 0u) continue;
                mi += ((double)nxy / dn) * (log((double)nxy) + log_n - lhx - log((double)hy[cl])); } }
        mi_perm[(long long)b * C + c] = mi > 0.0 ? mi : 0.0;
    }
}
"""


def launch_hint(nbins, ncls, nthreads=256):
    """shared bytes for offset_fused_mi: max(pass-1 W*nthreads*4 (W<=MAXW), pass-2 (nbins*ncls+ncls+nbins)*4)."""
    return max(MAXW * nthreads * 4 if 2 * (nbins - 1) > 18 else 2 * (nbins - 1) * nthreads * 4, (nbins * ncls + ncls + nbins) * 4)


# ------------------------------------------------------------------------------------------------ numpy emulation
def _cand_np(u, v, p, mode):
    """Candidate values for one parameter row in the same rounding order as the CUDA kernel; non-finite scrubbed to 0."""
    if mode == MODE_SHIFT:
        r = (u + p[0]) * v
    else:
        r = ((p[0] * u) * v + p[1] * v) + p[2] * u
    return np.where(np.isfinite(r), r, 0.0)


def _key_np(d):
    """Order-preserving unsigned 64-bit key of float64 values (the radix-select transform)."""
    k = d.astype(np.float64).view(np.uint64).copy()
    sign = np.uint64(0x8000000000000000)
    return np.where((k & sign) != 0, ~k, k | sign)


def _val_np(k):
    """Inverse of ``_key_np`` for one key."""
    sign = np.uint64(0x8000000000000000)
    k = np.uint64(k)
    k = (k & ~sign) if (k & sign) else ~k
    return np.array([k], dtype=np.uint64).view(np.float64)[0]


def emulate_offset_fused(U, V, P, G, y, ncls, nbins, mode=MODE_SHIFT):
    """Mirror of offset_fused_mi: returns (mi (C,), edges (C, nbins-1)).  Digit histograms use bincount (integers, order-free)."""
    K, n = U.shape
    C = P.shape[0]
    ne = nbins - 1
    nq = nbins + 1
    ranks, fr = [], []
    for kq in range(1, nbins):
        pos = (kq / (nq - 1)) * (n - 1)
        lo = int(np.floor(pos))
        hi = lo + 1 if lo < n - 1 else lo
        ranks += [lo, hi]
        fr.append(pos - lo)
    R = len(ranks)
    mi = np.empty(C)
    edges_all = np.empty((C, ne))
    for c in range(C):
        k = c // G
        x = _cand_np(U[k], V[k], P[c], mode)
        keys = _key_np(x)
        prefix = np.zeros(R, dtype=np.uint64)
        below = np.zeros(R, dtype=np.int64)
        for byte in range(7, -1, -1):
            shift = np.uint64(byte * 8)
            hmask = np.uint64(0) if byte == 7 else np.uint64((0xFFFFFFFFFFFFFFFF << ((byte + 1) * 8)) & 0xFFFFFFFFFFFFFFFF)
            wpref = []
            rank2w = []
            for r in range(R):
                p = prefix[r] & hmask
                if p in wpref:
                    rank2w.append(wpref.index(p))
                else:
                    wpref.append(p)
                    rank2w.append(len(wpref) - 1)
            pm = keys & hmask
            dig = ((keys >> shift) & np.uint64(0xFF)).astype(np.int64)
            hist = np.zeros((len(wpref), 256), dtype=np.int64)
            for w, pw in enumerate(wpref):
                hist[w] = np.bincount(dig[pm == pw], minlength=256)
            for r in range(R):
                acc = below[r]
                chosen = 0
                for b in range(256):
                    cn = hist[rank2w[r], b]
                    if acc + cn > ranks[r]:
                        chosen = b
                        break
                    acc += cn
                prefix[r] |= np.uint64(chosen) << shift
                below[r] = acc
        osv = np.array([_val_np(prefix[r]) for r in range(R)])
        e = np.array([osv[2 * j] + (osv[2 * j + 1] - osv[2 * j]) * fr[j] for j in range(ne)])
        edges_all[c] = e
        codes = np.searchsorted(e, x, side="right")
        hxy = np.zeros((nbins, ncls), dtype=np.int64)
        np.add.at(hxy, (codes, y), 1)
        hx = hxy.sum(1)
        hy = hxy.sum(0)
        log_n = np.log(float(n))
        m = 0.0
        for b in range(nbins):
            if hx[b] == 0:
                continue
            lhx = np.log(float(hx[b]))
            for cl in range(ncls):
                nxy = hxy[b, cl]
                if nxy == 0 or hy[cl] == 0:
                    continue
                m += (float(nxy) / n) * (np.log(float(nxy)) + log_n - lhx - np.log(float(hy[cl])))
        mi[c] = max(m, 0.0)
    return mi, edges_all


def emulate_null(U, V, P, G, Yperm, ncls, nbins, edges, mode=MODE_SHIFT):
    """Mirror of offset_fused_null: (B, C) permuted MI."""
    B = Yperm.shape[0]
    C = P.shape[0]
    n = U.shape[1]
    out = np.zeros((B, C))
    for c in range(C):
        x = _cand_np(U[c // G], V[c // G], P[c], mode)
        codes = np.searchsorted(edges[c], x, side="right")
        for b in range(B):
            hxy = np.zeros((nbins, ncls))
            np.add.at(hxy, (codes, Yperm[b].astype(np.int64)), 1)
            hx = hxy.sum(1)
            hy = hxy.sum(0)
            nz = hxy > 0
            out[b, c] = max(
                float((hxy[nz] / n * (np.log(hxy[nz]) + np.log(n) - np.log(hx[:, None].repeat(ncls, 1)[nz]) - np.log(hy[None, :].repeat(nbins, 0)[nz]))).sum()),
                0.0,
            )
    return out
