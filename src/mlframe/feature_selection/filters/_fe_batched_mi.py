"""Batched born-on-device MI / CMI for the FE candidate pool (replatform step 3, 2026-06-25).

PARALLEL / gated module -- imported by nothing in production yet; it lets the batched device CMI be
built + parity-validated against the per-call CPU path (``_mi_greedy_cmi_fe._cmi_from_binned``) WITHOUT
touching the primary path, then wired under MLFRAME_FE_GPU_STRICT.

The per-call cp.unique CMI ports (cmi_from_binned / fixed_yz) are launch-bound: nsys measured 118k kernel
launches + 5.8k H2D in one F2 1M fit because CMI is called per-candidate-per-permutation, each a separate
cp.unique (5-10 cub launches) + H2D. ``batched_cmi_gpu`` scores CMI(x_k; y | z) for EVERY candidate column
in ONE workload: the y/z-only terms (H(Z), H(Y,Z)) are computed once, then TWO cp.bincount passes over the
flat (k, cx, c*) joint index give H(X_k,Z) and H(X_k,Y,Z) for all k at once -> no per-candidate launch,
one H2D of the (n,K) candidate code matrix. Same Miller-Madow plug-in CMI as the CPU path; partition-based
-> value-order codes are fine -> selection-equivalent (fp reduction order ~1e-15).
"""
from __future__ import annotations

import os

import numpy as np


def _assert_codes_in_range(arr, K: int, name: str, codes_trusted: bool = False) -> None:
    """Guard integer code inputs to the device histogram kernels against out-of-range codes.

    The fused kernels use a code value DIRECTLY as a shared-memory / flat-histogram offset
    (e.g. ``sh[lo*Ky + y[i]]``, ``(col*Kx+xv)*Kb+b[row]``, ``(x*Ky+y)*Kz+zi``). The histogram
    width is sized from ``max()+1`` only, so any code < 0 (a -1 missing/sentinel) or >= K writes
    OUTSIDE the allocated histogram -> cudaErrorIllegalAddress (a hard GPU crash, not a Python
    error). The njit reference (_hermite_fe_mi) guards this exact class explicitly; mirror it here.

    ``codes_trusted`` (FIX1, 2026-06-28): when the caller KNOWS the codes are binner-produced
    (``_gpu_quantile_bin_codes`` / radix / rank always emit dense 0..K-1) the guard is a pure cost --
    it cannot fire, but on a device array it forces TWO blocking ``.item()`` syncs (cp.min + cp.max,
    ~5ms each on a GTX 1050 Ti) at every batched-MI entry. Trusted callers pass True to skip it,
    dropping the guard to ~0 on the resident hot path; untrusted/external code arrays keep the check
    (and the raise contract). For untrusted DEVICE arrays the min/max are computed in ONE stacked
    reduction + ONE ``.get()`` (a single blocking sync instead of two).

    Raises ValueError so an upstream -1 sentinel surfaces as a clear error instead of a GPU
    illegal-address crash. The resident binner never emits negative codes, so on the happy path this
    only ever fires on a genuine upstream bug.
    """
    if codes_trusted:
        return
    try:
        import cupy as cp
        is_dev = isinstance(arr, cp.ndarray)
        xp = cp if is_dev else np
    except Exception:
        cp = None
        is_dev = False
        xp = np
    if getattr(arr, "size", 1) == 0:
        return
    if is_dev:
        # ONE blocking sync: stack min+max into a 2-vector and a single .get(), not two .item() syncs.
        _lh = cp.stack((xp.min(arr), xp.max(arr))).get()
        lo = int(_lh[0])
        hi = int(_lh[1])
    else:
        lo = int(xp.min(arr))
        hi = int(xp.max(arr))
    if lo < 0:
        raise ValueError(
            "%s contains a negative integer code (min=%d); codes must be 0-based in [0, %d). A -1 "
            "missing/sentinel would index outside the device histogram (illegal address)." % (name, lo, int(K))
        )
    if hi >= int(K):
        raise ValueError(
            "%s code out of range (max=%d >= K=%d); a code >= histogram width would index outside "
            "the device histogram (illegal address)." % (name, hi, int(K))
        )


# FUSED MI-FROM-CODES RawKernel (launch-reduction rewrite, 2026-06-25). ONE launch computes plug-in
# MI(col_k; y) for EVERY column of an (n,K) int code matrix, replacing the cupy chain bincount + reshape
# + sum(axis) x2 + where + log + multiply + sum (~7 cuLaunchKernel) with a single kernel launch (also via
# cuLaunchKernel -> a genuine COUNT reduction, not a driver/runtime counter shift). One block per column:
# the column's (Kx, Ky) joint histogram lives in shared int32 (counts <= n < 2^31); threads stride rows
# and atomicAdd; then a single-thread reduction computes the plug-in MI from the small joint table. The
# math is identical to ``batched_binned_mi_gpu`` (plain plug-in MI, no MM bias) -> selection-equivalent
# (fp reduction order ~1e-15). Used only when the shared tile (Kx*Ky int32) fits; else the cupy path.
_MI_FROM_CODES_SRC = r"""
extern "C" __global__
void mi_from_codes(const long long* __restrict__ codes,   // (n, K) row-major int codes in [0,Kx)
                   const long long* __restrict__ y,        // (n,) int codes in [0,Ky)
                   const long long n, const int K, const int Kx, const int Ky,
                   const double inv_n, double* __restrict__ mi_out) {
    extern __shared__ int sh[];          // (Kx*Ky) joint histogram for this column
    int c = blockIdx.x;
    if (c >= K) return;
    int M = Kx * Ky;
    int tid = threadIdx.x, nt = blockDim.x;
    for (int s = tid; s < M; s += nt) sh[s] = 0;
    __syncthreads();
    for (long long i = tid; i < n; i += nt) {
        long long cx = codes[i * (long long)K + c];
        long long cy = y[i];
        atomicAdd(&sh[cx * Ky + cy], 1);
    }
    __syncthreads();
    if (tid == 0) {
        double mi = 0.0;
        for (int xx = 0; xx < Kx; ++xx) {
            long long rx = 0;
            for (int yy = 0; yy < Ky; ++yy) rx += sh[xx * Ky + yy];
            if (rx == 0) continue;
            double px = (double)rx * inv_n;
            for (int yy = 0; yy < Ky; ++yy) {
                long long nxy = sh[xx * Ky + yy];
                if (nxy == 0) continue;
                long long ry = 0;
                for (int xx2 = 0; xx2 < Kx; ++xx2) ry += sh[xx2 * Ky + yy];
                double pxy = (double)nxy * inv_n;
                double py = (double)ry * inv_n;
                mi += pxy * log(pxy / (px * py));
            }
        }
        mi_out[c] = mi > 0.0 ? mi : 0.0;
    }
}
"""
# FUSED VALUES->BIN->HIST->MI RawKernel (mega-fusion, 2026-06-25). For an (n,K) FLOAT matrix + per-column
# interior quantile edges, ONE launch bins each value in-kernel (binary-search upper_bound == cp.searchsorted
# side='right') AND builds the joint histogram AND computes plug-in MI -- replacing the separate
# _searchsorted_codes kernel + the (n,K) int code array + binned_mi_from_codes_gpu. One block per column,
# shared (nbins*Ky) int hist. Bin codes equal _searchsorted_codes bit-for-bit (same f64 edges, same
# side='right') -> selection-equivalent. edges: (nbins-1, K) row-major (interior); X: (n,K) row-major.
_MI_FROM_VALUES_SRC = r"""
extern "C" __global__
void mi_from_values(const double* __restrict__ X, const double* __restrict__ edges,
                    const long long* __restrict__ y, const long long n, const int K,
                    const int nbins, const int Ky, const double inv_n, double* __restrict__ mi_out) {
    extern __shared__ int sh[];          // (nbins*Ky) joint histogram for this column
    int c = blockIdx.x;
    if (c >= K) return;
    int ne = nbins - 1, M = nbins * Ky;
    int tid = threadIdx.x, nt = blockDim.x;
    for (int s = tid; s < M; s += nt) sh[s] = 0;
    __syncthreads();
    for (long long i = tid; i < n; i += nt) {
        double v = X[i * (long long)K + c];
        int lo = 0, hi = ne;                          // upper_bound: count of interior edges <= v
        while (lo < hi) { int mid = (lo + hi) >> 1; if (edges[(long long)mid * K + c] <= v) lo = mid + 1; else hi = mid; }
        atomicAdd(&sh[lo * Ky + y[i]], 1);
    }
    __syncthreads();
    if (tid == 0) {
        double mi = 0.0;
        for (int xx = 0; xx < nbins; ++xx) {
            long long rx = 0;
            for (int yy = 0; yy < Ky; ++yy) rx += sh[xx * Ky + yy];
            if (rx == 0) continue;
            double px = (double)rx * inv_n;
            for (int yy = 0; yy < Ky; ++yy) {
                long long nxy = sh[xx * Ky + yy];
                if (nxy == 0) continue;
                long long ry = 0;
                for (int xx2 = 0; xx2 < nbins; ++xx2) ry += sh[xx2 * Ky + yy];
                mi += (double)nxy * inv_n * log(((double)nxy * inv_n) / (px * ((double)ry * inv_n)));
            }
        }
        mi_out[c] = mi > 0.0 ? mi : 0.0;
    }
}
"""
_MI_FROM_VALUES_KERNEL = None

# FUSED VALUES->BIN->HIST->MILLER-MADOW-MI (2026-06-26). Same one-block-per-column fused bin+joint-hist as
# mi_from_values, but emits the Miller-Madow-corrected MARGINAL MI matching _usability_njit_pool's
# _marginal_mi_njit / _gpu_marginal_mi: mi = H(x)+H(y)-H(x,y) - (kx_occ + k_y - kxy - 1)/(2n), clamped >=0.
# h_y / k_y are the shared (fit-constant) target entropy + class count, passed in (computed once). This is the
# sync-free, batched, MM-correct kernel the resident pair-combo MI table needs (replaces the per-row
# cp.bincount + .max() sync loop in _gpu_marginal_mi). Selection-equivalent to the njit table (plug-in
# entropies in fp64; reduction-order ~1e-15).
_MI_MM_FROM_VALUES_SRC = r"""
extern "C" __global__
void mi_mm_from_values(const double* __restrict__ X, const double* __restrict__ edges,
                       const long long* __restrict__ y, const long long n, const int K, const int nbins,
                       const int Ky, const double inv_n, const double h_y, const int k_y,
                       double* __restrict__ mi_out) {
    extern __shared__ int sh[];          // (nbins*Ky) joint histogram for this column
    int c = blockIdx.x;
    if (c >= K) return;
    int ne = nbins - 1, M = nbins * Ky;
    int tid = threadIdx.x, nt = blockDim.x;
    for (int s = tid; s < M; s += nt) sh[s] = 0;
    __syncthreads();
    for (long long i = tid; i < n; i += nt) {
        double v = X[i * (long long)K + c];
        int lo = 0, hi = ne;                          // upper_bound == cp.searchsorted side='right'
        while (lo < hi) { int mid = (lo + hi) >> 1; if (edges[(long long)mid * K + c] <= v) lo = mid + 1; else hi = mid; }
        atomicAdd(&sh[lo * Ky + y[i]], 1);
    }
    __syncthreads();
    if (tid == 0) {
        double hx = 0.0, hxy = 0.0;
        int kx_occ = 0, kxy = 0;
        for (int xx = 0; xx < nbins; ++xx) {
            long long rx = 0;
            for (int yy = 0; yy < Ky; ++yy) { int nxy = sh[xx * Ky + yy]; rx += nxy; if (nxy > 0) kxy++; }
            if (rx == 0) continue;
            kx_occ++;
            double px = (double)rx * inv_n; hx -= px * log(px);
            for (int yy = 0; yy < Ky; ++yy) { int nxy = sh[xx * Ky + yy]; if (nxy > 0) { double pxy = (double)nxy * inv_n; hxy -= pxy * log(pxy); } }
        }
        double mi = hx + h_y - hxy - ((double)(kx_occ + k_y - kxy - 1)) * 0.5 * inv_n;
        mi_out[c] = mi > 0.0 ? mi : 0.0;
    }
}
"""
_MI_MM_FROM_VALUES_KERNEL = None

# NJIT-PARITY EDGE DEDUP for low-cardinality columns (2026-06-27). The fused mi_mm_from_values binary-searches
# over the FULL (nbins-1) interior radix edges; on a LOW-CARDINALITY / discrete column those edges contain
# DUPLICATES and BOUNDARY values (== the column min/max), which the njit reference _qbin_into does NOT: it
# dedups the FULL (nbins+1)-level quantile set (np.unique, INCLUDING the level-0 min and level-nbins max
# quantiles) then bins on only the strictly-interior distinct thresholds. The level-0/level-nbins quantiles
# are exactly the column min/max, so njit's deduped threshold set is reconstructed on device per column as:
#   de = adjacent-unique([cmin] + interior_edges + [cmax])   (the edges are ascending)
#   thresholds = de[1:]; drop the LAST element iff that leaves >= 2 entries
#   (== njit de[1:-1] for m>=3 distinct values, == [de[1]] for m==2, == [] for m<=1)
# Binary-searching only this compacted per-column prefix (length ne_k) yields the SAME data partition as njit
# -> identical occupied-bin entropies -> identical MM-MI (verified bit-faithful CODES on discrete/continuous/
# constant columns). Continuous columns have no duplicate/boundary interior edges so ne_k == nbins-1 and the
# binning -> MI is byte-for-byte unchanged from the no-dedup path.
_DEDUP_EDGES_SRC = r"""
extern "C" __global__
void dedup_njit_edges(const double* __restrict__ edges, const double* __restrict__ cmin,
                      const double* __restrict__ cmax, const int ne, const int K,
                      double* __restrict__ out, int* __restrict__ ne_out) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;   // one thread per column
    if (c >= K) return;
    // de[0] is cmin; emit de[1:] (everything strictly AFTER the first distinct value) into out[*, c].
    double prev = cmin[c];
    int w = 0;
    for (int e = 0; e < ne; ++e) {
        double v = edges[(long long)e * K + c];
        if (v != prev) { out[(long long)w * K + c] = v; w++; prev = v; }
    }
    double hv = cmax[c];
    if (hv != prev) { out[(long long)w * K + c] = hv; w++; prev = hv; }
    if (w >= 2) w -= 1;          // drop trailing de[-1] for m>=3 (njit de[1:-1]); keep [de[1]] for m==2
    ne_out[c] = w;
}
"""
_DEDUP_EDGES_KERNEL = None


def _get_dedup_edges_kernel():
    global _DEDUP_EDGES_KERNEL
    if _DEDUP_EDGES_KERNEL is None:
        import cupy as cp
        _DEDUP_EDGES_KERNEL = cp.RawKernel(_DEDUP_EDGES_SRC, "dedup_njit_edges")
    return _DEDUP_EDGES_KERNEL


# Length-aware twin of mi_mm_from_values: each column binary-searches only its VALID prefix ne_k[c] of the
# (dedup'd) interior edges -- everything else is byte-for-byte the mi_mm_from_values kernel. When ne_k[c] ==
# nbins-1 (continuous columns, no dup/boundary edges) it is bit-identical to mi_mm_from_values.
_MI_MM_FROM_VALUES_NEK_SRC = (
    _MI_MM_FROM_VALUES_SRC
    .replace("void mi_mm_from_values(", "void mi_mm_from_values_nek(")
    .replace(
        "const double h_y, const int k_y,\n                       double* __restrict__ mi_out) {",
        "const double h_y, const int k_y,\n                       const int* __restrict__ ne_k, double* __restrict__ mi_out) {")
    .replace("    int ne = nbins - 1, M = nbins * Ky;", "    int ne = ne_k[c], M = nbins * Ky;")
)
_MI_MM_FROM_VALUES_NEK_KERNEL = None


def _get_mi_mm_from_values_nek_kernel():
    global _MI_MM_FROM_VALUES_NEK_KERNEL
    if _MI_MM_FROM_VALUES_NEK_KERNEL is None:
        import cupy as cp
        _MI_MM_FROM_VALUES_NEK_KERNEL = cp.RawKernel(_MI_MM_FROM_VALUES_NEK_SRC, "mi_mm_from_values_nek")
    return _MI_MM_FROM_VALUES_NEK_KERNEL


def _get_mi_mm_from_values_kernel():
    global _MI_MM_FROM_VALUES_KERNEL
    if _MI_MM_FROM_VALUES_KERNEL is None:
        import cupy as cp
        _MI_MM_FROM_VALUES_KERNEL = cp.RawKernel(_MI_MM_FROM_VALUES_SRC, "mi_mm_from_values")
    return _MI_MM_FROM_VALUES_KERNEL


def binned_mm_mi_from_values_gpu(x_vals, interior_edges, y_codes, nbins, ky, h_y, k_y, codes_trusted: bool = False):
    """Miller-Madow MARGINAL MI(col_k; y) for an (n,K) float matrix binned by per-column ``interior_edges``
    ((nbins-1, K) cupy), in ONE fused RawKernel (bin + joint hist + MM-MI). ``ky`` is the y-cardinality
    (histogram width; y codes in [0, ky)); ``h_y`` / ``k_y`` are the shared target plug-in entropy +
    OCCUPIED class count used in the bias. Returns a host (K,) float64 array (clamped >=0), matching
    _gpu_marginal_mi / _marginal_mi_njit. Returns None if the (nbins*ky) shared tile won't fit."""
    import cupy as cp

    Xd = x_vals.astype(cp.float64, copy=False)
    if Xd.ndim == 1:
        Xd = Xd[:, None]
    n, K = int(Xd.shape[0]), int(Xd.shape[1])
    Ky = int(ky)
    if int(nbins) * Ky * 4 > _MI_FROM_CODES_MAX_SHARED:
        return None
    E = cp.ascontiguousarray(interior_edges.astype(cp.float64, copy=False))
    yv = y_codes.astype(cp.int64, copy=False).ravel() if isinstance(y_codes, cp.ndarray) else cp.asarray(np.ascontiguousarray(y_codes).astype(np.int64).ravel())
    _assert_codes_in_range(yv, Ky, "binned_mm_mi_from_values_gpu y codes", codes_trusted)
    mi_out = cp.empty(K, dtype=cp.float64)
    Xc = cp.ascontiguousarray(Xd)
    # NJIT-PARITY: dedup the per-column interior edges to njit's distinct-threshold set (drops duplicate +
    # boundary edges that over-bin low-cardinality columns), then bin on only each column's valid prefix.
    ne = int(E.shape[0])
    cmin = cp.ascontiguousarray(Xc.min(axis=0).astype(cp.float64))
    cmax = cp.ascontiguousarray(Xc.max(axis=0).astype(cp.float64))
    Ec = cp.zeros((ne, K), dtype=cp.float64)
    ne_k = cp.empty(K, dtype=cp.int32)
    threads = 256
    _get_dedup_edges_kernel()(((K + threads - 1) // threads,), (threads,),
        (E, cmin, cmax, np.int32(ne), np.int32(K), Ec, ne_k))
    _get_mi_mm_from_values_nek_kernel()((K,), (256,),
        (Xc, Ec, yv, np.int64(n), np.int32(K), np.int32(int(nbins)),
         np.int32(Ky), np.float64(1.0 / float(max(1, n))), np.float64(float(h_y)), np.int32(int(k_y)),
         ne_k, mi_out),
        shared_mem=int(nbins) * Ky * 4)
    return cp.asnumpy(mi_out)


def _get_mi_from_values_kernel():
    global _MI_FROM_VALUES_KERNEL
    if _MI_FROM_VALUES_KERNEL is None:
        import cupy as cp
        _MI_FROM_VALUES_KERNEL = cp.RawKernel(_MI_FROM_VALUES_SRC, "mi_from_values")
    return _MI_FROM_VALUES_KERNEL


def binned_mi_from_values_gpu(x_vals, interior_edges, y_codes, nbins: int, ky: int, codes_trusted: bool = False):
    """Plug-in MI(col_k; y) for an (n,K) float matrix ``x_vals`` binned by per-column ``interior_edges``
    ((nbins-1, K) cupy) in ONE fused RawKernel (bin + joint histogram + MI), replacing _searchsorted_codes
    + binned_mi_from_codes_gpu. Returns a host (K,) float64 array. Selection-equivalent (codes match
    cp.searchsorted side='right' bit-for-bit). Falls back to None if the (nbins*ky) shared tile won't fit."""
    import cupy as cp

    Xd = x_vals.astype(cp.float64, copy=False)
    if Xd.ndim == 1:
        Xd = Xd[:, None]
    n, K = int(Xd.shape[0]), int(Xd.shape[1])
    E = cp.ascontiguousarray(interior_edges.astype(cp.float64, copy=False))   # (nbins-1, K)
    yv = y_codes.astype(cp.int64, copy=False).ravel() if isinstance(y_codes, cp.ndarray) else cp.asarray(np.ascontiguousarray(y_codes).astype(np.int64).ravel())
    Ky = int(ky)
    if int(nbins) * Ky * 4 > _MI_FROM_CODES_MAX_SHARED:
        return None
    _assert_codes_in_range(yv, Ky, "binned_mi_from_values_gpu y codes", codes_trusted)
    mi_out = cp.empty(K, dtype=cp.float64)
    _get_mi_from_values_kernel()((K,), (256,), (cp.ascontiguousarray(Xd), E, yv, np.int64(n), np.int32(K),
                                              np.int32(int(nbins)), np.int32(Ky), np.float64(1.0 / float(max(1, n))), mi_out),
                                 shared_mem=int(nbins) * Ky * 4)
    return cp.asnumpy(mi_out)


_MI_FROM_CODES_KERNEL = None   # module-level singleton (lazy-compiled; never on an instance -> pickle-safe)
_MI_FROM_CODES_MAX_SHARED = 44000   # bytes; stay under the 48KB default shared cap (Kx*Ky*4 must fit)


def _get_mi_from_codes_kernel():
    global _MI_FROM_CODES_KERNEL
    if _MI_FROM_CODES_KERNEL is None:
        import cupy as cp
        _MI_FROM_CODES_KERNEL = cp.RawKernel(_MI_FROM_CODES_SRC, "mi_from_codes")
    return _MI_FROM_CODES_KERNEL


def binned_mi_from_codes_gpu(code_cols, y_codes, kx_per_col=None, ky: int = 0):
    """Plug-in MI(col_k; y) for EVERY column of ``code_cols`` (n,K) in ONE fused RawKernel launch.

    Drop-in for ``_wavelet_basis_fe_batched.batched_binned_mi_gpu`` (same plain plug-in MI, no MM bias).
    Falls back to that cupy path when the (Kx*Ky) shared tile would exceed the shared-memory cap. Returns
    a host (K,) float64 array. Accepts a host ndarray or a resident cupy code matrix."""
    import cupy as cp

    C = code_cols if isinstance(code_cols, cp.ndarray) else cp.asarray(np.ascontiguousarray(code_cols).astype(np.int64))
    if C.ndim == 1:
        C = C[:, None]
    C = cp.ascontiguousarray(C.astype(cp.int64, copy=False))
    n, K = int(C.shape[0]), int(C.shape[1])
    y = cp.asarray(np.ascontiguousarray(y_codes).astype(np.int64).ravel()) if not isinstance(y_codes, cp.ndarray) else y_codes.astype(cp.int64, copy=False).ravel()
    Ky = int(ky) if ky > 0 else (int(y.max()) + 1 if y.size else 1)
    Kx = int(np.max(np.asarray(kx_per_col))) if kx_per_col is not None else (int(C.max()) + 1 if C.size else 1)
    Kx = max(Kx, 1)
    if Kx * Ky * 4 > _MI_FROM_CODES_MAX_SHARED:
        from ._wavelet_basis_fe_batched import batched_binned_mi_gpu
        return batched_binned_mi_gpu(C, y, kx_per_col=kx_per_col, ky=Ky)
    mi_out = cp.empty(K, dtype=cp.float64)
    threads = 256
    self_kernel = _get_mi_from_codes_kernel()
    self_kernel((K,), (threads,), (C.ravel(), y, np.int64(n), np.int32(K), np.int32(Kx), np.int32(Ky),
                                   np.float64(1.0 / float(max(1, n))), mi_out),
                shared_mem=Kx * Ky * 4)
    return cp.asnumpy(mi_out)


_XLOGX_ROWS_EK = None


# FUSED per-row entropy + occupied-cell in ONE RawKernel (launch-reduction, 2026-06-25). The prior path was
# _XLOGX_ROWS_EK (1) + .sum(axis=1) (1) + (counts>0).sum(axis=1) (1) = 3 cuLaunchKernel per call; with 2-4
# calls per batched_cmi_gpu it was the residual #1 launch source. One block per row (candidate) grid-strides
# its M cells accumulating (sum xlogx, nnz), a shared-mem tree reduction folds the block, and thread 0 writes
# out_h[row] (negated) + out_k[row]. ONE launch; out_h/out_k are cp.empty (cudaMalloc, not a cuLaunchKernel).
# One-block-per-row -> deterministic tree-order sum (no cross-block atomics); same float64 plug-in entropy /
# occupied-cell definition -> selection-equivalent.
_ROWS_ENT_NNZ_SRC = r"""
extern "C" __global__
void rows_ent_nnz(const double* __restrict__ counts, const double inv_n, const int K, const long long M,
                  double* __restrict__ out_h, long long* __restrict__ out_k) {
    int row = blockIdx.x;
    if (row >= K) return;
    const double* cr = counts + (long long)row * M;
    int tid = threadIdx.x;
    double hloc = 0.0, kloc = 0.0;
    for (long long i = tid; i < M; i += blockDim.x) {
        double ci = cr[i];
        if (ci > 0.0) { double p = ci * inv_n; hloc += p * log(p); kloc += 1.0; }
    }
    __shared__ double sh_h[256];
    __shared__ double sh_k[256];
    sh_h[tid] = hloc; sh_k[tid] = kloc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { out_h[row] = -sh_h[0]; out_k[row] = (long long)(sh_k[0] + 0.5); }
}
"""
_ROWS_ENT_NNZ_KERNEL = None


def _get_rows_ent_nnz_kernel(cp):
    global _ROWS_ENT_NNZ_KERNEL
    if _ROWS_ENT_NNZ_KERNEL is None:
        _ROWS_ENT_NNZ_KERNEL = cp.RawKernel(_ROWS_ENT_NNZ_SRC, "rows_ent_nnz")
    return _ROWS_ENT_NNZ_KERNEL


def _rows_entropy_and_k(counts, inv_n):
    """Per-row (per-candidate) plug-in entropy + occupied-cell count from a (K, M) device count matrix.

    Fused into ONE RawKernel (one block per row, shared-mem reduction) -> a single cuLaunchKernel replacing
    the elementwise + two axis sums. ``counts`` is float64 (K, M). Returns (h (K,) float64, k (K,) int64),
    selection-equivalent to the prior path (deterministic per-row tree-order sum). Falls back to the
    ElementwiseKernel + axis sums on any kernel error."""
    import cupy as cp

    global _XLOGX_ROWS_EK
    try:
        c = counts if counts.dtype == cp.float64 else counts.astype(cp.float64)
        c = cp.ascontiguousarray(c)
        K = int(c.shape[0])
        M = int(c.shape[1]) if c.ndim > 1 else int(c.size)
        out_h = cp.empty(K, dtype=cp.float64)
        out_k = cp.empty(K, dtype=cp.int64)
        threads = 256
        _get_rows_ent_nnz_kernel(cp)((K,), (threads,),
                                     (c, float(inv_n), np.int32(K), np.int64(M), out_h, out_k))
        return out_h, out_k
    except Exception:  # noqa: BLE001
        if _XLOGX_ROWS_EK is None:
            _XLOGX_ROWS_EK = cp.ElementwiseKernel("T c, float64 invn", "float64 o",
                                                  "o = c > 0 ? (c * invn) * log(c * invn) : 0.0", "mrmr_xlogx_rows_ek")
        h = -_XLOGX_ROWS_EK(counts, float(inv_n)).sum(axis=1)
        k = cp.sum(counts > 0, axis=1)
        return h, k


# FUSED joint-histogram RawKernels (launch-reduction, 2026-06-25). The per-call cupy CMI built each joint
# histogram as caller int-arithmetic flat key (dx*Kb+db -> cupy multiply + add) followed by cp.bincount
# (itself scan + two cub_any passes). These kernels compute the flat key IN-KERNEL and atomicAdd into a
# caller-zeroed counts buffer -> ONE launch per joint, no intermediate key array, no bincount cub_any.
# Counts are then reduced by the shared entropy/NNZ ReductionKernels. Same partition counts -> identical
# plug-in entropy -> selection-equivalent.
_JOINT_HIST_SRC = r"""
extern "C" __global__
void joint_hist1(const long long* __restrict__ a, const long long n, long long* __restrict__ counts) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd((unsigned long long*)&counts[a[i]], 1ULL);
}
extern "C" __global__
void joint_hist2(const long long* __restrict__ a, const long long* __restrict__ b,
                 const int Kb, const long long n, long long* __restrict__ counts) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd((unsigned long long*)&counts[a[i] * Kb + b[i]], 1ULL);
}
extern "C" __global__
void joint_hist3(const long long* __restrict__ a, const long long* __restrict__ b,
                 const long long* __restrict__ c, const int Kb, const int Kc,
                 const long long n, long long* __restrict__ counts) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd((unsigned long long*)&counts[(a[i] * Kb + b[i]) * Kc + c[i]], 1ULL);
}
// BATCHED (K columns at once) joint histograms. X is (n,K) row-major int codes; one thread per (row,col)
// cell builds the flat key IN-KERNEL and atomicAdds into the caller-zeroed (K*Kx*Kb,) counts buffer (reshape
// to (K, Kx*Kb) for the entropy reduction). Replaces the per-joint cupy flat-key build
// (X*Kb + b[:,None] + col_off*(Kx*Kb)).ravel() + cp.bincount(scan+cub) with ONE launch, no key array.
extern "C" __global__
void batched_joint_hist2(const long long* __restrict__ X, const long long* __restrict__ b,
                         const int Kx, const int Kb, const long long n, const int K,
                         long long* __restrict__ counts) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (t >= total) return;
    int col = (int)(t % (long long)K);
    long long row = t / (long long)K;
    long long xv = X[row * (long long)K + col];
    atomicAdd((unsigned long long*)&counts[(col * (long long)Kx + xv) * Kb + b[row]], 1ULL);
}
extern "C" __global__
void batched_joint_hist1(const long long* __restrict__ X, const int Kx, const long long n, const int K,
                         long long* __restrict__ counts) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (t >= total) return;
    int col = (int)(t % (long long)K);
    long long row = t / (long long)K;
    long long xv = X[row * (long long)K + col];
    atomicAdd((unsigned long long*)&counts[col * (long long)Kx + xv], 1ULL);
}
// OCCUPIED-CELL count in the SAME pass as the histogram: atomicAdd returns the OLD value, so a cell's
// 0 -> 1 transition (first sample landing there) is detected race-free and bumps a single global nnz
// counter. ONE launch yields the cardinality joint_cardinalities_cupy needs -- no separate _nnz reduction.
extern "C" __global__
void joint_hist_nnz1(const long long* __restrict__ a, const long long n,
                     long long* __restrict__ counts, unsigned long long* __restrict__ nnz) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && atomicAdd((unsigned long long*)&counts[a[i]], 1ULL) == 0ULL) atomicAdd(nnz, 1ULL);
}
extern "C" __global__
void joint_hist_nnz2(const long long* __restrict__ a, const long long* __restrict__ b, const int Kb,
                     const long long n, long long* __restrict__ counts, unsigned long long* __restrict__ nnz) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && atomicAdd((unsigned long long*)&counts[a[i] * Kb + b[i]], 1ULL) == 0ULL) atomicAdd(nnz, 1ULL);
}
extern "C" __global__
void joint_hist_nnz3(const long long* __restrict__ a, const long long* __restrict__ b,
                     const long long* __restrict__ c, const int Kb, const int Kc, const long long n,
                     long long* __restrict__ counts, unsigned long long* __restrict__ nnz) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && atomicAdd((unsigned long long*)&counts[(a[i] * Kb + b[i]) * Kc + c[i]], 1ULL) == 0ULL)
        atomicAdd(nnz, 1ULL);
}
"""
_JOINT_HIST_KERNELS = None


def _get_joint_hist_kernels():
    global _JOINT_HIST_KERNELS
    if _JOINT_HIST_KERNELS is None:
        import cupy as cp
        mod = cp.RawModule(code=_JOINT_HIST_SRC)
        _JOINT_HIST_KERNELS = (mod.get_function("joint_hist1"),
                               mod.get_function("joint_hist2"),
                               mod.get_function("joint_hist3"))
    return _JOINT_HIST_KERNELS


_BATCHED_JOINT_HIST_KERNELS = None


def _get_batched_joint_hist_kernels():
    global _BATCHED_JOINT_HIST_KERNELS
    if _BATCHED_JOINT_HIST_KERNELS is None:
        import cupy as cp
        mod = cp.RawModule(code=_JOINT_HIST_SRC)
        _BATCHED_JOINT_HIST_KERNELS = (mod.get_function("batched_joint_hist1"),
                                       mod.get_function("batched_joint_hist2"))
    return _BATCHED_JOINT_HIST_KERNELS


def _batched_joint_counts2(X, b, Kx, Kb):
    """Per-column joint counts of (n,K) int codes ``X`` with a single (n,) code ``b`` -> (K, Kx*Kb) int64,
    via ONE in-kernel-flat-key atomicAdd launch. Identical counts to
    ``cp.bincount((X*Kb + b[:,None] + col_off*(Kx*Kb)).ravel(), minlength=K*Kx*Kb).reshape(K, Kx*Kb)``."""
    import cupy as cp

    n, K = int(X.shape[0]), int(X.shape[1])
    counts = cp.zeros(int(K) * int(Kx) * int(Kb), dtype=cp.int64)
    _, h2 = _get_batched_joint_hist_kernels()
    threads = 256
    blocks = (n * K + threads - 1) // threads
    h2((blocks,), (threads,), (X, b, np.int32(int(Kx)), np.int32(int(Kb)), np.int64(n), np.int32(K), counts))
    return counts.reshape(K, int(Kx) * int(Kb))


def _batched_marginal_counts(X, Kx):
    """Per-column marginal counts of (n,K) int codes ``X`` -> (K, Kx) int64, via ONE atomicAdd launch.
    Identical counts to ``cp.bincount((X + col_off*Kx).ravel(), minlength=K*Kx).reshape(K, Kx)``."""
    import cupy as cp

    n, K = int(X.shape[0]), int(X.shape[1])
    counts = cp.zeros(int(K) * int(Kx), dtype=cp.int64)
    h1, _ = _get_batched_joint_hist_kernels()
    threads = 256
    blocks = (n * K + threads - 1) // threads
    h1((blocks,), (threads,), (X, np.int32(int(Kx)), np.int64(n), np.int32(K), counts))
    return counts.reshape(K, int(Kx))


# LAUNCH-FUSION (2026-06-27): per-column joint ENTROPY+NNZ in ONE launch -- the per-column analogue of the
# single-joint ``joint_entropy2``. The conditional CMI path fired _batched_joint_counts2 (atomicAdd launch ->
# global (K, Kx*Kb) f64 counts) THEN _rows_entropy_and_k (a second launch reducing that matrix) for BOTH the
# (x,z) and (x,y*z) joints (4 launches + 2 large f64 intermediates / fit). This kernel runs ONE BLOCK PER
# COLUMN: build that column's (Kx*Kb) histogram in SHARED uint, then fused shared tree-reduce -> (h_k, k_k).
# Counts are identical (same flat key (col,x,b)->x*Kb+b), so plug-in entropy + nnz are BIT-IDENTICAL to the
# count-then-reduce path (verified maxdiff 0). Gated on the per-column hist (M=Kx*Kb uint) fitting shared mem;
# over the limit -> caller falls back to the unfused two-launch path (bit-identical).
_BATCHED_JOINT_ENTROPY_SRC = r"""
extern "C" __global__
void batched_joint_entropy2(const long long* __restrict__ X, const long long* __restrict__ b,
                            const int Kx, const int Kb, const long long n, const int K, const int M,
                            const double inv_n, double* __restrict__ out_h, long long* __restrict__ out_k) {
    extern __shared__ unsigned int hist[];
    int col = blockIdx.x;
    if (col >= K) return;
    int tid = threadIdx.x, nt = blockDim.x;
    for (int i = tid; i < M; i += nt) hist[i] = 0u;
    __syncthreads();
    for (long long i = tid; i < n; i += nt) {
        long long xv = X[i * (long long)K + col];
        atomicAdd(&hist[(unsigned int)(xv * Kb + b[i])], 1u);
    }
    __syncthreads();
    double hloc = 0.0, kloc = 0.0;
    for (int c = tid; c < M; c += nt) { unsigned int v = hist[c]; if (v > 0u) { double p = (double)v * inv_n; hloc += p * log(p); kloc += 1.0; } }
    __shared__ double sh_h[256]; __shared__ double sh_k[256];
    sh_h[tid] = hloc; sh_k[tid] = kloc; __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; } __syncthreads(); }
    if (tid == 0) { out_h[col] = -sh_h[0]; out_k[col] = (long long)(sh_k[0] + 0.5); }
}
"""
_BATCHED_JOINT_ENTROPY_KERNEL = None
_BATCHED_JOINT_ENTROPY_SH_LIMIT = None


def _get_batched_joint_entropy_kernel(cp):
    global _BATCHED_JOINT_ENTROPY_KERNEL, _BATCHED_JOINT_ENTROPY_SH_LIMIT
    if _BATCHED_JOINT_ENTROPY_KERNEL is None:
        _BATCHED_JOINT_ENTROPY_KERNEL = cp.RawKernel(_BATCHED_JOINT_ENTROPY_SRC, "batched_joint_entropy2")
        try:
            _BATCHED_JOINT_ENTROPY_SH_LIMIT = int(cp.cuda.Device().attributes.get("MaxSharedMemoryPerBlock", 48 * 1024))
        except Exception:
            _BATCHED_JOINT_ENTROPY_SH_LIMIT = 48 * 1024
    return _BATCHED_JOINT_ENTROPY_KERNEL


def _batched_joint_entropy_and_k2(X, b, Kx, Kb, inv_n):
    """Fused per-column joint entropy+nnz of (n,K) codes ``X`` with (n,) code ``b`` -> (h (K,) f64, k (K,) int64).
    ONE launch (block/column, shared hist + tree-reduce). BIT-IDENTICAL to
    ``_rows_entropy_and_k(_batched_joint_counts2(X,b,Kx,Kb).astype(f64), inv_n)``. Returns None when the per-
    column hist won't fit shared memory (caller uses the unfused two-launch path) or on any kernel error.

    DISABLED by default (bench-attempt-rejected, see below); set MLFRAME_FE_GPU_FUSE_CMI_ENTROPY=1 to opt in."""
    # bench-attempt-rejected (2026-06-27): count+reduce -> 1 fused launch. nsys: removed ~18-36 cuLaunchKernel/
    # fit (total launch APIs 159->135 across the combined A/B). BIT-IDENTICAL (h/k maxdiff 0; full batched_cmi
    # + return_cards maxdiff 0). REJECTED on GPU KERNEL TIME: the one-block-per-column shared-hist kernel runs at
    # a near-CONSTANT ~36ms (n=100k, K=384, GTX 1050 Ti) regardless of M -- it serializes n shared-mem atomicAdds
    # per column across only K blocks x 256 threads (Pascal shared-atomic + low-occupancy bound). The unfused
    # count+reduce is GLOBAL-atomic over n*K threads (massively parallel) + a cheap reduce: 4.5ms at the joint
    # sizes the conditional CMI path actually hits (Kx*Kz, Kx*Kyz ~ 40-500 since Ky/Kz are small). Measured
    # fused/unfused: M=40 -> 8.05x SLOWER, M=100 -> 8.24x SLOWER, M=400 -> ~par, M=1000 -> 0.50x (only wins for
    # rare huge joints). Net GPU time far worse at production cardinalities despite the launch-count win; nsys
    # made batched_joint_entropy2 62% of GPU time (284ms/8) vs hist2 17% (44ms/8). Kept gated for the rare
    # large-M case + as a documented bench artifact; default OFF.
    import cupy as cp

    if os.environ.get("MLFRAME_FE_GPU_FUSE_CMI_ENTROPY", "0").strip() not in ("1", "true", "True", "yes", "on"):
        return None
    try:
        n, K = int(X.shape[0]), int(X.shape[1])
        M = int(Kx) * int(Kb)
        ker = _get_batched_joint_entropy_kernel(cp)
        # shared = M uint32 hist + 2*256 f64 reduction scratch (the sh_h/sh_k arrays are static, not counted here);
        # gate the dynamic hist against the device per-block limit (leave headroom for the static doubles).
        if M <= 0 or M * 4 + 4096 > int(_BATCHED_JOINT_ENTROPY_SH_LIMIT):
            return None
        Xc = cp.ascontiguousarray(X)
        out_h = cp.empty(K, dtype=cp.float64)
        out_k = cp.empty(K, dtype=cp.int64)
        threads = 256
        ker((K,), (threads,),
            (Xc, b, np.int32(int(Kx)), np.int32(int(Kb)), np.int64(n), np.int32(K), np.int32(M),
             float(inv_n), out_h, out_k),
            shared_mem=M * 4)
        return out_h, out_k
    except Exception:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).debug("fused batched joint entropy failed; count+reduce fallback", exc_info=True)
        return None


# FUSED 1D scalar entropy + occupied-cell (launch-reduction, 2026-06-25). The column-invariant H(z)/H(y,z)/
# H(y) terms in batched_cmi_gpu were cp.bincount (scan+cub) -> boolean-mask getitem -> *inv_n -> log -> mul
# -> sum (~6-7 cuLaunchKernel) plus a separate (c>0).sum() nnz. Computing the histogram with the in-kernel
# atomicAdd joint_counts_gpu (one launch, no bincount) and reducing it with this grid-stride kernel (one
# launch, out is cp.zeros(2)=memset) collapses each scalar term to two launches. Same float64 plug-in
# entropy / occupied-cell -> selection-equivalent.
_ENT_NNZ_1D_SRC = r"""
extern "C" __global__
void ent_nnz_1d(const long long* __restrict__ c, const double inv_n, const long long M,
                double* __restrict__ out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    double hloc = 0.0, kloc = 0.0;
    for (long long i = t; i < M; i += stride) {
        long long ci = c[i];
        if (ci > 0) { double p = (double)ci * inv_n; hloc += p * log(p); kloc += 1.0; }
    }
    __shared__ double sh_h[256];
    __shared__ double sh_k[256];
    int tid = threadIdx.x;
    sh_h[tid] = hloc; sh_k[tid] = kloc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { atomicAdd(&out[0], sh_h[0]); atomicAdd(&out[1], sh_k[0]); }
}
"""
_ENT_NNZ_1D_KERNEL = None


def _ent_nnz_1d(c, inv_n):
    """Plug-in entropy (float) + occupied-cell count (int) of an int64 count vector ``c`` in ONE fused
    RawKernel launch. Falls back to the cupy entropy chain on any kernel error (bit-equivalent)."""
    import cupy as cp

    global _ENT_NNZ_1D_KERNEL
    try:
        if _ENT_NNZ_1D_KERNEL is None:
            _ENT_NNZ_1D_KERNEL = cp.RawKernel(_ENT_NNZ_1D_SRC, "ent_nnz_1d")
        M = int(c.size)
        out = cp.zeros(2, dtype=cp.float64)             # cudaMemsetAsync, not a cuLaunchKernel
        threads = 256
        blocks = min(1024, max(1, (M + threads - 1) // threads))
        _ENT_NNZ_1D_KERNEL((blocks,), (threads,), (c, float(inv_n), np.int64(M), out))
        h_k = cp.asnumpy(out)
        return float(-h_k[0]), int(round(h_k[1]))
    except Exception:  # noqa: BLE001
        cf = c.astype(cp.float64) if c.dtype != cp.float64 else c
        p = cf[cf > 0] * float(inv_n)
        return float(-(p * cp.log(p)).sum()), int((cf > 0).sum())


def joint_counts_gpu(codes, cards):
    """Joint-histogram counts of 1-3 device code arrays via an in-kernel flat key + atomic add (ONE launch,
    no intermediate key array, no cp.bincount cub_any passes). ``codes`` are cupy int64 (n,); ``cards`` the
    matching cardinalities (upper bounds; empty bins cost only memory). Returns a cupy int64 (prod(cards),)
    count vector for the shared entropy/NNZ reduction."""
    import cupy as cp

    n = int(codes[0].size)
    M = 1
    for kc in cards:
        M *= int(kc)
    counts = cp.zeros(int(max(M, 1)), dtype=cp.int64)
    threads = 256
    blocks = (n + threads - 1) // threads
    h1, h2, h3 = _get_joint_hist_kernels()
    if len(codes) == 1:
        h1((blocks,), (threads,), (codes[0], np.int64(n), counts))
    elif len(codes) == 2:
        h2((blocks,), (threads,), (codes[0], codes[1], np.int32(int(cards[1])), np.int64(n), counts))
    else:
        h3((blocks,), (threads,), (codes[0], codes[1], codes[2],
                                   np.int32(int(cards[1])), np.int32(int(cards[2])), np.int64(n), counts))
    return counts


# FUSED joint histogram + plug-in entropy in ONE launch (launch-reduction, 2026-06-25). The per-candidate
# CMI path computed each term as joint_counts_gpu (atomicAdd hist) + _ent_from_counts (entropy reduce) = TWO
# launches. When the joint cell count M = prod(cards) fits in shared memory, ONE block builds the histogram in
# SHARED via atomicAdd, then reduces the plug-in entropy (-sum p log p) + occupied-cell count in the same
# launch. out[0] = sum xlogx (h = -out[0]), out[1] = nnz. Same float64 plug-in entropy / occupied-cell
# definition -> selection-equivalent (exact integer counts; reduction-order ~1e-14). Falls back to the
# two-kernel path (joint_counts_gpu + ent_nnz_1d) when M is over the shared-memory budget.
_JOINT_ENTROPY_SRC = r"""
extern "C" __global__
void joint_entropy1(const long long* __restrict__ a, const long long n, const int M,
                    const double inv_n, double* __restrict__ out) {
    extern __shared__ unsigned int hist[];
    int tid = threadIdx.x, nt = blockDim.x;
    for (int i = tid; i < M; i += nt) hist[i] = 0u;
    __syncthreads();
    for (long long i = tid; i < n; i += nt) atomicAdd(&hist[a[i]], 1u);
    __syncthreads();
    double hloc = 0.0, kloc = 0.0;
    for (int c = tid; c < M; c += nt) { unsigned int v = hist[c]; if (v > 0u) { double p = (double)v * inv_n; hloc += p * log(p); kloc += 1.0; } }
    __shared__ double sh_h[256]; __shared__ double sh_k[256];
    sh_h[tid] = hloc; sh_k[tid] = kloc; __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; } __syncthreads(); }
    if (tid == 0) { out[0] = sh_h[0]; out[1] = sh_k[0]; }
}
extern "C" __global__
void joint_entropy2(const long long* __restrict__ a, const long long* __restrict__ b, const int Kb,
                    const long long n, const int M, const double inv_n, double* __restrict__ out) {
    extern __shared__ unsigned int hist[];
    int tid = threadIdx.x, nt = blockDim.x;
    for (int i = tid; i < M; i += nt) hist[i] = 0u;
    __syncthreads();
    for (long long i = tid; i < n; i += nt) atomicAdd(&hist[a[i] * Kb + b[i]], 1u);
    __syncthreads();
    double hloc = 0.0, kloc = 0.0;
    for (int c = tid; c < M; c += nt) { unsigned int v = hist[c]; if (v > 0u) { double p = (double)v * inv_n; hloc += p * log(p); kloc += 1.0; } }
    __shared__ double sh_h[256]; __shared__ double sh_k[256];
    sh_h[tid] = hloc; sh_k[tid] = kloc; __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; } __syncthreads(); }
    if (tid == 0) { out[0] = sh_h[0]; out[1] = sh_k[0]; }
}
extern "C" __global__
void joint_entropy3(const long long* __restrict__ a, const long long* __restrict__ b,
                    const long long* __restrict__ c, const int Kb, const int Kc, const long long n,
                    const int M, const double inv_n, double* __restrict__ out) {
    extern __shared__ unsigned int hist[];
    int tid = threadIdx.x, nt = blockDim.x;
    for (int i = tid; i < M; i += nt) hist[i] = 0u;
    __syncthreads();
    for (long long i = tid; i < n; i += nt) atomicAdd(&hist[(a[i] * Kb + b[i]) * Kc + c[i]], 1u);
    __syncthreads();
    double hloc = 0.0, kloc = 0.0;
    for (int cc = tid; cc < M; cc += nt) { unsigned int v = hist[cc]; if (v > 0u) { double p = (double)v * inv_n; hloc += p * log(p); kloc += 1.0; } }
    __shared__ double sh_h[256]; __shared__ double sh_k[256];
    sh_h[tid] = hloc; sh_k[tid] = kloc; __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; } __syncthreads(); }
    if (tid == 0) { out[0] = sh_h[0]; out[1] = sh_k[0]; }
}
"""
_JOINT_ENTROPY_KERNELS = None
_JOINT_ENTROPY_SH_LIMIT = None


def _get_joint_entropy_kernels():
    global _JOINT_ENTROPY_KERNELS, _JOINT_ENTROPY_SH_LIMIT
    if _JOINT_ENTROPY_KERNELS is None:
        import cupy as cp
        mod = cp.RawModule(code=_JOINT_ENTROPY_SRC)
        _JOINT_ENTROPY_KERNELS = (mod.get_function("joint_entropy1"),
                                  mod.get_function("joint_entropy2"),
                                  mod.get_function("joint_entropy3"))
        try:
            _JOINT_ENTROPY_SH_LIMIT = int(cp.cuda.Device().attributes.get("MaxSharedMemoryPerBlock", 48 * 1024))
        except Exception:
            _JOINT_ENTROPY_SH_LIMIT = 48 * 1024
    return _JOINT_ENTROPY_KERNELS


def joint_entropy_gpu(codes, cards, inv_n):
    """Plug-in entropy (h) + occupied-cell count (k) of the joint of 1-3 device code arrays in ONE launch
    when the joint cell count fits shared memory (block builds the histogram in shared + reduces entropy);
    else the two-kernel path (joint_counts_gpu + ent_nnz_1d). ``codes`` cupy int64 (n,); ``cards`` the
    matching cardinalities. Returns (float h, int k). Selection-equivalent to _ent_from_counts(joint_counts)."""
    import cupy as cp

    M = 1
    for kc in cards:
        M *= int(kc)
    M = int(max(M, 1))
    kers = _get_joint_entropy_kernels()
    # static shared (sh_h+sh_k = 256*8*2 = 4096 B) + dynamic hist (M*4 B) must clear the per-block limit.
    if M * 4 + 4096 <= _JOINT_ENTROPY_SH_LIMIT:
        try:
            n = int(codes[0].size)
            out = cp.zeros(2, dtype=cp.float64)
            shmem = M * 4
            if len(codes) == 1:
                kers[0]((1,), (256,), (codes[0], np.int64(n), np.int32(M), float(inv_n), out), shared_mem=shmem)
            elif len(codes) == 2:
                kers[1]((1,), (256,), (codes[0], codes[1], np.int32(int(cards[1])), np.int64(n),
                                       np.int32(M), float(inv_n), out), shared_mem=shmem)
            else:
                kers[2]((1,), (256,), (codes[0], codes[1], codes[2], np.int32(int(cards[1])),
                                       np.int32(int(cards[2])), np.int64(n), np.int32(M), float(inv_n), out),
                        shared_mem=shmem)
            hk = cp.asnumpy(out)
            return float(-hk[0]), int(round(hk[1]))
        except Exception:  # noqa: BLE001
            pass
    return _ent_nnz_1d(joint_counts_gpu(codes, cards), inv_n)


# FUSED four-joint conditional-CMI entropies in ONE launch (launch-reduction, 2026-06-26). The conditional
# CMI(x;y|z) needs the plug-in entropy + occupied-cell count of FOUR joints -- (z), (x,z), (y,z), (x,y,z) --
# which _cmi_from_binned_cupy computed as four separate joint_entropy_gpu launches (the #1 cuLaunchKernel
# source on the F2 STRICT redundancy gate after the analytic-null card reuse removed joint_cardinalities).
# This kernel runs all four as FOUR BLOCKS of one grid (block b builds joint b's histogram from the shared
# x/y/z code arrays into its own dynamic-shared hist, then the same tree-reduction over the block's threads
# emits (entropy, nnz) for that joint). All threads in a block share blockIdx -> no branch divergence. The
# dynamic shared is sized to the largest joint (x,y,z); engages only when that fits the per-block limit (the
# SAME condition under which joint_entropy_gpu took its fast single-launch path for x,y,z), else the caller
# uses the per-joint path. Same f64 plug-in entropy, same occupied-cell (c>0) definition, same cell-index
# reduction order as joint_entropy1/2/3 -> the assembled CMI is bit-identical.
_CMI_JOINT_ENTROPIES_SRC = r"""
extern "C" __global__
void cmi_joint_entropies(const long long* __restrict__ x, const long long* __restrict__ y,
                         const long long* __restrict__ z, const int Kx, const int Ky, const int Kz,
                         const long long n, const double inv_n, double* __restrict__ out) {
    extern __shared__ unsigned int hist[];
    int blk = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    int M;
    if (blk == 0) M = Kz;
    else if (blk == 1) M = Kx * Kz;
    else if (blk == 2) M = Ky * Kz;
    else M = Kx * Ky * Kz;
    for (int i = tid; i < M; i += nt) hist[i] = 0u;
    __syncthreads();
    for (long long i = tid; i < n; i += nt) {
        long long zi = z[i];
        int idx;
        if (blk == 0) idx = (int)zi;
        else if (blk == 1) idx = (int)(x[i] * Kz + zi);
        else if (blk == 2) idx = (int)(y[i] * Kz + zi);
        else idx = (int)((x[i] * Ky + y[i]) * Kz + zi);
        atomicAdd(&hist[idx], 1u);
    }
    __syncthreads();
    double hloc = 0.0, kloc = 0.0;
    for (int c = tid; c < M; c += nt) { unsigned int v = hist[c]; if (v > 0u) { double p = (double)v * inv_n; hloc += p * log(p); kloc += 1.0; } }
    __shared__ double sh_h[256]; __shared__ double sh_k[256];
    sh_h[tid] = hloc; sh_k[tid] = kloc; __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; } __syncthreads(); }
    if (tid == 0) { out[blk * 2] = -sh_h[0]; out[blk * 2 + 1] = sh_k[0]; }
}
"""
_CMI_JOINT_ENTROPIES_KERNEL = None


def cmi_joint_entropies_gpu(dx, dy, dz, Kx, ky, kz, inv_n):
    """The four conditional-CMI joint (plug-in entropy, occupied-cell count) terms -- (z), (x,z), (y,z),
    (x,y,z) -- in ONE launch. Returns ((h_z,k_z),(h_xz,k_xz),(h_yz,k_yz),(h_xyz,k_xyz)), or None when the
    largest (x,y,z) joint won't fit the per-block shared limit (caller falls back to four joint_entropy_gpu
    launches). Bit-identical to four separate joint_entropy_gpu calls."""
    import cupy as cp

    global _CMI_JOINT_ENTROPIES_KERNEL
    _get_joint_entropy_kernels()  # ensures _JOINT_ENTROPY_SH_LIMIT is populated
    Mmax = int(Kx) * int(ky) * int(kz)
    if Mmax <= 0 or Mmax * 4 + 4096 > int(_JOINT_ENTROPY_SH_LIMIT):
        return None
    try:
        if _CMI_JOINT_ENTROPIES_KERNEL is None:
            _CMI_JOINT_ENTROPIES_KERNEL = cp.RawKernel(_CMI_JOINT_ENTROPIES_SRC, "cmi_joint_entropies")
        n = int(dz.size)
        out = cp.zeros(8, dtype=cp.float64)
        _CMI_JOINT_ENTROPIES_KERNEL((4,), (256,), (dx, dy, dz, np.int32(int(Kx)), np.int32(int(ky)),
                                    np.int32(int(kz)), np.int64(n), float(inv_n), out), shared_mem=Mmax * 4)
        hk = cp.asnumpy(out)
        return ((float(hk[0]), int(round(hk[1]))), (float(hk[2]), int(round(hk[3]))),
                (float(hk[4]), int(round(hk[5]))), (float(hk[6]), int(round(hk[7]))))
    except Exception:  # noqa: BLE001
        return None


# bench-attempt-rejected (2026-06-26): a two-block fused (x,z)+(x,y,z) kernel for the FIXED-yz greedy CMI path
# (cmi_from_binned_fixed_yz_cupy, where H(z)/H(y,z) are precomputed so only xz/xyz remain) was BIT-IDENTICAL
# to the two separate joint_entropy_gpu calls on random codes (10/10 A/B) yet FLIPPED the full-MRMR selection
# on test_gpu_cpu_mi_selection_equivalence[reg_two_pairs] and [adv_wide_p60] -- a real divergence the random
# A/B did not surface (interaction with the precomputed-yz terms at a card combo). The four-block conditional
# fusion (cmi_joint_entropies_gpu) is SAFE there (all four joints come from the one kernel -> self-consistent;
# full suite green); the fixed-yz two-block split mixing fused xz/xyz with precomputed h_z/h_yz is not. Kept
# the fixed-yz path on the two per-joint joint_entropy_gpu launches.


# FUSED three-joint marginal-MI entropies in ONE launch -- the marginal MI(x;y) path (the seed anchor and the
# redundancy gate's per-raw marginal anchor) needs H(x), H(y), H(x,y), three separate joint_entropy_gpu
# launches per call. Three blocks of one grid (block 0: x, M=Kx; block 1: y, M=Ky; block 2: xy, M=Kx*Ky),
# each the same per-block histogram + tree reduction. All three terms come from THIS kernel (the marginal MI
# combines only its own h_x/h_y/h_xy, no precomputed cross-path term), so it is self-consistent like the
# four-block conditional fusion -- not the rejected two-block fixed-yz split. Engages when the (x,y) joint
# fits the per-block shared limit (it is tiny: Kx*Ky). Same f64 plug-in entropy, occupied-cell definition,
# and cell-index reduction order -> bit-identical.
_MARGINAL_MI_ENTROPIES_SRC = r"""
extern "C" __global__
void marginal_mi_entropies(const long long* __restrict__ x, const long long* __restrict__ y,
                           const int Kx, const int Ky, const long long n, const double inv_n,
                           double* __restrict__ out) {
    extern __shared__ unsigned int hist[];
    int blk = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    int M = (blk == 0) ? Kx : (blk == 1) ? Ky : (Kx * Ky);
    for (int i = tid; i < M; i += nt) hist[i] = 0u;
    __syncthreads();
    for (long long i = tid; i < n; i += nt) {
        int idx = (blk == 0) ? (int)x[i] : (blk == 1) ? (int)y[i] : (int)(x[i] * Ky + y[i]);
        atomicAdd(&hist[idx], 1u);
    }
    __syncthreads();
    double hloc = 0.0, kloc = 0.0;
    for (int c = tid; c < M; c += nt) { unsigned int v = hist[c]; if (v > 0u) { double p = (double)v * inv_n; hloc += p * log(p); kloc += 1.0; } }
    __shared__ double sh_h[256]; __shared__ double sh_k[256];
    sh_h[tid] = hloc; sh_k[tid] = kloc; __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; } __syncthreads(); }
    if (tid == 0) { out[blk * 2] = -sh_h[0]; out[blk * 2 + 1] = sh_k[0]; }
}
"""
_MARGINAL_MI_ENTROPIES_KERNEL = None


def marginal_mi_entropies_gpu(dx, dy, Kx, ky, inv_n):
    """The three marginal-MI joint terms -- (x), (y), (x,y) -- in ONE launch. Returns
    ((h_x,k_x),(h_y,k_y),(h_xy,k_xy)), or None when the (x,y) joint won't fit the per-block shared limit
    (caller falls back to three joint_entropy_gpu launches). Bit-identical to the three separate calls."""
    import cupy as cp

    global _MARGINAL_MI_ENTROPIES_KERNEL
    _get_joint_entropy_kernels()  # ensures _JOINT_ENTROPY_SH_LIMIT is populated
    Mmax = int(Kx) * int(ky)
    if Mmax <= 0 or Mmax * 4 + 4096 > int(_JOINT_ENTROPY_SH_LIMIT):
        return None
    try:
        if _MARGINAL_MI_ENTROPIES_KERNEL is None:
            _MARGINAL_MI_ENTROPIES_KERNEL = cp.RawKernel(_MARGINAL_MI_ENTROPIES_SRC, "marginal_mi_entropies")
        n = int(dx.size)
        out = cp.zeros(6, dtype=cp.float64)
        _MARGINAL_MI_ENTROPIES_KERNEL((3,), (256,), (dx, dy, np.int32(int(Kx)), np.int32(int(ky)),
                                      np.int64(n), float(inv_n), out), shared_mem=Mmax * 4)
        hk = cp.asnumpy(out)
        return ((float(hk[0]), int(round(hk[1]))), (float(hk[2]), int(round(hk[3]))),
                (float(hk[4]), int(round(hk[5]))))
    except Exception:  # noqa: BLE001
        return None


# FUSED final CMI/MI assembly (launch-reduction, 2026-06-26). batched_cmi_gpu assembled the per-column result
# with a cupy chain over (K,) arrays: cmi = h_xz + h_yz - h_z - h_xyz; bias = (k_xyz + k_z - k_xz - k_yz)/2n;
# max(cmi - bias, 0) -- ~7 launches. Both the conditional and marginal forms reduce to
#   out[i] = max( he1[i] - he2[i] + hc - (kc1[i] - kc2[i] + kc) * inv2n , 0 )
# so one (K,) kernel emits the result. Same f64 ops -> bit-identical.
_CMI_ASSEMBLE_SRC = r"""
extern "C" __global__
void cmi_assemble(const double* __restrict__ he1, const double* __restrict__ he2, const double hc,
                  const long long* __restrict__ kc1, const long long* __restrict__ kc2, const double kc,
                  const double inv2n, const int K, double* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;
    double v = (he1[i] - he2[i] + hc) - ((double)(kc1[i] - kc2[i]) + kc) * inv2n;
    out[i] = v > 0.0 ? v : 0.0;
}
"""
_CMI_ASSEMBLE_KERNEL = None


def _cmi_assemble(he1, he2, hc, kc1, kc2, kc, inv2n):
    """out = max((he1 - he2 + hc) - ((kc1 - kc2) + kc)*inv2n, 0) per column, in ONE launch. he1/he2 are (K,)
    f64 entropy terms, kc1/kc2 (K,) int64 occupied-cell counts, hc/kc/inv2n scalars. Bit-identical to the
    cupy assembly. Falls back to the cupy chain on any kernel error."""
    import cupy as cp

    global _CMI_ASSEMBLE_KERNEL
    try:
        if _CMI_ASSEMBLE_KERNEL is None:
            _CMI_ASSEMBLE_KERNEL = cp.RawKernel(_CMI_ASSEMBLE_SRC, "cmi_assemble")
        K = int(he1.shape[0])
        out = cp.empty(K, dtype=cp.float64)
        threads = 128
        _CMI_ASSEMBLE_KERNEL(((K + threads - 1) // threads,), (threads,),
                             (cp.ascontiguousarray(he1.astype(cp.float64, copy=False)),
                              cp.ascontiguousarray(he2.astype(cp.float64, copy=False)), float(hc),
                              cp.ascontiguousarray(kc1.astype(cp.int64, copy=False)),
                              cp.ascontiguousarray(kc2.astype(cp.int64, copy=False)), float(kc),
                              float(inv2n), np.int32(K), out))
        return out
    except Exception:  # noqa: BLE001
        return cp.maximum((he1 - he2 + hc) - ((kc1 - kc2) + kc) * inv2n, 0.0)


_JOINT_NNZ_KERNELS = None


def _get_joint_nnz_kernels():
    global _JOINT_NNZ_KERNELS
    if _JOINT_NNZ_KERNELS is None:
        import cupy as cp
        mod = cp.RawModule(code=_JOINT_HIST_SRC)
        _JOINT_NNZ_KERNELS = (mod.get_function("joint_hist_nnz1"),
                              mod.get_function("joint_hist_nnz2"),
                              mod.get_function("joint_hist_nnz3"))
    return _JOINT_NNZ_KERNELS


def joint_nnz_gpu(codes, cards):
    """Occupied-cell COUNT (cardinality) of the joint of 1-3 device code arrays in ONE launch -- the
    histogram atomicAdd and the nonzero count fuse via the atomicAdd-returns-old 0->1 trick, so no separate
    cp.bincount / _nnz reduction. ``codes`` cupy int64 (n,); ``cards`` the matching cardinalities. Returns a
    Python int. The count is exact (integer) -> identical to ``_nnz_from_counts(joint_counts_gpu(...))``."""
    import cupy as cp

    n = int(codes[0].size)
    M = 1
    for kc in cards:
        M *= int(kc)
    counts = cp.zeros(int(max(M, 1)), dtype=cp.int64)        # cudaMemsetAsync, not a cuLaunchKernel
    nnz = cp.zeros(1, dtype=cp.uint64)                       # cudaMemsetAsync, not a cuLaunchKernel
    threads = 256
    blocks = (n + threads - 1) // threads
    h1, h2, h3 = _get_joint_nnz_kernels()
    if len(codes) == 1:
        h1((blocks,), (threads,), (codes[0], np.int64(n), counts, nnz))
    elif len(codes) == 2:
        h2((blocks,), (threads,), (codes[0], codes[1], np.int32(int(cards[1])), np.int64(n), counts, nnz))
    else:
        h3((blocks,), (threads,), (codes[0], codes[1], codes[2],
                                   np.int32(int(cards[1])), np.int32(int(cards[2])), np.int64(n), counts, nnz))
    return int(nnz.get()[0])


def batched_quantile_bin_gpu(x_cols, nbins: int):
    """Born-on-device equi-frequency binning of an (n,K) float matrix -> (n,K) int codes, RESIDENT on GPU.

    Device twin of ``_mi_greedy_cmi_fe._quantile_bin`` applied per column. ONE batched ``cp.percentile``
    (axis=0) computes all K columns' edges in a single device sort -- replacing K host ``np.quantile``
    (introselect partition) calls -- then per-column ``cp.searchsorted`` on the (deduped) inner edges. The
    per-column ``cp.unique`` is over only (nbins+1) edge values (negligible), so the n-sized sort stays
    batched. Returns a cupy int64 (n,K) array (kept on device to feed ``batched_cmi_gpu`` without a code
    H2D). Selection-equivalent to the host binning: same equi-frequency partition (value-edge, rank-based).
    Assumes all-finite input (the production nan-scrubbed case); the caller falls back to the host path
    otherwise.
    """
    import cupy as cp

    Xd = x_cols if isinstance(x_cols, cp.ndarray) else cp.asarray(np.ascontiguousarray(np.asarray(x_cols, dtype=np.float64)))
    if Xd.ndim == 1:
        Xd = Xd[:, None]
    Xd = Xd.astype(cp.float64, copy=False)
    n, K = int(Xd.shape[0]), int(Xd.shape[1])
    qs = cp.linspace(0.0, 100.0, nbins + 1)
    edges_all = cp.percentile(Xd, qs, axis=0)        # (nbins+1, K) -- one batched device sort
    codes = cp.zeros((n, K), dtype=cp.int64)
    for k in range(K):
        edges = cp.unique(edges_all[:, k])           # dedupe equi-freq edges (mirror np.unique on host)
        if int(edges.size) <= 2:
            if int(edges.size) == 2:
                codes[:, k] = (Xd[:, k] >= edges[1]).astype(cp.int64)
            continue
        codes[:, k] = cp.searchsorted(edges[1:-1], Xd[:, k], side="right").astype(cp.int64)
    return codes


def batched_cmi_gpu(x_cols, y: np.ndarray, z=None, return_cards: bool = False, codes_trusted: bool = False):
    """Miller-Madow plug-in CMI(x_k; y | z) in nats for EVERY column of ``x_cols``, in ONE device workload.

    ``x_cols`` (n,K) int codes -- a host ndarray OR an already-resident cupy array (born-on-device codes
    from ``batched_quantile_bin_gpu``, no code H2D); ``y`` (n,) int codes; ``z`` (n,) int codes or None
    (marginal MI). Returns a host (K,) float64 array. Matches ``_mi_greedy_cmi_fe._cmi_from_binned`` per
    column (selection-equivalent).

    ``return_cards`` (conditional path only): also return the occupied-cell cardinalities the analytic
    CMI-null df needs -- ``(cmi[K], k_z, k_xz[K], k_yz, k_xyz[K])`` -- computed in the SAME workload (they
    are already produced internally by ``_rows_entropy_and_k`` + the shared y/z terms). Lets the gate score
    the analytic floor/df of ALL round candidates from ONE call instead of a per-candidate
    ``joint_cardinalities_cupy``. The cell counts equal the per-candidate path's (same occupied-cell
    definition) -> df bit-identical.
    """
    import cupy as cp

    if isinstance(x_cols, cp.ndarray):
        X = x_cols.astype(cp.int64, copy=False)
    else:
        X = cp.asarray(np.ascontiguousarray(x_cols).astype(np.int64))
    if X.ndim == 1:
        X = X[:, None]
    X = cp.ascontiguousarray(X)   # batched joint-hist kernels read X[row*K+col] (C-order (n,K)); no-op if already
    n, K = int(X.shape[0]), int(X.shape[1])
    dy = cp.asarray(np.ascontiguousarray(y).astype(np.int64).ravel())
    nf = float(max(1, n))
    inv_n = 1.0 / nf
    Kx = int(X.max()) + 1 if X.size else 1
    _assert_codes_in_range(X, Kx, "batched_cmi_gpu X codes", codes_trusted)

    if z is None or (hasattr(z, "size") and np.asarray(z).size == 0):
        # Marginal MI(x_k; y), MM-corrected:  H(x)+H(y)-H(x,y) - (k_x+k_y-k_xy-1)/2n
        Ky = int(dy.max()) + 1 if dy.size else 1
        _assert_codes_in_range(dy, Ky, "batched_cmi_gpu y codes", codes_trusted)
        cnt_xy = _batched_joint_counts2(X, dy, Kx, Ky).astype(cp.float64)   # (K, Kx*Ky), in-kernel flat key
        h_xy, k_xy = _rows_entropy_and_k(cnt_xy, inv_n)
        cnt_x = _batched_marginal_counts(X, Kx).astype(cp.float64)          # (K, Kx)
        h_x, k_x = _rows_entropy_and_k(cnt_x, inv_n)
        h_y, k_y = joint_entropy_gpu([dy], [Ky], inv_n)                     # H(y): fused hist+entropy in ONE launch (same 2-launch fallback if it won't fit shared)
        # FUSED assembly: max(h_x - h_xy + h_y - (k_x - k_xy + (k_y-1))/2n, 0) in one (K,) launch.
        mi = _cmi_assemble(h_x, h_xy, float(h_y), k_x, k_xy, float(k_y - 1), 1.0 / (2.0 * nf))
        return cp.asnumpy(mi)

    dz = cp.asarray(np.ascontiguousarray(z).astype(np.int64).ravel())
    Kz = int(dz.max()) + 1 if dz.size else 1
    # Conditional path: y and z both index flat histograms (yz = dy*Kz+dz, then (x*Kyz+yz)); guard both.
    Ky_cond = int(dy.max()) + 1 if dy.size else 1
    _assert_codes_in_range(dy, Ky_cond, "batched_cmi_gpu y codes", codes_trusted)
    _assert_codes_in_range(dz, Kz, "batched_cmi_gpu z codes", codes_trusted)
    # shared y/z terms (column-invariant): fused hist+entropy in ONE launch each (same 2-launch fallback when
    # the 1D joint won't fit shared) -- bit-identical to _ent_nnz_1d(joint_counts_gpu(...)).
    h_z, k_z = joint_entropy_gpu([dz], [Kz], inv_n)
    yz = dy * Kz + dz                      # dense (y,z) code (also feeds cnt_xyz below)
    Kyz = int(yz.max()) + 1
    h_yz, k_yz = joint_entropy_gpu([yz], [Kyz], inv_n)

    # H(x_k, z) for all k. LAUNCH-FUSION: ONE block/column kernel (shared hist + tree-reduce) collapses the
    # atomicAdd-count + reduce into a single launch (drops the (K, Kx*Kz) f64 intermediate); falls back to the
    # two-launch count+reduce (in-kernel flat key k*(Kx*Kz)+x*Kz+z) when the per-column hist won't fit shared.
    _fused = _batched_joint_entropy_and_k2(X, dz, Kx, Kz, inv_n)
    if _fused is not None:
        h_xz, k_xz = _fused
    else:
        cnt_xz = _batched_joint_counts2(X, dz, Kx, Kz).astype(cp.float64)   # (K, Kx*Kz)
        h_xz, k_xz = _rows_entropy_and_k(cnt_xz, inv_n)
    # H(x_k, y, z) for all k: in-kernel flat key k*(Kx*Kyz) + x*Kyz + yz (same fuse; large Kyz -> fallback)
    _fused = _batched_joint_entropy_and_k2(X, yz, Kx, Kyz, inv_n)
    if _fused is not None:
        h_xyz, k_xyz = _fused
    else:
        cnt_xyz = _batched_joint_counts2(X, yz, Kx, Kyz).astype(cp.float64) # (K, Kx*Kyz)
        h_xyz, k_xyz = _rows_entropy_and_k(cnt_xyz, inv_n)

    # FUSED assembly: max(h_xz - h_xyz + (h_yz - h_z) - (k_xyz - k_xz + (k_z - k_yz))/2n, 0) in one (K,) launch.
    cmi_host = cp.asnumpy(_cmi_assemble(h_xz, h_xyz, float(h_yz - h_z),
                                        k_xyz, k_xz, float(k_z - k_yz), 1.0 / (2.0 * nf)))
    if return_cards:
        return (cmi_host, int(k_z), cp.asnumpy(k_xz).astype(np.int64),
                int(k_yz), cp.asnumpy(k_xyz).astype(np.int64))
    return cmi_host
