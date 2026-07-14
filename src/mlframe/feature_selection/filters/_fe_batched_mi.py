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

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


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
                // bench-attempt-rejected (2026-07-04): ry (the y-marginal) is xx-invariant so this is an
                // O(Kx^2*Ky) redundant single-thread recompute, but it runs in the tid==0 tail after the
                // n-row atomicAdd histogram and is immeasurable vs it (doubling Kx/Ky moves the wall <4%,
                // that being histogram size not the reduce). See _benchmarks/bench_mi_from_codes_ymarginal_hoist.py.
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
    const double* Xc = X + (long long)c * n;          // COLUMN-MAJOR (K,n): coalesced across threads (nvprof
    for (long long i = tid; i < n; i += nt) {          // 2026-07-02: row-major X[i*K+c] was gld_efficiency 28.9%)
        double v = Xc[i];
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
// SPLIT-N variant (SM-occupancy fix, 2026-07-02): one block per column starves the 6-SM card when K is
// narrow (nsys: 60/66 mi_from_values launches were K<=32 blocks, the giant 1.56s instances among them).
// Phase A (grid (K,S)): each of S segment blocks bins+histograms its n-slice into a shared (nbins*Ky) tile,
// then merges the tile into a global per-column (nbins*Ky) int counts buffer with one atomicAdd pass -> K*S
// blocks fill the SMs. Phase B (grid (K,)): plug-in MI per column from the merged counts. Same bin codes
// (same f64 edges, upper_bound) + same plug-in formula -> selection-equivalent to mi_from_values.
extern "C" __global__
void mi_hist_split(const double* __restrict__ X, const double* __restrict__ edges,
                   const long long* __restrict__ y, const long long n, const int K,
                   const int nbins, const int Ky, const int S, int* __restrict__ gcounts) {
    extern __shared__ int sh[];
    int c = blockIdx.x, seg = blockIdx.y;
    if (c >= K || seg >= S) return;
    int ne = nbins - 1, M = nbins * Ky, tid = threadIdx.x, nt = blockDim.x;
    for (int s = tid; s < M; s += nt) sh[s] = 0;
    __syncthreads();
    long long seg_len = (n + S - 1) / S, lo = (long long)seg * seg_len, hi = lo + seg_len;
    if (hi > n) hi = n;
    const double* Xc = X + (long long)c * n;          // COLUMN-MAJOR (K,n): coalesced load (see mi_from_values)
    for (long long i = lo + tid; i < hi; i += nt) {
        double v = Xc[i];
        int a = 0, b = ne;
        while (a < b) { int mid = (a + b) >> 1; if (edges[(long long)mid * K + c] <= v) a = mid + 1; else b = mid; }
        atomicAdd(&sh[a * Ky + y[i]], 1);
    }
    __syncthreads();
    int* gc = gcounts + (long long)c * M;
    for (int s = tid; s < M; s += nt) if (sh[s]) atomicAdd(&gc[s], sh[s]);
}
extern "C" __global__
void mi_from_counts_col(const int* __restrict__ gcounts, const int K, const int nbins, const int Ky,
                        const double inv_n, double* __restrict__ mi_out) {
    int c = blockIdx.x;
    if (c >= K) return;
    const int* gc = gcounts + (long long)c * (nbins * Ky);
    if (threadIdx.x != 0) return;                 // M = nbins*Ky ~ 100: single-thread reduce is trivial
    double mi = 0.0;
    for (int xx = 0; xx < nbins; ++xx) {
        long long rx = 0;
        for (int yy = 0; yy < Ky; ++yy) rx += gc[xx * Ky + yy];
        if (rx == 0) continue;
        double px = (double)rx * inv_n;
        for (int yy = 0; yy < Ky; ++yy) {
            long long nxy = gc[xx * Ky + yy];
            if (nxy == 0) continue;
            long long ry = 0;
            for (int xx2 = 0; xx2 < nbins; ++xx2) ry += gc[xx2 * Ky + yy];
            mi += (double)nxy * inv_n * log(((double)nxy * inv_n) / (px * ((double)ry * inv_n)));
        }
    }
    mi_out[c] = mi > 0.0 ? mi : 0.0;
}
"""
_MI_FROM_VALUES_KERNEL = None
_MI_SPLIT_KERNELS = None

# f32-X twin of the values->bin->hist->MI module (dtype-churn kill, 2026-07-03, nsys-driven). The opt-in
# f32 binning path (keep_dtype / relax_binning) holds the (n,K) candidate matrix in float32 to feed the
# f32 radix-select edges, but ``binned_mi_from_values_gpu`` re-UPCAST that whole matrix to f64
# (X.astype(f64)) before this kernel AND transposed it in f64 (transpose_f64) -- pure churn: the f32->f64
# downcast+upcast round-trip (nsys F2 1M STRICT: 562 cupy_copy__float32_float64 + 199 transpose_f64). The
# ONLY use of X inside the kernel is ``double v = Xc[i]`` (a widening float->double promotion) fed to the
# edge comparison; the f64 path stored exactly ``(double)f32`` there, so reading the value as f32 and
# promoting per-element yields the IDENTICAL f64 -> the bin codes and plug-in MI are BIT-IDENTICAL. The
# EDGES stay ``const double*`` (the interp emits f64; kept f64 so the comparison is byte-for-byte the f64
# path's). Only X's storage/transpose dtype changes: the (n,K) upcast is gone and the transpose runs in
# f32 (transpose_f32, half the bytes). Everything else (histogram int32, MI f64 math) is unchanged.
_MI_FROM_VALUES_F32_SRC = (
    _MI_FROM_VALUES_SRC.replace(
        "const double* __restrict__ X, const double* __restrict__ edges,", "const float* __restrict__ X, const double* __restrict__ edges,"
    )
    .replace("const double* Xc = X", "const float* Xc = X")
    .replace("void mi_from_values(", "void mi_from_values_f32(")
    .replace("void mi_hist_split(", "void mi_hist_split_f32(")
    .replace("void mi_from_counts_col(", "void mi_from_counts_col_f32(")
)
_MI_FROM_VALUES_F32_KERNEL = None
_MI_SPLIT_F32_KERNELS = None

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
    """Lazy-compile + cache the ``dedup_njit_edges`` RawKernel (module-level singleton, pickle-safe)."""
    global _DEDUP_EDGES_KERNEL
    if _DEDUP_EDGES_KERNEL is None:
        import cupy as cp
        _DEDUP_EDGES_KERNEL = cp.RawKernel(_DEDUP_EDGES_SRC, "dedup_njit_edges")
    return _DEDUP_EDGES_KERNEL


# Length-aware twin of mi_mm_from_values: each column binary-searches only its VALID prefix ne_k[c] of the
# (dedup'd) interior edges -- everything else is byte-for-byte the mi_mm_from_values kernel. When ne_k[c] ==
# nbins-1 (continuous columns, no dup/boundary edges) it is bit-identical to mi_mm_from_values.
_MI_MM_FROM_VALUES_NEK_SRC = (
    _MI_MM_FROM_VALUES_SRC.replace("void mi_mm_from_values(", "void mi_mm_from_values_nek(")
    .replace(
        "const double h_y, const int k_y,\n                       double* __restrict__ mi_out) {",
        "const double h_y, const int k_y,\n                       const int* __restrict__ ne_k, double* __restrict__ mi_out) {")
    .replace("    int ne = nbins - 1, M = nbins * Ky;", "    int ne = ne_k[c], M = nbins * Ky;")
)
_MI_MM_FROM_VALUES_NEK_KERNEL = None

# f32-X twin of the length-aware MM kernel (dtype-churn kill, 2026-07-03). Same rationale as the f32
# mi_from_values twin: the only use of X is ``double v = X[i*K+c]`` (widening float->double promotion), so
# reading X as f32 and promoting per-element yields the exact f64 the f64 path stored via X.astype(f64) ->
# BIT-IDENTICAL codes/MM-MI. Edges stay f64. Kills the (n,K) f32->f64 upcast on the pair-combo MM path.
_MI_MM_FROM_VALUES_NEK_F32_SRC = _MI_MM_FROM_VALUES_NEK_SRC.replace(
    "void mi_mm_from_values_nek(const double* __restrict__ X, const double* __restrict__ edges,",
    "void mi_mm_from_values_nek_f32(const float* __restrict__ X, const double* __restrict__ edges,",
)
_MI_MM_FROM_VALUES_NEK_F32_KERNEL = None


def _get_mi_mm_from_values_nek_kernel():
    """Lazy-compile + cache the f64 length-aware Miller-Madow ``mi_mm_from_values_nek`` RawKernel."""
    global _MI_MM_FROM_VALUES_NEK_KERNEL
    if _MI_MM_FROM_VALUES_NEK_KERNEL is None:
        import cupy as cp
        _MI_MM_FROM_VALUES_NEK_KERNEL = cp.RawKernel(_MI_MM_FROM_VALUES_NEK_SRC, "mi_mm_from_values_nek")
    return _MI_MM_FROM_VALUES_NEK_KERNEL


def _get_mi_mm_from_values_nek_f32_kernel():
    """Lazy-compile + cache the f32-X twin of ``mi_mm_from_values_nek`` (bit-identical MI, halved X bytes)."""
    global _MI_MM_FROM_VALUES_NEK_F32_KERNEL
    if _MI_MM_FROM_VALUES_NEK_F32_KERNEL is None:
        import cupy as cp
        _MI_MM_FROM_VALUES_NEK_F32_KERNEL = cp.RawKernel(_MI_MM_FROM_VALUES_NEK_F32_SRC, "mi_mm_from_values_nek_f32")
    return _MI_MM_FROM_VALUES_NEK_F32_KERNEL


def _get_mi_mm_from_values_kernel():
    """Lazy-compile + cache the non-length-aware ``mi_mm_from_values`` RawKernel (full nbins-1 edge set)."""
    global _MI_MM_FROM_VALUES_KERNEL
    if _MI_MM_FROM_VALUES_KERNEL is None:
        import cupy as cp
        _MI_MM_FROM_VALUES_KERNEL = cp.RawKernel(_MI_MM_FROM_VALUES_SRC, "mi_mm_from_values")
    return _MI_MM_FROM_VALUES_KERNEL


def binned_mm_mi_from_values_gpu(x_vals: Any, interior_edges: Any, y_codes: Any, nbins: int, ky: int, h_y: float, k_y: int, codes_trusted: bool = False,
                                 return_device: bool = False) -> Any:
    """Miller-Madow MARGINAL MI(col_k; y) for an (n,K) float matrix binned by per-column ``interior_edges``
    ((nbins-1, K) cupy), in ONE fused RawKernel (bin + joint hist + MM-MI). ``ky`` is the y-cardinality
    (histogram width; y codes in [0, ky)); ``h_y`` / ``k_y`` are the shared target plug-in entropy +
    OCCUPIED class count used in the bias. Returns a host (K,) float64 array (clamped >=0), matching
    _gpu_marginal_mi / _marginal_mi_njit. Returns None if the (nbins*ky) shared tile won't fit."""
    import cupy as cp

    # DTYPE-CHURN KILL (2026-07-03): keep an f32 candidate matrix at f32 (feed the f32-X MM kernel twin,
    # edges f64) instead of upcasting the whole (n,K) matrix to f64. BIT-IDENTICAL: the kernel reads X only
    # via ``double v = X[i*K+c]`` (widening float->double promotion == the exact f64 the f64 path stored),
    # and cmin/cmax are order statistics of the same f32 values (max/min then exact f64 upcast, unchanged).
    _is_f32 = getattr(x_vals, "dtype", None) == cp.float32
    Xd = x_vals if _is_f32 else x_vals.astype(cp.float64, copy=False)
    if Xd.ndim == 1:
        Xd = Xd[:, None]
    n, K = int(Xd.shape[0]), int(Xd.shape[1])
    Ky = int(ky)
    if int(nbins) * Ky * 4 > _MI_FROM_CODES_MAX_SHARED:
        return None
    E = cp.ascontiguousarray(interior_edges.astype(cp.float64, copy=False))
    # x_vals / interior_edges are per-candidate (transient) -> NOT cached. A host y_codes is a fit-constant
    # re-uploaded on every candidate column's MM-MI -> resident operand cache (uploaded once per fit).
    if isinstance(y_codes, cp.ndarray):
        yv = y_codes.astype(cp.int64, copy=False).ravel()
    else:
        from ._fe_resident_operands import resident_operand
        yv = resident_operand(np.asarray(y_codes).ravel(), "binned_mm_ycodes", dtype=np.int64)
    _assert_codes_in_range(yv, Ky, "binned_mm_mi_from_values_gpu y codes", codes_trusted)
    mi_out = cp.empty(K, dtype=cp.float64)
    Xc = cp.ascontiguousarray(Xd)
    # NJIT-PARITY: dedup the per-column interior edges to njit's distinct-threshold set (drops duplicate +
    # boundary edges that over-bin low-cardinality columns), then bin on only each column's valid prefix.
    ne = int(E.shape[0])
    cmin = cp.ascontiguousarray(Xc.min(axis=0).astype(cp.float64, copy=False))
    cmax = cp.ascontiguousarray(Xc.max(axis=0).astype(cp.float64, copy=False))
    # cp.empty (not cp.zeros): the dedup kernel writes out[0:ne_k[c], c] for every column and the nek MI
    # kernel binary-searches ONLY each column's valid prefix (ne = ne_k[c]), so the uninitialised tail rows
    # of Ec are never read -> dropping the per-call (ne,K) zero-fill is bit-identical (was 17k zero-fills/fit).
    # (ne+1, K) not (ne, K): dedup_njit_edges appends the cmax row at index w==ne (out[ne*K+c]) BEFORE the
    # w-=1 decrement (lines 239-241) for all-distinct interior edges, so the buffer must hold nbins rows.
    # That trailing row is written transiently but never read (after w-=1 always ne_k[c] <= ne, and the nek
    # MI kernel reads only rows 0..ne_k[c]-1), so ne_k and all downstream MI are bit-identical; the extra row
    # only stops the stray write from corrupting the adjacent pool block (was cudaErrorIllegalAddress).
    Ec = cp.empty((ne + 1, K), dtype=cp.float64)
    ne_k = cp.empty(K, dtype=cp.int32)
    threads = 256
    _get_dedup_edges_kernel()(((K + threads - 1) // threads,), (threads,), (E, cmin, cmax, np.int32(ne), np.int32(K), Ec, ne_k))
    _mm_nek = _get_mi_mm_from_values_nek_f32_kernel() if _is_f32 else _get_mi_mm_from_values_nek_kernel()
    _mm_nek((K,), (256,),
        (Xc, Ec, yv, np.int64(n), np.int32(K), np.int32(int(nbins)),
         np.int32(Ky), np.float64(1.0 / float(max(1, n))), np.float64(float(h_y)), np.int32(int(k_y)),
         ne_k, mi_out),
        shared_mem=int(nbins) * Ky * 4)
    # return_device: hand the resident (K,) mi_out back so the caller can fuse a device-side mask + ONE D2H
    # (the resident pool path masks std<=1e-9 combos and gets both in a single sync) instead of two .get()s.
    if return_device:
        return mi_out
    return cp.asnumpy(mi_out)


def _get_mi_from_values_kernel():
    """Lazy-compile + cache the plug-in ``mi_from_values`` RawKernel (one block per column, f64 X)."""
    global _MI_FROM_VALUES_KERNEL
    if _MI_FROM_VALUES_KERNEL is None:
        import cupy as cp
        _MI_FROM_VALUES_KERNEL = cp.RawKernel(_MI_FROM_VALUES_SRC, "mi_from_values")
    return _MI_FROM_VALUES_KERNEL


def _get_mi_split_kernels(cp):
    """Lazy-compile + cache the SPLIT-N (``mi_hist_split`` + ``mi_from_counts_col``) kernel pair used when
    one-block-per-column would starve the SMs for narrow K."""
    global _MI_SPLIT_KERNELS
    if _MI_SPLIT_KERNELS is None:
        mod = cp.RawModule(code=_MI_FROM_VALUES_SRC)
        _MI_SPLIT_KERNELS = (mod.get_function("mi_hist_split"), mod.get_function("mi_from_counts_col"))
    return _MI_SPLIT_KERNELS


def _get_mi_from_values_f32_kernel():
    """f32-X twin of ``mi_from_values`` (f64 edges). Bit-identical codes/MI: X is read via a widening
    float->double promotion, the exact f64 the f64 path stored after its bulk X.astype(f64) upcast."""
    global _MI_FROM_VALUES_F32_KERNEL
    if _MI_FROM_VALUES_F32_KERNEL is None:
        import cupy as cp
        _MI_FROM_VALUES_F32_KERNEL = cp.RawKernel(_MI_FROM_VALUES_F32_SRC, "mi_from_values_f32")
    return _MI_FROM_VALUES_F32_KERNEL


def _get_mi_split_f32_kernels(cp):
    """Lazy-compile + cache the f32-X twin of the SPLIT-N kernel pair."""
    global _MI_SPLIT_F32_KERNELS
    if _MI_SPLIT_F32_KERNELS is None:
        mod = cp.RawModule(code=_MI_FROM_VALUES_F32_SRC)
        _MI_SPLIT_F32_KERNELS = (mod.get_function("mi_hist_split_f32"), mod.get_function("mi_from_counts_col_f32"))
    return _MI_SPLIT_F32_KERNELS


def binned_mi_from_values_gpu(x_vals: Any, interior_edges: Any, y_codes: Any, nbins: int, ky: int, codes_trusted: bool = False) -> np.ndarray | None:
    """Plug-in MI(col_k; y) for an (n,K) float matrix ``x_vals`` binned by per-column ``interior_edges``
    ((nbins-1, K) cupy) in ONE fused RawKernel (bin + joint histogram + MI), replacing _searchsorted_codes
    + binned_mi_from_codes_gpu. Returns a host (K,) float64 array. Selection-equivalent (codes match
    cp.searchsorted side='right' bit-for-bit). Falls back to None if the (nbins*ky) shared tile won't fit."""
    import cupy as cp

    # DTYPE-CHURN KILL (2026-07-03): keep an f32 candidate matrix at f32 through the transpose + kernel
    # (feed the f32-X kernel twin, edges stay f64) instead of upcasting the whole (n,K) matrix back to f64.
    # The f32 X is fed to the SELECTION callers' f32 radix-edge path; re-upcasting it here undid that (nsys
    # F2 1M STRICT: 562 cupy_copy__float32_float64 + 199 transpose_f64). BIT-IDENTICAL: the kernel's only use
    # of X is ``double v = Xc[i]`` (widening float->double promotion == the exact f64 the f64 path stored via
    # X.astype(f64)); edges kept f64 so the comparison is byte-for-byte the f64 path's. Non-f32 input upcasts
    # to f64 as before (the generic contract). The transpose runs in f32 (transpose_f32, half the bytes).
    _is_f32 = getattr(x_vals, "dtype", None) == cp.float32
    Xd = x_vals if _is_f32 else x_vals.astype(cp.float64, copy=False)
    if Xd.ndim == 1:
        Xd = Xd[:, None]
    n, K = int(Xd.shape[0]), int(Xd.shape[1])
    E = cp.ascontiguousarray(interior_edges.astype(cp.float64, copy=False))  # (nbins-1, K)
    yv = y_codes.astype(cp.int64, copy=False).ravel() if isinstance(y_codes, cp.ndarray) else cp.asarray(np.ascontiguousarray(y_codes).astype(np.int64).ravel())
    Ky = int(ky)
    if int(nbins) * Ky * 4 > _MI_FROM_CODES_MAX_SHARED:
        return None
    _assert_codes_in_range(yv, Ky, "binned_mi_from_values_gpu y codes", codes_trusted)
    mi_out = cp.empty(K, dtype=cp.float64)
    # COLUMN-MAJOR X for a COALESCED load (nvprof 2026-07-02: the row-major X[i*K+c] scan was gld_efficiency
    # 28.9% -- strided by K). One coalesced tiled transpose to (K,n) feeds both the single + split kernels,
    # which now index X[c*n+i] (consecutive threads -> consecutive addresses). The kernels' bin codes/MI are
    # unchanged (same values, just a different read layout). Falls back to the row-major contiguous copy if the
    # transpose can't apply (the kernels then need row-major -- but _transpose_to_cm only returns (K,n) here).
    from ._gpu_resident_select import _transpose_to_cm
    Xc = _transpose_to_cm(cp.ascontiguousarray(Xd))   # (K, n) C-order == column-major over the (n,K) matrix
    _inv = np.float64(1.0 / float(max(1, n)))
    # SPLIT-N when one-block-per-column can't fill the SMs (narrow K, big n): 60/66 of these launches were
    # K<=32 blocks on a 6-SM card (nsys 2026-07-02) -> segment the n rows across S blocks/column into a merged
    # global histogram, then a per-column MI pass. Same codes + plug-in MI -> selection-equivalent. The
    # single-vs-split crossover is HW-specific (SM count); consult the per-host kernel_tuning_cache instead of
    # the HW-overfit K<48/n>=262144 magic constants, which stay as the measurement-backed fallback + the
    # MLFRAME_FE_MI_SPLIT env override. Either leg is selection-equivalent, so a cache miss is safe.
    try:
        from .._benchmarks.kernel_tuning_cache.dispatch import lookup_fe_mi_split_backend
        _use_split = lookup_fe_mi_split_backend(n, K) == "split"
    except Exception:
        _use_split = K < 48 and n >= 262144
    if _use_split:
        try:
            S = max(2, min(64, (48 + K - 1) // max(1, K)))
            M = int(nbins) * Ky
            gcounts = cp.zeros(K * M, dtype=cp.int32)
            _hist, _mi = _get_mi_split_f32_kernels(cp) if _is_f32 else _get_mi_split_kernels(cp)
            _hist((K, S), (256,), (Xc, E, yv, np.int64(n), np.int32(K), np.int32(int(nbins)), np.int32(Ky), np.int32(S), gcounts), shared_mem=M * 4)
            _mi((K,), (32,), (gcounts, np.int32(K), np.int32(int(nbins)), np.int32(Ky), _inv, mi_out))
            return np.asarray(cp.asnumpy(mi_out))
        except Exception:
            import logging
            logging.getLogger(__name__).debug("mi split-n path failed; single-launch mi_from_values", exc_info=True)
    _single = _get_mi_from_values_f32_kernel() if _is_f32 else _get_mi_from_values_kernel()
    _single((K,), (256,), (Xc, E, yv, np.int64(n), np.int32(K), np.int32(int(nbins)), np.int32(Ky), _inv, mi_out), shared_mem=int(nbins) * Ky * 4)
    return np.asarray(cp.asnumpy(mi_out))


_MI_FROM_CODES_KERNEL = None  # module-level singleton (lazy-compiled; never on an instance -> pickle-safe)
_MI_FROM_CODES_MAX_SHARED = 44000  # bytes; stay under the 48KB default shared cap (Kx*Ky*4 must fit)


def _get_mi_from_codes_kernel():
    """Lazy-compile + cache the ``mi_from_codes`` RawKernel (plug-in MI over a pre-binned (n,K) code matrix)."""
    global _MI_FROM_CODES_KERNEL
    if _MI_FROM_CODES_KERNEL is None:
        import cupy as cp
        _MI_FROM_CODES_KERNEL = cp.RawKernel(_MI_FROM_CODES_SRC, "mi_from_codes")
    return _MI_FROM_CODES_KERNEL


def binned_mi_from_codes_gpu(code_cols: Any, y_codes: Any, kx_per_col: Any = None, ky: int = 0, codes_trusted: bool = False) -> np.ndarray:
    """Plug-in MI(col_k; y) for EVERY column of ``code_cols`` (n,K) in ONE fused RawKernel launch.

    Drop-in for ``_wavelet_basis_fe_batched.batched_binned_mi_gpu`` (same plain plug-in MI, no MM bias).
    Falls back to that cupy path when the (Kx*Ky) shared tile would exceed the shared-memory cap. Returns
    a host (K,) float64 array. Accepts a host ndarray or a resident cupy code matrix.

    ``codes_trusted`` (default False): the kernel uses each code directly as a shared-tile offset
    (``sh[c*Ky + y]``), so a negative/out-of-range code is an illegal-memory-access crash. The range guard
    screens it (raise ValueError) unless the caller passes binner-produced 0-based codes (then skipped, free)."""
    import cupy as cp

    C = code_cols if isinstance(code_cols, cp.ndarray) else cp.asarray(np.ascontiguousarray(code_cols).astype(np.int64))
    if C.ndim == 1:
        C = C[:, None]
    C = cp.ascontiguousarray(C.astype(cp.int64, copy=False))
    n, K = int(C.shape[0]), int(C.shape[1])
    # code_cols is the per-candidate code matrix (transient) -> NOT cached. A host y_codes is a fit-constant
    # re-uploaded per candidate batch -> resident operand cache (uploaded once per fit; same int64 labels).
    if isinstance(y_codes, cp.ndarray):
        y = y_codes.astype(cp.int64, copy=False).ravel()
    else:
        from ._fe_resident_operands import resident_code_operand
        y = resident_code_operand(np.asarray(y_codes).ravel(), "binned_mi_ycodes")
    Ky = int(ky) if ky > 0 else (int(y.max()) + 1 if y.size else 1)
    Kx = int(np.max(np.asarray(kx_per_col))) if kx_per_col is not None else (int(C.max()) + 1 if C.size else 1)
    Kx = max(Kx, 1)
    _assert_codes_in_range(C, Kx, "binned_mi_from_codes_gpu X codes", codes_trusted)
    _assert_codes_in_range(y, Ky, "binned_mi_from_codes_gpu y codes", codes_trusted)
    if Kx * Ky * 4 > _MI_FROM_CODES_MAX_SHARED:
        from ._wavelet_basis_fe_batched import batched_binned_mi_gpu
        return batched_binned_mi_gpu(C, y, kx_per_col=kx_per_col, ky=Ky)
    mi_out = cp.empty(K, dtype=cp.float64)
    threads = 256
    self_kernel = _get_mi_from_codes_kernel()
    self_kernel((K,), (threads,), (C.ravel(), y, np.int64(n), np.int32(K), np.int32(Kx), np.int32(Ky),
                                   np.float64(1.0 / float(max(1, n))), mi_out),
                shared_mem=Kx * Ky * 4)
    return np.asarray(cp.asnumpy(mi_out))


_QBIN_CODER_RAW = None


def _get_qbin_coder_kernel():
    """Fused one-pass coder for batched_quantile_bin_gpu: per element, count the distinct interior edge
    values <= x (plus the 2-distinct special case) in a single (n, K) read/write, instead of the
    ~3*(nbins+1) full-matrix elementwise passes the broadcast loop costs (nsys on the cupy polynom
    search: that loop was ~76%% of kernel GPU time -- add 41.3%% + greater 15.8%% + bool-copy 14.1%%).
    Bit-identical: same interior/first-occurrence/ndistinct==2 terms, just fused."""
    global _QBIN_CODER_RAW
    if _QBIN_CODER_RAW is None:
        import cupy as cp
        # FP64-compare avoidance (ncu 2026-07-15, cc 8.9: the double-compare loop pinned the FP64 pipeline
        # at 84.4% while memory sat at 24.5% -- consumer Ada runs FP64 at 1:64). Finite IEEE doubles order
        # identically (signed compare) to key = b >= 0 ? b : b ^ 0x7FFF..F (flip magnitude bits of negatives), so convert x
        # ONCE per element and the edges once per call, then all ne comparisons run on the integer pipeline.
        # Signed zeros normalized via +0.0 (else key(-0) < key(+0) diverges from IEEE -0 == +0); non-finite
        # values only occur in columns the caller discards wholesale, so their key order is irrelevant.
        _QBIN_CODER_RAW = cp.RawKernel(r"""
__device__ __forceinline__ long long mono_key(double v) {
    v = v + 0.0;                                 // normalize -0.0 -> +0.0
    long long b = __double_as_longlong(v);
    return b >= 0 ? b : (b ^ 0x7FFFFFFFFFFFFFFFLL);   // signed-compare monotonic map
}
extern "C" __global__ void qbin_code(const double* __restrict__ x, const long long* __restrict__ edge_keys,
                                     const bool* __restrict__ interior, const long long* __restrict__ ndistinct,
                                     const long long n, const long long K, const int ne,
                                     long long* __restrict__ codes) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * K;
    if (t >= total) return;
    long long k = t % K;
    long long vk = mono_key(x[t]);
    long long c = 0;
    for (int j = 1; j < ne; ++j) {              // j=0 is never interior (== column min)
        if (interior[(long long)j * K + k] && vk >= edge_keys[(long long)j * K + k]) ++c;
    }
    if (ndistinct[k] == 2 && vk >= edge_keys[(long long)(ne - 1) * K + k]) ++c;
    codes[t] = c;
}
""", "qbin_code")
    return _QBIN_CODER_RAW


def batched_quantile_bin_gpu(x_cols: Any, nbins: int) -> Any:
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
    # Sort-free radix-select edges first (default ON): cp.percentile(axis=0) implements the batched edge
    # computation as ONE FULL MERGE SORT of the whole (n, K) matrix -- nsys on the wellbore-100k strict fit
    # showed exactly this cub DeviceMergeSort at 74% of ALL GPU time (143.5s, 180 launches, ~2.4s each at
    # 61M elements). _radix_select_interior_edges produces bit-identical interior edges (documented maxdiff
    # 0 in the resulting codes) via rank-select without sorting; q=0/q=100 are exact column min/max (plain
    # reductions). cp.percentile stays the exact fallback when radix is inapplicable (R/shared-mem caps) or
    # gated off via MLFRAME_FE_GPU_RADIX_EDGES=0.
    edges_all = None
    try:
        from ._gpu_resident_select import _radix_select_interior_edges, fe_gpu_radix_edges_enabled
        if fe_gpu_radix_edges_enabled():
            interior = _radix_select_interior_edges(cp.ascontiguousarray(Xd), int(nbins))  # (nbins-1, K) or None
            if interior is not None:
                edges_all = cp.concatenate([Xd.min(axis=0)[None, :], interior, Xd.max(axis=0)[None, :]], axis=0)
    except Exception:
        edges_all = None
    if edges_all is None:
        qs = cp.linspace(0.0, 100.0, nbins + 1)
        edges_all = cp.percentile(Xd, qs, axis=0)  # (nbins+1, K) -- one batched device sort
    # Vectorized per-column coding (replaces a K-iteration Python loop of cp.unique + cp.searchsorted +
    # per-column copy -- ~3 kernel launches PER COLUMN, 617 columns = ~1850 launches per call on the
    # wellbore-100k trace). The searchsorted-on-deduped-interior-edges semantics reduce to counting, per
    # row value x, the DISTINCT interior edge values <= x: with the ascending (nbins+1, K) edge matrix,
    # a first-occurrence mask kills duplicate edges, and excluding each column's min/max edge leaves the
    # interior. The <=2-distinct-edges branches fold in exactly: 1 distinct -> no interior, all-zero codes;
    # 2 distinct -> the legacy special case codes = (x >= max_edge), reproduced via the ndistinct==2 term.
    # Total launches: ~nbins+3 fused elementwise ops over (n, K), independent of K.
    e_first = edges_all[0, :]  # (K,) column min edge (q=0)
    e_last = edges_all[-1, :]  # (K,) column max edge (q=100)
    first_occ = cp.ones(edges_all.shape, dtype=cp.bool_)
    first_occ[1:, :] = edges_all[1:, :] != edges_all[:-1, :]
    interior = first_occ & (edges_all > e_first[None, :]) & (edges_all < e_last[None, :])
    ndistinct = first_occ.sum(axis=0).astype(cp.int64)  # (K,)
    try:
        # Fused one-pass coder (see _get_qbin_coder_kernel) -- bit-identical to the broadcast loop below.
        kern = _get_qbin_coder_kernel()
        codes = cp.empty((n, K), dtype=cp.int64)
        total = n * K
        threads = 256
        blocks = (total + threads - 1) // threads
        eb = (cp.ascontiguousarray(edges_all) + 0.0).view(cp.int64)  # normalize -0.0, reinterpret bits
        edge_keys = cp.where(eb >= 0, eb, eb ^ np.int64(0x7FFFFFFFFFFFFFFF))
        kern((int(blocks),), (threads,), (
            cp.ascontiguousarray(Xd), cp.ascontiguousarray(edge_keys),
            cp.ascontiguousarray(interior), cp.ascontiguousarray(ndistinct),
            np.int64(n), np.int64(K), np.int32(edges_all.shape[0]), codes,
        ))
        return codes
    except Exception:
        logger.debug("qbin fused coder failed; broadcast-loop fallback", exc_info=True)
    codes = cp.zeros((n, K), dtype=cp.int64)
    for j in range(1, int(edges_all.shape[0])):  # j=0 is never interior (== column min)
        codes += (interior[j, :][None, :] & (Xd >= edges_all[j, :][None, :])).astype(cp.int64)
    codes += ((ndistinct == 2)[None, :] & (Xd >= e_last[None, :])).astype(cp.int64)
    return codes


def cmi_device_argmax(mi_d: Any) -> tuple[int, float]:
    """First-max argmax of a RESIDENT (K,) cupy CMI vector, returning ONLY ``(int best_idx, float best_val)``
    via one tiny scalar D2H -- NOT the (K,) vector.

    Tiebreak matches ``np.argmax`` (LOWEST index among ties): ``cp.argmax`` already returns the first
    occurrence of the maximum on a contiguous 1-D array, and that value is byte-identical to the host vector
    the kernel would have produced (same buffer), so ``np.argmax(mi_d.get())`` and this agree exactly. We pull
    the scalar index (int64, 8 B) and gather its value (float64, 8 B) -- two sub-``BULK_BYTES`` D2Hs -- instead
    of D2H-ing the whole (K,) vector and arg-maxing on the host."""
    import cupy as cp

    if mi_d.size == 0:
        return -1, float("-inf")
    idx_d = cp.argmax(mi_d)  # first-max index on device (matches np.argmax tiebreak)
    best_idx = int(idx_d.get())  # 8-byte scalar D2H
    best_val = float(mi_d[best_idx].get())  # 8-byte scalar D2H (gather the single winning value)
    return best_idx, best_val


def batched_cmi_gpu(x_cols: Any, y: np.ndarray, z: Any = None, return_cards: bool = False, codes_trusted: bool = False,
                    return_device: bool = False, precomp_yz: Any = None, kx: int = 0, ky: int = 0) -> Any:
    """Miller-Madow plug-in CMI(x_k; y | z) in nats for EVERY column of ``x_cols``, in ONE device workload.

    ``x_cols`` (n,K) int codes -- a host ndarray OR an already-resident cupy array (born-on-device codes
    from ``batched_quantile_bin_gpu``, no code H2D); ``y`` (n,) int codes (host OR resident cupy -- the
    FIT-CONSTANT label, uploaded ONCE via the resident-operand cache when host); ``z`` (n,) int codes or None
    (marginal MI; host OR resident cupy -- the ROUND-CONSTANT conditioning support, also resident-cached).
    Returns a host (K,) float64 array. Matches ``_mi_greedy_cmi_fe._cmi_from_binned`` per column
    (selection-equivalent).

    ``return_device`` (default False -> byte-identical to today's ``cp.asnumpy`` return): keep the (K,)
    float64 ``mi`` RESIDENT and return the cupy device array instead of D2H-ing the whole vector. The greedy
    loop then takes the argmax on-device (:func:`cmi_device_argmax`) so only the 2 winning scalars cross back
    rather than the (K,) vector. With ``return_cards`` the card arrays are likewise returned resident.

    ``return_cards`` (conditional path only): also return the occupied-cell cardinalities the analytic
    CMI-null df needs -- ``(cmi[K], k_z, k_xz[K], k_yz, k_xyz[K])`` -- computed in the SAME workload (they
    are already produced internally by ``_rows_entropy_and_k`` + the shared y/z terms). Lets the gate score
    the analytic floor/df of ALL round candidates from ONE call instead of a per-candidate
    ``joint_cardinalities_cupy``. The cell counts equal the per-candidate path's (same occupied-cell
    definition) -> df bit-identical.
    """
    import cupy as cp
    from ._fe_resident_operands import resident_operand

    if isinstance(x_cols, cp.ndarray):
        X = x_cols.astype(cp.int64, copy=False)
    else:
        X = cp.asarray(np.ascontiguousarray(x_cols).astype(np.int64))
    if X.ndim == 1:
        X = X[:, None]
    X = cp.ascontiguousarray(X)  # batched joint-hist kernels read X[row*K+col] (C-order (n,K)); no-op if already
    n = int(X.shape[0])
    # y is FIT-CONSTANT: when host, route through the resident-operand cache so it uploads ONCE per fit (keyed
    # on role + shape + content fingerprint) instead of one H2D per candidate batch; when already resident
    # (caller kept a born-on-device y), use as-is (no copy).
    if isinstance(y, cp.ndarray):
        dy = y.astype(cp.int64, copy=False).ravel()
    else:
        from ._fe_resident_operands import resident_code_operand
        dy = resident_code_operand(np.asarray(y).ravel(), "cmi_y")
    nf = float(max(1, n))
    inv_n = 1.0 / nf
    # ``kx`` / ``ky`` (2026-07-02, scalar-sync kill): the histogram WIDTH upper bound. The codes are 0-based
    # equi-frequency bins in [0, nbins-1] and the labels in [0, n_classes-1], so the caller knows the width
    # (nbins / n_classes) -- pass it to SKIP the ``int(X.max())`` / ``int(dy.max())`` blocking device syncs (each
    # drains the GPU queue ~ms; the kernel-timeline gap analysis put ~4,900 such scalar D2H as the dominant
    # remaining GPU-idle source). A width >= the true occupied max is SELECTION-IDENTICAL: the extra trailing
    # bins are always empty -> 0 count -> 0 entropy contribution, and the occupied-cell df (k_x) is counted
    # separately (nnz), unchanged. kx/ky<=0 -> the original .max() sync (unknown-cardinality callers).
    Kx = int(kx) if kx and kx > 0 else (int(X.max()) + 1 if X.size else 1)
    _assert_codes_in_range(X, Kx, "batched_cmi_gpu X codes", codes_trusted)

    if z is None or (hasattr(z, "size") and (z.size if isinstance(z, cp.ndarray) else np.asarray(z).size) == 0):
        # Marginal MI(x_k; y), MM-corrected:  H(x)+H(y)-H(x,y) - (k_x+k_y-k_xy-1)/2n
        Ky = int(ky) if ky and ky > 0 else (int(dy.max()) + 1 if dy.size else 1)
        _assert_codes_in_range(dy, Ky, "batched_cmi_gpu y codes", codes_trusted)
        cnt_xy = _batched_joint_counts2(X, dy, Kx, Ky)  # (K, Kx*Ky) int32; the reduce casts in-register
        h_xy, k_xy = _rows_entropy_and_k(cnt_xy, inv_n)
        cnt_x = _batched_marginal_counts(X, Kx)  # (K, Kx) int32
        h_x, k_x = _rows_entropy_and_k(cnt_x, inv_n)
        h_y, k_y = joint_entropy_gpu([dy], [Ky], inv_n)  # H(y): fused hist+entropy in ONE launch (same 2-launch fallback if it won't fit shared)
        # FUSED assembly: max(h_x - h_xy + h_y - (k_x - k_xy + (k_y-1))/2n, 0) in one (K,) launch.
        mi = _cmi_assemble(h_x, h_xy, float(h_y), k_x, k_xy, float(k_y - 1), 1.0 / (2.0 * nf))
        # return_device: keep the (K,) vector resident (caller device-argmaxes -> 2 scalars cross back, not K).
        if return_device:
            return mi
        return cp.asnumpy(mi)

    # z is ROUND-CONSTANT: when host, resident-cache it (one H2D per round, reused across candidate batches);
    # when already resident, use as-is. The shape+content fingerprint distinguishes successive rounds' z.
    # ``precomp_yz`` (2026-07-02, perm-null chunk hoist): the column-invariant y/z terms
    # ``(dz, Kz, h_z, k_z, yz, Kyz, h_yz, k_yz)`` computed by a PRIOR call over the SAME (y, z). The perm-null's
    # VRAM-chunked driver calls this per perm-chunk (down to 1 perm/chunk at the gate's huge joints), and each
    # chunk re-derived the identical z entropies + yz key + the .max() syncs -- up to ~25 recomputes per null.
    # The SAME values are reused verbatim -> bit-identical CMI; None (all other callers) is unchanged.
    if precomp_yz is not None:
        dz, Kz, h_z, k_z, yz, Kyz, h_yz, k_yz = precomp_yz
    else:
        if isinstance(z, cp.ndarray):
            dz = z.astype(cp.int64, copy=False).ravel()
        else:
            dz = resident_operand(np.asarray(z).ravel(), "cmi_z", dtype=np.int64)
        Kz = int(dz.max()) + 1 if dz.size else 1
        # Conditional path: y and z both index flat histograms (yz = dy*Kz+dz, then (x*Kyz+yz)); guard both.
        Ky_cond = int(dy.max()) + 1 if dy.size else 1
        _assert_codes_in_range(dy, Ky_cond, "batched_cmi_gpu y codes", codes_trusted)
        _assert_codes_in_range(dz, Kz, "batched_cmi_gpu z codes", codes_trusted)
        # shared y/z terms (column-invariant): fused hist+entropy in ONE launch each (same 2-launch fallback when
        # the 1D joint won't fit shared) -- bit-identical to _ent_nnz_1d(joint_counts_gpu(...)).
        h_z, k_z = joint_entropy_gpu([dz], [Kz], inv_n)
        yz = dy * Kz + dz  # dense (y,z) code (also feeds cnt_xyz below)
        Kyz = int(yz.max()) + 1
        h_yz, k_yz = joint_entropy_gpu([yz], [Kyz], inv_n)

    # H(x_k, z) for all k. LAUNCH-FUSION: ONE block/column kernel (shared hist + tree-reduce) collapses the
    # atomicAdd-count + reduce into a single launch (drops the (K, Kx*Kz) f64 intermediate); falls back to the
    # two-launch count+reduce (in-kernel flat key k*(Kx*Kz)+x*Kz+z) when the per-column hist won't fit shared.
    _fused = _batched_joint_entropy_and_k2(X, dz, Kx, Kz, inv_n)
    if _fused is not None:
        h_xz, k_xz = _fused
    else:
        cnt_xz = _batched_joint_counts2(X, dz, Kx, Kz)  # (K, Kx*Kz) int32
        h_xz, k_xz = _rows_entropy_and_k(cnt_xz, inv_n)
    # H(x_k, y, z) for all k: in-kernel flat key k*(Kx*Kyz) + x*Kyz + yz (same fuse; large Kyz -> fallback)
    _fused = _batched_joint_entropy_and_k2(X, yz, Kx, Kyz, inv_n)
    if _fused is not None:
        h_xyz, k_xyz = _fused
    else:
        cnt_xyz = _batched_joint_counts2(X, yz, Kx, Kyz)  # (K, Kx*Kyz) int32
        h_xyz, k_xyz = _rows_entropy_and_k(cnt_xyz, inv_n)

    # FUSED assembly: max(h_xz - h_xyz + (h_yz - h_z) - (k_xyz - k_xz + (k_z - k_yz))/2n, 0) in one (K,) launch.
    cmi_d = _cmi_assemble(h_xz, h_xyz, float(h_yz - h_z), k_xyz, k_xz, float(k_z - k_yz), 1.0 / (2.0 * nf))
    if return_cards:
        # return_device: keep cmi + the per-column card arrays RESIDENT (k_z / k_yz are already host scalars);
        # default False = byte-identical D2H of the cmi vector + the two (K,) int card arrays as today.
        if return_device:
            return (cmi_d, int(k_z), k_xz.astype(cp.int64), int(k_yz), k_xyz.astype(cp.int64))
        return (cp.asnumpy(cmi_d), int(k_z), cp.asnumpy(k_xz).astype(np.int64), int(k_yz), cp.asnumpy(k_xyz).astype(np.int64))
    if return_device:
        return cmi_d
    return cp.asnumpy(cmi_d)


# Back-compat re-exports: the batched CMI count/entropy kernel infrastructure was carved into
# ``_fe_batched_mi_cmi`` (carve-wave, 2026-06-28). Importers use ``from ._fe_batched_mi import X``;
# keep every previously-public name resolving through the parent.
from ._fe_batched_mi_cmi import (  # noqa: F401
    _rows_entropy_and_k,
    _batched_joint_counts2,
    _batched_marginal_counts,
    _batched_joint_entropy_and_k2,
    _ent_nnz_1d,
    joint_counts_gpu,
    joint_entropy_gpu,
    cmi_joint_entropies_gpu,
    marginal_mi_entropies_gpu,
    _cmi_assemble,
    joint_nnz_gpu,
)
