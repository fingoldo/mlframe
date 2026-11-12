"""Batched born-on-device MI / CMI for the FE candidate pool (replatform step 3, 2026-06-25).

WIRED (this docstring was stale): ``batched_cmi_gpu`` /
``batched_quantile_bin_gpu`` / ``cmi_device_argmax`` ARE imported and called from
``_mi_greedy_cmi_fe.py``'s ``greedy_cmi_fe_construct``, production-reachable via
``MRMR.fit(fe_mi_greedy_cmi_enable=True)`` whenever ``_cmi_gpu_enabled()`` is true
(``MLFRAME_CMI_GPU=1`` or STRICT-GPU). Originally built as a parallel/gated module so the batched device
CMI could be built + parity-validated against the per-call CPU path
(``_mi_greedy_cmi_fe._cmi_from_binned``) without touching the primary path before wiring.

The per-call cp.unique CMI ports (cmi_from_binned / fixed_yz) are launch-bound: nsys measured 118k kernel
launches + 5.8k H2D in one F2 1M fit because CMI is called per-candidate-per-permutation, each a separate
cp.unique (5-10 cub launches) + H2D. ``batched_cmi_gpu`` scores CMI(x_k; y | z) for EVERY candidate column
in ONE workload: the y/z-only terms (H(Z), H(Y,Z)) are computed once, then TWO cp.bincount passes over the
flat (k, cx, c*) joint index give H(X_k,Z) and H(X_k,Y,Z) for all k at once -> no per-candidate launch,
one H2D of the (n,K) candidate code matrix. Same Miller-Madow plug-in CMI as the CPU path; partition-based
-> value-order codes are fine -> selection-equivalent (fp reduction order ~1e-15).
"""
from __future__ import annotations

import threading

import logging

from mlframe.utils.log_throttle import log_throttle
import os
from typing import Any

import numpy as np

from pyutilz.performance.gpu_hw_launch import block_size_for

logger = logging.getLogger(__name__)


# The device-code range guard lives in a sibling (file-size carve only); re-exported because the parent is
# its only caller and the kernels below reference it by this name.
from ._fe_batched_mi_guards import _assert_codes_in_range
from ._kernel_types import render

# FUSED MI-FROM-CODES RawKernel (launch-reduction rewrite). ONE launch computes plug-in
# MI(col_k; y) for EVERY column of an (n,K) int code matrix, replacing the cupy chain bincount + reshape
# + sum(axis) x2 + where + log + multiply + sum (~7 cuLaunchKernel) with a single kernel launch (also via
# cuLaunchKernel -> a genuine COUNT reduction, not a driver/runtime counter shift). One block per column:
# the column's (Kx, Ky) joint histogram lives in shared int32 (counts <= n < 2^31); threads stride rows
# and atomicAdd; then a single-thread reduction computes the plug-in MI from the small joint table. The
# math is identical to ``batched_binned_mi_gpu`` (plain plug-in MI, no MM bias) -> selection-equivalent
# (fp reduction order ~1e-15). Used only when the shared tile (Kx*Ky int32) fits; else the cupy path.
_MI_FROM_CODES_SRC = render(r"""
extern "C" __global__
void mi_from_codes(const long long* __restrict__ codes,   // (n, K) row-major int codes in [0,Kx)
                   const long long* __restrict__ y,        // (n,) int codes in [0,Ky)
                   const long long n, const int K, const int Kx, const int Ky,
                   const {ACC} inv_n, {ACC}* __restrict__ mi_out) {
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
        {ACC} mi = 0.0;
        for (int xx = 0; xx < Kx; ++xx) {
            long long rx = 0;
            for (int yy = 0; yy < Ky; ++yy) rx += sh[xx * Ky + yy];
            if (rx == 0) continue;
            {ACC} px = ({ACC})rx * inv_n;
            for (int yy = 0; yy < Ky; ++yy) {
                long long nxy = sh[xx * Ky + yy];
                if (nxy == 0) continue;
                long long ry = 0;
                // bench-attempt-rejected (2026-07-04): ry (the y-marginal) is xx-invariant so this is an
                // O(Kx^2*Ky) redundant single-thread recompute, but it runs in the tid==0 tail after the
                // n-row atomicAdd histogram and is immeasurable vs it (doubling Kx/Ky moves the wall <4%,
                // that being histogram size not the reduce). See _benchmarks/bench_mi_from_codes_ymarginal_hoist.py.
                for (int xx2 = 0; xx2 < Kx; ++xx2) ry += sh[xx2 * Ky + yy];
                {ACC} pxy = ({ACC})nxy * inv_n;
                {ACC} py = ({ACC})ry * inv_n;
                mi += pxy * log(pxy / (px * py));
            }
        }
        mi_out[c] = mi > 0.0 ? mi : 0.0;
    }
}
""")
# TILED mi_from_codes (coalesced, no transpose). One block owns a group of G columns of the row-major (n,K)
# code matrix: the flat loop over rows x G reads codes[i*K + c0 + j] with consecutive threads on consecutive j, so a
# warp's loads are one contiguous run of the row instead of a K-element stride. Each column keeps its own
# (Kx, Ky) histogram in shared memory. G is capped so the G histograms fit the shared tile cap. G=8 chosen from a
# sweep of 4/8/16/32 (G=32 gives 4 blocks at K=100 and leaves the card idle); the public call with this path measured
# 2.20 vs 2.91 ms at 100000x100 and 6.55 vs 19.20 ms at 30000x930 (bench_mi_from_codes_tiled.py, interleaved).
_MI_FROM_CODES_TILE_SRC = render(r"""
extern "C" __global__
void mi_from_codes_tile(const long long* __restrict__ codes,  // (n, K) row-major int codes in [0,Kx)
                        const long long* __restrict__ y,       // (n,) int codes in [0,Ky)
                        const long long n, const int K, const int G, const int Kx, const int Ky,
                        const {ACC} inv_n, {ACC}* __restrict__ mi_out) {
    extern __shared__ int sh[];          // (G, Kx*Ky) joint histograms for this column group
    int M = Kx * Ky;
    int c0 = blockIdx.x * G;
    int g = min(G, K - c0);
    int tid = threadIdx.x, nt = blockDim.x;
    for (int s = tid; s < g * M; s += nt) sh[s] = 0;
    __syncthreads();
    long long total = n * g;
    for (long long e = tid; e < total; e += nt) {
        long long i = e / g;
        int j = (int)(e - i * g);
        long long cx = codes[i * K + c0 + j];
        long long cy = y[i];
        atomicAdd(&sh[j * M + (int)cx * Ky + (int)cy], 1);
    }
    __syncthreads();
    if (tid < g) {
        int* t = sh + tid * M;
        {ACC} mi = 0.0;
        for (int xx = 0; xx < Kx; ++xx) {
            long long rx = 0;
            for (int yy = 0; yy < Ky; ++yy) rx += t[xx * Ky + yy];
            if (rx == 0) continue;
            {ACC} px = ({ACC})rx * inv_n;
            for (int yy = 0; yy < Ky; ++yy) {
                long long nxy = t[xx * Ky + yy];
                if (nxy == 0) continue;
                long long ry = 0;
                for (int xx2 = 0; xx2 < Kx; ++xx2) ry += t[xx2 * Ky + yy];
                {ACC} pxy = ({ACC})nxy * inv_n;
                {ACC} py = ({ACC})ry * inv_n;
                mi += pxy * log(pxy / (px * py));
            }
        }
        mi_out[c0 + tid] = mi > 0.0 ? mi : 0.0;
    }
}
""")
_MI_FROM_CODES_TILE_COLS = 8
_MI_FROM_CODES_TILE_KERNEL = None


def _mi_from_codes_tile_width(cells: int) -> int:
    """Columns per block for the tiled kernel: at most ``_MI_FROM_CODES_TILE_COLS``, and G histograms of ``cells`` int32 must fit the shared cap (0 = none fit)."""
    return min(_MI_FROM_CODES_TILE_COLS, _MI_FROM_CODES_MAX_SHARED // (4 * max(1, cells)))


def _get_mi_from_codes_tile_kernel():
    """Lazy-compile + cache the tiled coalesced ``mi_from_codes_tile`` RawKernel (module-level singleton, pickle-safe)."""
    global _MI_FROM_CODES_TILE_KERNEL
    if _MI_FROM_CODES_TILE_KERNEL is None:
        import cupy as cp
        _MI_FROM_CODES_TILE_KERNEL = cp.RawKernel(_MI_FROM_CODES_TILE_SRC, "mi_from_codes_tile")
    return _MI_FROM_CODES_TILE_KERNEL


# FUSED VALUES->BIN->HIST->MI RawKernel (mega-fusion). For an (n,K) FLOAT matrix + per-column
# interior quantile edges, ONE launch bins each value in-kernel (binary-search upper_bound == cp.searchsorted
# side='right') AND builds the joint histogram AND computes plug-in MI - replacing the separate
# _searchsorted_codes kernel + the (n,K) int code array + binned_mi_from_codes_gpu. One block per column,
# shared (nbins*Ky) int hist. Bin codes equal _searchsorted_codes bit-for-bit (same f64 edges, same
# side='right') -> selection-equivalent. edges: (nbins-1, K) row-major (interior); X: (n,K) row-major.
_MI_FROM_VALUES_TEMPLATE = r"""
#define EDGE_CACHE 256
extern "C" __global__
void mi_from_values(const {REAL}* __restrict__ X, const {REAL}* __restrict__ edges,
                    const long long* __restrict__ y, const long long n, const int K,
                    const int nbins, const int Ky, const {ACC} inv_n, {ACC}* __restrict__ mi_out) {
    extern __shared__ int sh[];          // (nbins*Ky) joint histogram for this column
    int c = blockIdx.x;
    if (c >= K) return;
    int ne = nbins - 1, M = nbins * Ky;
    int tid = threadIdx.x, nt = blockDim.x;
    for (int s = tid; s < M; s += nt) sh[s] = 0;
    __shared__ {REAL} se[EDGE_CACHE];   // this column's interior edges: the binary search reads them per value, strided by K in global memory
    const bool cached = ne <= EDGE_CACHE;
    if (cached) for (int s = tid; s < ne; s += nt) se[s] = edges[(long long)s * K + c];
    __syncthreads();
    const {REAL}* Xc = X + (long long)c * n;          // COLUMN-MAJOR (K,n): coalesced across threads (nvprof
    for (long long i = tid; i < n; i += nt) {          // 2026-07-02: row-major X[i*K+c] was gld_efficiency 28.9%)
        const {REAL} v = Xc[i];
        int lo = 0, hi = ne;                          // upper_bound: count of interior edges <= v
        if (v != v) lo = hi;                        // NaN sorts last: the last bin, as cp.searchsorted does
        while (lo < hi) { int mid = (lo + hi) >> 1; if ((cached ? se[mid] : edges[(long long)mid * K + c]) <= v) lo = mid + 1; else hi = mid; }
        atomicAdd(&sh[lo * Ky + y[i]], 1);
    }
    __syncthreads();
    if (tid == 0) {
        {ACC} mi = 0.0;
        for (int xx = 0; xx < nbins; ++xx) {
            long long rx = 0;
            for (int yy = 0; yy < Ky; ++yy) rx += sh[xx * Ky + yy];
            if (rx == 0) continue;
            {ACC} px = ({ACC})rx * inv_n;
            for (int yy = 0; yy < Ky; ++yy) {
                long long nxy = sh[xx * Ky + yy];
                if (nxy == 0) continue;
                long long ry = 0;
                for (int xx2 = 0; xx2 < nbins; ++xx2) ry += sh[xx2 * Ky + yy];
                mi += ({ACC})nxy * inv_n * log((({ACC})nxy * inv_n) / (px * (({ACC})ry * inv_n)));
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
void mi_hist_split(const {REAL}* __restrict__ X, const {REAL}* __restrict__ edges,
                   const long long* __restrict__ y, const long long n, const int K,
                   const int nbins, const int Ky, const int S, int* __restrict__ gcounts) {
    extern __shared__ int sh[];
    int c = blockIdx.x, seg = blockIdx.y;
    if (c >= K || seg >= S) return;
    int ne = nbins - 1, M = nbins * Ky, tid = threadIdx.x, nt = blockDim.x;
    for (int s = tid; s < M; s += nt) sh[s] = 0;
    __shared__ {REAL} se[EDGE_CACHE];
    const bool cached = ne <= EDGE_CACHE;
    if (cached) for (int s = tid; s < ne; s += nt) se[s] = edges[(long long)s * K + c];
    __syncthreads();
    long long seg_len = (n + S - 1) / S, lo = (long long)seg * seg_len, hi = lo + seg_len;
    if (hi > n) hi = n;
    const {REAL}* Xc = X + (long long)c * n;          // COLUMN-MAJOR (K,n): coalesced load (see mi_from_values)
    for (long long i = lo + tid; i < hi; i += nt) {
        const {REAL} v = Xc[i];
        int a = 0, b = ne;
        while (a < b) { int mid = (a + b) >> 1; if ((cached ? se[mid] : edges[(long long)mid * K + c]) <= v) a = mid + 1; else b = mid; }
        atomicAdd(&sh[a * Ky + y[i]], 1);
    }
    __syncthreads();
    int* gc = gcounts + (long long)c * M;
    for (int s = tid; s < M; s += nt) if (sh[s]) atomicAdd(&gc[s], sh[s]);
}
extern "C" __global__
void mi_from_counts_col(const int* __restrict__ gcounts, const int K, const int nbins, const int Ky,
                        const {ACC} inv_n, {ACC}* __restrict__ mi_out) {
    int c = blockIdx.x;
    if (c >= K) return;
    const int* gc = gcounts + (long long)c * (nbins * Ky);
    if (threadIdx.x != 0) return;                 // M = nbins*Ky ~ 100: single-thread reduce is trivial
    {ACC} mi = 0.0;
    for (int xx = 0; xx < nbins; ++xx) {
        long long rx = 0;
        for (int yy = 0; yy < Ky; ++yy) rx += gc[xx * Ky + yy];
        if (rx == 0) continue;
        {ACC} px = ({ACC})rx * inv_n;
        for (int yy = 0; yy < Ky; ++yy) {
            long long nxy = gc[xx * Ky + yy];
            if (nxy == 0) continue;
            long long ry = 0;
            for (int xx2 = 0; xx2 < nbins; ++xx2) ry += gc[xx2 * Ky + yy];
            mi += ({ACC})nxy * inv_n * log((({ACC})nxy * inv_n) / (px * (({ACC})ry * inv_n)));
        }
    }
    mi_out[c] = mi > 0.0 ? mi : 0.0;
}
"""
_MI_FROM_VALUES_KERNEL = None
_MI_SPLIT_KERNELS = None

# f32-X twin of the values->bin->hist->MI module (dtype-churn kill,, nsys-driven). The opt-in
# f32 binning path (keep_dtype / relax_binning) holds the (n,K) candidate matrix in float32 to feed the
# f32 radix-select edges, but ``binned_mi_from_values_gpu`` re-UPCAST that whole matrix to f64
# (X.astype(f64)) before this kernel AND transposed it in f64 (transpose_f64) - pure churn: the f32->f64
# downcast+upcast round-trip (nsys F2 1M STRICT: 562 cupy_copy__float32_float64 + 199 transpose_f64). The
# ONLY use of X inside the kernel is ``double v = Xc[i]`` (a widening float->double promotion) fed to the
# edge comparison; the f64 path stored exactly ``(double)f32`` there, so reading the value as f32 and
# promoting per-element yields the IDENTICAL f64 -> the bin codes and plug-in MI are BIT-IDENTICAL. The
# EDGES stay ``const double*`` (the interp emits f64; kept f64 so the comparison is byte-for-byte the f64
# path's). Only X's storage/transpose dtype changes: the (n,K) upcast is gone and the transpose runs in
# f32 (transpose_f32, half the bytes). Everything else (histogram int32, MI f64 math) is unchanged.
_MI_FROM_VALUES_SRC = render(_MI_FROM_VALUES_TEMPLATE, real=np.float64)  # Intentional float64: the MI accumulators of these kernels are double
# float32 twin: X and the edges are float32 (the caller passes each double edge rounded UP to float32): for a float32 value x, ``edge <= (double)x`` holds exactly when
# ``ceil_f32(edge) <= x``, so the binary search runs in float32 (double compares are 1/32 rate on the target cards) and the codes are the same.
_MI_FROM_VALUES_F32_SRC = (
    render(_MI_FROM_VALUES_TEMPLATE, real=np.float32)
    .replace("void mi_from_values(", "void mi_from_values_f32(")
    .replace("void mi_hist_split(", "void mi_hist_split_f32(")
    .replace("void mi_from_counts_col(", "void mi_from_counts_col_f32(")
)
_MI_FROM_VALUES_F32_KERNEL = None
_MI_SPLIT_F32_KERNELS = None

# FUSED VALUES->BIN->HIST->MILLER-MADOW-MI. Same one-block-per-column fused bin+joint-hist as
# mi_from_values, but emits the Miller-Madow-corrected MARGINAL MI matching _usability_njit_pool's
# _marginal_mi_njit / _gpu_marginal_mi: mi = H(x)+H(y)-H(x,y) + (kx_occ + k_y - kxy - 1)/(2n), clamped >=0 (Miller-Madow: the added term is the negative of the plug-in bias).
# h_y / k_y are the shared (fit-constant) target entropy + class count, passed in (computed once). This is the
# sync-free, batched, MM-correct kernel the resident pair-combo MI table needs (replaces the per-row
# cp.bincount + .max() sync loop in _gpu_marginal_mi). Selection-equivalent to the njit table (plug-in
# entropies in fp64; reduction-order ~1e-15).
_MI_MM_FROM_VALUES_TEMPLATE = r"""
extern "C" __global__
void mi_mm_from_values(const {REAL}* __restrict__ X, const {EDGE}* __restrict__ edges,
                       const long long* __restrict__ y, const long long n, const int K, const int nbins,
                       const int Ky, const {ACC} inv_n, const {ACC} h_y, const int k_y,
                       {ACC}* __restrict__ mi_out) {
    extern __shared__ int sh[];          // (nbins*Ky) joint histogram for this column
    int c = blockIdx.x;
    if (c >= K) return;
    int ne = nbins - 1, M = nbins * Ky;
    int tid = threadIdx.x, nt = blockDim.x;
    for (int s = tid; s < M; s += nt) sh[s] = 0;
    __shared__ {EDGE} se[256];   // this column's interior edges, read per value by the search
    const bool cached = ne <= 256;
    if (cached) for (int s = threadIdx.x; s < ne; s += blockDim.x) se[s] = edges[(long long)s * K + c];
    __syncthreads();
    for (long long i = tid; i < n; i += nt) {
        {REAL} v = X[i * (long long)K + c];
        int lo = 0, hi = ne;                          // upper_bound == cp.searchsorted side='right'
        if (v != v) lo = hi;                        // NaN sorts last: the last bin, as cp.searchsorted does
        while (lo < hi) { int mid = (lo + hi) >> 1; if ((cached ? se[mid] : edges[(long long)mid * K + c]) <= v) lo = mid + 1; else hi = mid; }
        atomicAdd(&sh[lo * Ky + y[i]], 1);
    }
    __syncthreads();
    if (tid == 0) {
        {ACC} hx = 0.0, hxy = 0.0;
        int kx_occ = 0, kxy = 0;
        for (int xx = 0; xx < nbins; ++xx) {
            long long rx = 0;
            for (int yy = 0; yy < Ky; ++yy) { int nxy = sh[xx * Ky + yy]; rx += nxy; if (nxy > 0) kxy++; }
            if (rx == 0) continue;
            kx_occ++;
            {ACC} px = ({ACC})rx * inv_n; hx -= px * log(px);
            for (int yy = 0; yy < Ky; ++yy) { int nxy = sh[xx * Ky + yy]; if (nxy > 0) { {ACC} pxy = ({ACC})nxy * inv_n; hxy -= pxy * log(pxy); } }
        }
        {ACC} mi = hx + h_y - hxy + (({ACC})(kx_occ + k_y - kxy - 1)) * 0.5 * inv_n;
        mi_out[c] = mi > 0.0 ? mi : 0.0;
    }
}
"""
_MI_MM_FROM_VALUES_KERNEL = None

# NJIT-PARITY EDGE DEDUP for low-cardinality columns. The fused mi_mm_from_values binary-searches
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
void dedup_njit_edges(const {REAL}* __restrict__ edges, const {REAL}* __restrict__ cmin,
                      const {REAL}* __restrict__ cmax, const int ne, const int K,
                      double* __restrict__ out, int* __restrict__ ne_out) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;   // one thread per column
    if (c >= K) return;
    // de[0] is cmin; emit de[1:] (everything strictly AFTER the first distinct value) into out[*, c].
    double prev = (double)cmin[c];
    int w = 0;
    for (int e = 0; e < ne; ++e) {
        double v = (double)edges[(long long)e * K + c];
        if (v != prev) { out[(long long)w * K + c] = v; w++; prev = v; }
    }
    double hv = (double)cmax[c];
    if (hv != prev) { out[(long long)w * K + c] = hv; w++; }
    if (w >= 2) w -= 1;          // drop trailing de[-1] for m>=3 (njit de[1:-1]); keep [de[1]] for m==2
    ne_out[c] = w;
}
"""
_DEDUP_EDGES_KERNELS: dict = {}
_KERNEL_CACHE_LOCK = threading.RLock()


def _get_dedup_edges_kernel(real_t: str = "double"):
    """Lazy-compile + cache the ``dedup_njit_edges`` RawKernel for the edge/bound element type ``real_t`` (module-level, pickle-safe).

    The edges and bounds are read as ``real_t`` (the buffers' own dtype, float32 under the crit default); the emitted edge table is always double."""
    with _KERNEL_CACHE_LOCK:
        kern = _DEDUP_EDGES_KERNELS.get(real_t)
        if kern is None:
            import cupy as cp
            kern = cp.RawKernel(_DEDUP_EDGES_SRC.replace("{REAL}", real_t), "dedup_njit_edges")
            _DEDUP_EDGES_KERNELS[real_t] = kern
    return kern


def _launch_dedup_edges(E, cmin, cmax, ne: int, K: int, Ec, ne_k) -> None:
    """Run ``dedup_njit_edges`` over ``(ne, K)`` edges and ``(K,)`` bounds, reading each at its own dtype (``E.dtype``) and writing the double ``Ec`` / int32 ``ne_k``."""
    import cupy as cp

    real_t = "float" if E.dtype == cp.float32 else "double"
    dedup = _get_dedup_edges_kernel(real_t)
    threads = block_size_for(dedup, work_items=K)
    dedup(((K + threads - 1) // threads,), (threads,), (E, cmin, cmax, np.int32(ne), np.int32(K), Ec, ne_k))


# Length-aware twin of mi_mm_from_values: each column binary-searches only its VALID prefix ne_k[c] of the
# (dedup'd) interior edges - everything else is byte-for-byte the mi_mm_from_values kernel. When ne_k[c] ==
# nbins-1 (continuous columns, no dup/boundary edges) it is bit-identical to mi_mm_from_values.
_MI_MM_FROM_VALUES_SRC = render(_MI_MM_FROM_VALUES_TEMPLATE, real=np.float64, edge=np.float64)  # Intentional float64: the MI accumulators of these kernels are double
_MI_MM_FROM_VALUES_NEK_TEMPLATE = (
    _MI_MM_FROM_VALUES_TEMPLATE.replace("void mi_mm_from_values(", "void mi_mm_from_values_nek(")
    .replace(
        "const {ACC} h_y, const int k_y,\n                       {ACC}* __restrict__ mi_out) {",
        "const {ACC} h_y, const int k_y,\n                       const int* __restrict__ ne_k, {ACC}* __restrict__ mi_out) {")
    .replace("    int ne = nbins - 1, M = nbins * Ky;", "    int ne = ne_k[c], M = nbins * Ky;")
)
if not ("ne_k" in _MI_MM_FROM_VALUES_NEK_TEMPLATE and "int ne = ne_k[c]" in _MI_MM_FROM_VALUES_NEK_TEMPLATE):
    raise AssertionError('"ne_k" in _MI_MM_FROM_VALUES_NEK_TEMPLATE and "int ne = ne_k[c]" in _MI_MM_FROM_VALUES_NEK_TEMPLATE')
_MI_MM_FROM_VALUES_NEK_SRC = render(_MI_MM_FROM_VALUES_NEK_TEMPLATE, real=np.float64, edge=np.float64)  # Intentional float64: the MI accumulators of these kernels are double
_MI_MM_FROM_VALUES_NEK_KERNEL = None

# f32-X twin of the length-aware MM kernel (dtype-churn kill). Same rationale as the f32
# mi_from_values twin: the only use of X is ``double v = X[i*K+c]`` (widening float->double promotion), so
# reading X as f32 and promoting per-element yields the exact f64 the f64 path stored via X.astype(f64) ->
# BIT-IDENTICAL codes/MM-MI. Edges stay f64. Kills the (n,K) f32->f64 upcast on the pair-combo MM path.
_MI_MM_FROM_VALUES_NEK_F32_SRC = render(_MI_MM_FROM_VALUES_NEK_TEMPLATE, real=np.float32, edge=np.float64).replace("void mi_mm_from_values_nek(", "void mi_mm_from_values_nek_f32(")  # Intentional float64: the MI accumulators of these kernels are double
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


# TILED length-aware Miller-Madow MARGINAL MI (coalesced, row-major X, no transpose). The nek kernel gives one block per
# column and reads X[i*K + c] with threads striding rows, so a warp's loads are K elements apart. This block owns G
# columns: the flat loop over rows x G reads X[i*K + c0 + j] with consecutive threads on consecutive j (one contiguous
# run of the row), and each column keeps its own (nbins, Ky) histogram and its cached interior edges in shared memory.
# Same math as mi_mm_from_values_nek: each column searches only its valid prefix ne_k[c]. Measured on the GTX 1050 Ti
# (interleaved, public call): G=4 with 256 threads was fastest of G in {4, 8, 16} x threads in {256, 512}.
_MM_NEK_TILE_TEMPLATE = r"""
extern "C" __global__
void mm_nek_tile(const {REAL}* __restrict__ X, const double* __restrict__ edges,
                 const long long* __restrict__ y, const long long n, const int K, const int nbins,
                 const int Ky, const int G, const double inv_n, const double h_y, const int k_y,
                 const int* __restrict__ ne_k, double* __restrict__ mi_out) {
    extern __shared__ char smem[];
    int M = nbins * Ky;
    int* sh = (int*)smem;                        // (G, M) joint histograms
    double* se = (double*)(smem + G * M * 4);    // (G, 256) cached interior edges of each column
    int c0 = blockIdx.x * G;
    int g = min(G, K - c0);
    int tid = threadIdx.x, nt = blockDim.x;
    for (int s = tid; s < g * M; s += nt) sh[s] = 0;
    int rg = threadIdx.x / g;                    // row group: consecutive threads take consecutive columns of one edge row
    int jl = threadIdx.x % g;
    int ne_l = ne_k[c0 + jl];
    for (int r = rg; r < 256; r += nt / g) {
        if (r < ne_l && ne_l <= 256) se[jl * 256 + r] = edges[(long long)r * K + c0 + jl];
    }
    __syncthreads();
    long long total = n * g;
    for (long long e = tid; e < total; e += nt) {
        long long i = e / g;
        int j = (int)(e - i * g);
        int c = c0 + j;
        double v = X[i * (long long)K + c];
        int ne = ne_k[c];
        bool cached = ne <= 256;
        int lo = 0, hi = ne;                     // upper_bound == cp.searchsorted side='right'
        if (v != v) lo = hi;                     // NaN sorts last: the last bin, as cp.searchsorted does
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            double em = cached ? se[j * 256 + mid] : edges[(long long)mid * K + c];
            if (em <= v) lo = mid + 1; else hi = mid;
        }
        atomicAdd(&sh[j * M + lo * Ky + (int)y[i]], 1);
    }
    __syncthreads();
    if (tid < g) {
        int c = c0 + tid;
        int* t = sh + tid * M;
        double hx = 0.0, hxy = 0.0;
        int kx_occ = 0, kxy = 0;
        for (int xx = 0; xx < nbins; ++xx) {
            long long rx = 0;
            for (int yy = 0; yy < Ky; ++yy) { int nxy = t[xx * Ky + yy]; rx += nxy; if (nxy > 0) kxy++; }
            if (rx == 0) continue;
            kx_occ++;
            double px = (double)rx * inv_n; hx -= px * log(px);
            for (int yy = 0; yy < Ky; ++yy) { int nxy = t[xx * Ky + yy]; if (nxy > 0) { double pxy = (double)nxy * inv_n; hxy -= pxy * log(pxy); } }
        }
        double mi = hx + h_y - hxy + ((double)(kx_occ + k_y - kxy - 1)) * 0.5 * inv_n;
        mi_out[c] = mi > 0.0 ? mi : 0.0;
    }
}
"""
_MM_NEK_TILE_COLS = 4
_MM_NEK_TILE_THREADS = 256
_MM_NEK_TILE_KERNELS: dict = {}  # {is_f32: RawKernel}; module-level, pickle-safe


def _launch_mm_nek_tile(Xc: Any, Ec: Any, yv: Any, n: int, K: int, nbins: int, Ky: int, h_y: float, k_y: int, ne_k: Any, mi_out: Any, is_f32: bool) -> bool:
    """Launch the tiled length-aware MM kernel on the row-major ``Xc``. Returns False when the tile does not fit shared memory or the launch raises, so the caller runs the one-column-per-block kernel."""
    import cupy as cp

    M = nbins * Ky
    G = min(_MM_NEK_TILE_COLS, 44000 // (4 * M + 256 * 8))  # the same 44 KB shared budget as the other MI kernels
    if G < 1:
        return False
    try:
        kern = _MM_NEK_TILE_KERNELS.get(bool(is_f32))
        if kern is None:
            kern = cp.RawKernel(render(_MM_NEK_TILE_TEMPLATE, real=np.float32 if is_f32 else np.float64), "mm_nek_tile")
            _MM_NEK_TILE_KERNELS[bool(is_f32)] = kern
        kern(((K + G - 1) // G,), (_MM_NEK_TILE_THREADS,), (Xc, Ec, yv, np.int64(n), np.int32(K), np.int32(nbins), np.int32(Ky), np.int32(G),  # launch-ok: the tile's edge cache and per-column reduction are written for a fixed 256-lane block (the loader strides by blockDim / G)
                                                             np.float64(1.0 / float(max(1, n))), np.float64(float(h_y)), np.int32(int(k_y)), ne_k, mi_out),
             shared_mem=G * (4 * M + 256 * 8))
    except Exception:
        logger.debug("mm_nek tiled launch failed; falling back to the one-column-per-block kernel", exc_info=True)
        return False
    return True


# Per-column min and max of a row-major (n, K) matrix in ONE coalesced pass. cupy's axis-0 reductions cost 4.4 ms at
# 100000 x 100 (more than the MM kernel they feed); here a block owns 8 columns, thread t reads column t % 8 of rows
# t / 8 + k * (blockDim / 8), so each warp reads a contiguous run of a row. A NaN in a column makes its bound NaN, as
# cupy's min and max do (fmin/fmax would skip it).
_COL_MINMAX_TEMPLATE = r"""
extern "C" __global__
void col_minmax(const {REAL}* __restrict__ X, const long long n, const int K,
                double* __restrict__ cmin, double* __restrict__ cmax) {
    const int G = 8;
    __shared__ double smin[256], smax[256];
    __shared__ int snan[256];
    int c0 = blockIdx.x * G;
    int g = min(G, K - c0);
    int tid = threadIdx.x, nt = blockDim.x;
    int j = tid % G;
    int rg = threadIdx.x / G;                    // row group of the 8 columns: a warp reads whole 64-byte runs of four rows
    double lo = __longlong_as_double(0x7ff0000000000000LL), hi = __longlong_as_double(0xfff0000000000000LL);
    int nan = 0;
    if (j < g) {
        for (long long i = rg; i < n; i += nt / G) {
            double v = (double)X[i * (long long)K + c0 + j];
            if (v != v) nan = 1;
            else { if (v < lo) lo = v; if (v > hi) hi = v; }
        }
    }
    smin[tid] = lo; smax[tid] = hi; snan[tid] = nan;
    __syncthreads();
    if (tid < g) {
        double a = __longlong_as_double(0x7ff0000000000000LL), b = __longlong_as_double(0xfff0000000000000LL);
        int z = 0;
        for (int t = tid; t < nt; t += G) { a = fmin(a, smin[t]); b = fmax(b, smax[t]); z |= snan[t]; }
        double nanv = __longlong_as_double(0x7ff8000000000000LL);
        cmin[c0 + tid] = z ? nanv : a;
        cmax[c0 + tid] = z ? nanv : b;
    }
}
"""
_COL_MINMAX_THREADS = 256  # a multiple of the 8 columns per block
_COL_MINMAX_KERNELS: dict = {}  # {is_f32: RawKernel}; module-level, pickle-safe


def _column_min_max(Xc: Any, n: int, K: int, is_f32: bool) -> tuple:
    """``(cmin, cmax)`` float64 per-column bounds of the row-major ``Xc``: one coalesced kernel, cupy's axis-0 reductions when the launch fails."""
    import cupy as cp

    try:
        kern = _COL_MINMAX_KERNELS.get(bool(is_f32))
        if kern is None:
            kern = cp.RawKernel(render(_COL_MINMAX_TEMPLATE, real=np.float32 if is_f32 else np.float64), "col_minmax")
            _COL_MINMAX_KERNELS[bool(is_f32)] = kern
        cmin = cp.empty(K, dtype=cp.float64)  # Intentional float64: GPU MI edge bounds; compared at full precision
        cmax = cp.empty(K, dtype=cp.float64)  # Intentional float64: GPU MI edge bounds; compared at full precision
        kern(((K + 7) // 8,), (_COL_MINMAX_THREADS,), (Xc, np.int64(n), np.int32(K), cmin, cmax))  # launch-ok: 8 columns per block with the per-thread column index tid % 8: the shared min and max arrays are sized for 256 threads
        return cmin, cmax
    except Exception:
        logger.debug("col_minmax launch failed; cupy axis-0 reductions", exc_info=True)
        return (cp.ascontiguousarray(Xc.min(axis=0).astype(cp.float64, copy=False)),  # Intentional float64: GPU MI edge bounds; compared at full precision
                cp.ascontiguousarray(Xc.max(axis=0).astype(cp.float64, copy=False)))  # Intentional float64: GPU MI edge bounds; compared at full precision


def binned_mm_mi_from_values_gpu(x_vals: Any, interior_edges: Any, y_codes: Any, nbins: int, ky: int, h_y: float, k_y: int, codes_trusted: bool = False,
                                 return_device: bool = False) -> Any:
    """Miller-Madow MARGINAL MI(col_k; y) for an (n,K) float matrix binned by per-column ``interior_edges``
    ((nbins-1, K) cupy), in ONE fused RawKernel (bin + joint hist + MM-MI). ``ky`` is the y-cardinality
    (histogram width; y codes in [0, ky)); ``h_y`` / ``k_y`` are the shared target plug-in entropy +
    OCCUPIED class count used in the bias. Returns a host (K,) double array (clamped >=0), matching
    _gpu_marginal_mi / _marginal_mi_njit. Returns None if the (nbins*ky) shared tile won't fit."""
    import cupy as cp

    # DTYPE-CHURN KILL: keep an f32 candidate matrix at f32 (feed the f32-X MM kernel twin,
    # edges f64) instead of upcasting the whole (n,K) matrix to f64. BIT-IDENTICAL: the kernel reads X only
    # via ``double v = X[i*K+c]`` (widening float->double promotion == the exact f64 the f64 path stored),
    # and cmin/cmax are order statistics of the same f32 values (max/min then exact f64 upcast, unchanged).
    _is_f32 = getattr(x_vals, "dtype", None) == cp.float32
    Xd = x_vals if _is_f32 else x_vals.astype(cp.float64, copy=False)  # Intentional float64: GPU MI kernel operand; kernel contract is float64 (crit switch pending GPU helper review)
    if Xd.ndim == 1:
        Xd = Xd[:, None]
    n, K = int(Xd.shape[0]), int(Xd.shape[1])
    Ky = int(ky)
    if int(nbins) * Ky * 4 > _MI_FROM_CODES_MAX_SHARED:
        return None
    E = cp.ascontiguousarray(interior_edges.astype(cp.float64, copy=False))  # Intentional float64: GPU MI kernel edge vector; compared at full precision
    # x_vals / interior_edges are per-candidate (transient) -> NOT cached. A host y_codes is a fit-constant
    # re-uploaded on every candidate column's MM-MI -> resident operand cache (uploaded once per fit).
    if isinstance(y_codes, cp.ndarray):
        yv = y_codes.astype(cp.int64, copy=False).ravel()
    else:
        from ._fe_resident_operands import resident_operand
        yv = resident_operand(np.asarray(y_codes).ravel(), "binned_mm_ycodes", dtype=np.int64)
    _assert_codes_in_range(yv, Ky, "binned_mm_mi_from_values_gpu y codes", codes_trusted)
    mi_out = cp.empty(K, dtype=cp.float64)  # Intentional float64: GPU MI output accumulator
    Xc = cp.ascontiguousarray(Xd)
    # NJIT-PARITY: dedup the per-column interior edges to njit's distinct-threshold set (drops duplicate +
    # boundary edges that over-bin low-cardinality columns), then bin on only each column's valid prefix.
    ne = int(E.shape[0])
    cmin, cmax = _column_min_max(Xc, n, K, _is_f32)
    # cp.empty (not cp.zeros): the dedup kernel writes out[0:ne_k[c], c] for every column and the nek MI
    # kernel binary-searches ONLY each column's valid prefix (ne = ne_k[c]), so the uninitialised tail rows
    # of Ec are never read -> dropping the per-call (ne,K) zero-fill is bit-identical (was 17k zero-fills/fit).
    # (ne+1, K) not (ne, K): dedup_njit_edges appends the cmax row at index w==ne (out[ne*K+c]) BEFORE the
    # w-=1 decrement (lines 239-241) for all-distinct interior edges, so the buffer must hold nbins rows.
    # That trailing row is written transiently but never read (after w-=1 always ne_k[c] <= ne, and the nek
    # MI kernel reads only rows 0..ne_k[c]-1), so ne_k and all downstream MI are bit-identical; the extra row
    # only stops the stray write from corrupting the adjacent pool block (was cudaErrorIllegalAddress).
    Ec = cp.empty((ne + 1, K), dtype=cp.float64)  # Intentional float64: GPU MI edge buffer; compared at full precision
    ne_k = cp.empty(K, dtype=cp.int32)
    _launch_dedup_edges(E, cmin, cmax, ne, K, Ec, ne_k)
    _mm_nek = _get_mi_mm_from_values_nek_f32_kernel() if _is_f32 else _get_mi_mm_from_values_nek_kernel()
    mm_threads = block_size_for(_mm_nek, dyn_smem=int(nbins) * Ky * 4, work_items=n)
    if not _launch_mm_nek_tile(Xc, Ec, yv, n, K, int(nbins), Ky, h_y, k_y, ne_k, mi_out, _is_f32):
        _mm_nek((K,), (mm_threads,),
            (Xc, Ec, yv, np.int64(n), np.int32(K), np.int32(int(nbins)),
             np.int32(Ky), np.float64(1.0 / float(max(1, n))), np.float64(float(h_y)), np.int32(int(k_y)),  # Intentional float64: GPU MI kernel scalar argument (float64 normalisation)
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


def _edges_rounded_up_f32(cp: Any, edges: Any) -> Any:
    """Each double edge as the smallest float32 not below it (NaN and +-inf kept): ``e <= (double)x`` iff ``up(e) <= x`` for every float32 ``x``."""
    return _ROUND_UP_F32(cp)(edges)  # one launch; the unfused form was six cupy operations (cast, cast, compare, nextafter, where, copy) per call, 4765 calls in a 30000 x 60 fit


_ROUND_UP_F32_KERNEL: Any = None


def _ROUND_UP_F32(cp: Any) -> Any:
    """The fused kernel of :func:`_edges_rounded_up_f32` (built once): ``f = (float)e``; when ``(double)f < e`` the next float32 above ``f``; NaN and infinities pass through."""
    global _ROUND_UP_F32_KERNEL
    if _ROUND_UP_F32_KERNEL is None:
        _ROUND_UP_F32_KERNEL = cp.ElementwiseKernel(
            "float64 e", "float32 o", "float f = (float)e; o = ((double)f < e) ? nextafterf(f, __int_as_float(0x7f800000)) : f;", "mlframe_edges_rounded_up_f32"  # dtype-ok: Intentional float64: the edge is compared with its float32 rounding at full precision
        )
    return _ROUND_UP_F32_KERNEL


def binned_mi_from_values_gpu(x_vals: Any, interior_edges: Any, y_codes: Any, nbins: int, ky: int, codes_trusted: bool = False, x_cm: Any = None) -> np.ndarray | None:
    """Plug-in MI(col_k; y) for an (n,K) float matrix ``x_vals`` binned by per-column ``interior_edges``
    ((nbins-1, K) cupy) in ONE fused RawKernel (bin + joint histogram + MI), replacing _searchsorted_codes
    + binned_mi_from_codes_gpu. Returns a host (K,) double array. Selection-equivalent (codes match
    cp.searchsorted side='right' bit-for-bit). Falls back to None if the (nbins*ky) shared tile won't fit.

    ``x_cm``: the ``(K, n)`` C-order column-major copy of ``x_vals`` when the caller already has one (the radix edge select needs the same buffer), so the matrix is not transposed a second time."""
    import cupy as cp

    # DTYPE-CHURN KILL: keep an f32 candidate matrix at f32 through the transpose + kernel
    # (feed the f32-X kernel twin, edges stay f64) instead of upcasting the whole (n,K) matrix back to f64.
    # The f32 X is fed to the SELECTION callers' f32 radix-edge path; re-upcasting it here undid that (nsys
    # F2 1M STRICT: 562 cupy_copy__float32_float64 + 199 transpose_f64). BIT-IDENTICAL: the kernel's only use
    # of X is ``double v = Xc[i]`` (widening float->double promotion == the exact f64 the f64 path stored via
    # X.astype(f64)); edges kept f64 so the comparison is byte-for-byte the f64 path's. Non-f32 input upcasts
    # to f64 as before (the generic contract). The transpose runs in f32 (transpose_f32, half the bytes).
    if x_vals is None:
        # column-major only: the (K, n) block is all there is, the row-major matrix never exists
        _is_f32 = x_cm.dtype == cp.float32
        Xd = None
        K, n = int(x_cm.shape[0]), int(x_cm.shape[1])
    else:
        _is_f32 = getattr(x_vals, "dtype", None) == cp.float32
        Xd = x_vals if _is_f32 else x_vals.astype(cp.float64, copy=False)  # Intentional float64: GPU MI kernel operand; kernel contract is float64
        if Xd.ndim == 1:
            Xd = Xd[:, None]
        n, K = int(Xd.shape[0]), int(Xd.shape[1])
    E = cp.ascontiguousarray(interior_edges.astype(cp.float64, copy=False))  # (nbins-1, K)  # Intentional float64: GPU MI edge vector; compared at full precision
    if _is_f32:
        E = _edges_rounded_up_f32(cp, E)  # the float32 search of the f32-X kernels gives the double comparison's codes
    yv = y_codes.astype(cp.int64, copy=False).ravel() if isinstance(y_codes, cp.ndarray) else cp.asarray(np.ascontiguousarray(y_codes).astype(np.int64).ravel())
    Ky = int(ky)
    if int(nbins) * Ky * 4 > _MI_FROM_CODES_MAX_SHARED:
        return None
    _assert_codes_in_range(yv, Ky, "binned_mi_from_values_gpu y codes", codes_trusted)
    mi_out = cp.empty(K, dtype=cp.float64)  # Intentional float64: GPU MI output accumulator
    # COLUMN-MAJOR X for a COALESCED load (nvprof: the row-major X[i*K+c] scan was gld_efficiency
    # 28.9% - strided by K). One coalesced tiled transpose to (K,n) feeds both the single + split kernels,
    # which now index X[c*n+i] (consecutive threads -> consecutive addresses). The kernels' bin codes/MI are
    # unchanged (same values, just a different read layout). Falls back to the row-major contiguous copy if the
    # transpose can't apply (the kernels then need row-major - but _transpose_to_cm only returns (K,n) here).
    from ._gpu_resident_select import _transpose_to_cm
    if x_vals is None:
        Xc = x_cm if x_cm.dtype == (cp.float32 if _is_f32 else cp.float64) and x_cm.flags.c_contiguous else cp.ascontiguousarray(x_cm, dtype=cp.float32 if _is_f32 else cp.float64)  # Intentional float64: GPU MI operand; kernel contract is float64
    elif x_cm is not None and Xd is not None and x_cm.shape == (K, n) and x_cm.dtype == Xd.dtype and x_cm.flags.c_contiguous:
        Xc = x_cm
    else:
        Xc = _transpose_to_cm(cp.ascontiguousarray(Xd))  # (K, n) C-order == column-major over the (n,K) matrix
    _inv = np.float64(1.0 / float(max(1, n)))  # Intentional float64: MI normalisation scalar (float64)
    # SPLIT-N when one-block-per-column can't fill the SMs (narrow K, big n): 60/66 of these launches were
    # K<=32 blocks on a 6-SM card (nsys) -> segment the n rows across S blocks/column into a merged
    # global histogram, then a per-column MI pass. Same codes + plug-in MI -> selection-equivalent. The
    # single-vs-split crossover is HW-specific (SM count); consult the per-host kernel_tuning_cache instead of
    # the HW-overfit K<48/n>=262144 magic constants, which stay as the measurement-backed fallback + the
    # MLFRAME_FE_MI_SPLIT env override. Either leg is selection-equivalent, so a cache miss is safe.
    try:
        from .._benchmarks.kernel_tuning_cache.dispatch import lookup_fe_mi_split_backend
        _use_split = lookup_fe_mi_split_backend(n, K) == "split"
    except Exception as e:
        logger.warning("kernel_tuning_cache lookup for fe_mi split backend failed, using the size-based fallback: %s", e)
        _use_split = K < 48 and n >= 262144
    if _use_split:
        try:
            S = max(2, min(64, (48 + K - 1) // max(1, K)))
            M = int(nbins) * Ky
            gcounts = cp.zeros(K * M, dtype=cp.int32)
            _hist, _mi = _get_mi_split_f32_kernels(cp) if _is_f32 else _get_mi_split_kernels(cp)
            hist_threads = block_size_for(_hist, dyn_smem=M * 4, work_items=n)
            _hist((K, S), (hist_threads,), (Xc, E, yv, np.int64(n), np.int32(K), np.int32(int(nbins)), np.int32(Ky), np.int32(S), gcounts), shared_mem=M * 4)
            _mi((K,), (32,), (gcounts, np.int32(K), np.int32(int(nbins)), np.int32(Ky), _inv, mi_out))  # launch-ok: only thread 0 of each block works (a ~100-cell serial reduce), the block width has no effect
            return np.asarray(cp.asnumpy(mi_out))
        except Exception:
            import logging
            logging.getLogger(__name__).debug("mi split-n path failed; single-launch mi_from_values", exc_info=True)
    _single = _get_mi_from_values_f32_kernel() if _is_f32 else _get_mi_from_values_kernel()
    single_threads = block_size_for(_single, dyn_smem=int(nbins) * Ky * 4, work_items=n)
    _single((K,), (single_threads,), (Xc, E, yv, np.int64(n), np.int32(K), np.int32(int(nbins)), np.int32(Ky), _inv, mi_out), shared_mem=int(nbins) * Ky * 4)
    return np.asarray(cp.asnumpy(mi_out))


_MI_FROM_CODES_KERNEL = None  # module-level singleton (lazy-compiled; never on an instance -> pickle-safe)
_MI_FROM_CODES_MAX_SHARED = 44000  # bytes; stay under the 48KB default shared cap (Kx*Ky*4 must fit)


# bench-attempt-rejected x2 (ncu-led, 2026-07-15, cc 8.9): ncu flagged mi_from_codes' stride-K reads as 80%
# excessive sectors (est. speedup 72%) - both coalescing rewrites LOST at (99401,100)/Kx=Ky=21:
#   (a) flat grid-stride + GLOBAL per-column int32 hist atomics: 9.14ms vs 4.07ms (0.44x) - each warp's 32
#       consecutive t hit 32 DIFFERENT columns' histograms = one cache line per thread, serialized L2 atomics;
#   (b) this int16-cast + tiled-transpose + contiguous-column variant (kept, opt-in MLFRAME_MI_FROM_CODES_V2=1):
#       7.49ms (0.52x) - the cast+transpose passes cost more than the strided reads they remove.
# Root cause of the ncu false lead: the per-column blocks stream the same rows nearly in lockstep, so the
# "excessive" L1 wavefronts are served from L2 (no DRAM amplification); the real limiter is the shared-memory
# histogram atomics, which every variant pays identically. The row-major one-kernel form is the measured
# optimum; v2 stays as a runnable option for future hardware where L2 no longer absorbs the broadcast.
_MI_FROM_CODES_V2_SRC = render(r"""
extern "C" __global__
void mi_from_codes_cm_i16(const short* __restrict__ codes_cm,  // (K, n) C-order: column c contiguous
                          const short* __restrict__ y,          // (n,) int16 codes in [0,Ky)
                          const long long n, const int K, const int Kx, const int Ky,
                          const {ACC} inv_n, {ACC}* __restrict__ mi_out) {
    extern __shared__ int sh[];
    int c = blockIdx.x;
    if (c >= K) return;
    int M = Kx * Ky;
    int tid = threadIdx.x, nt = blockDim.x;
    for (int s = tid; s < M; s += nt) sh[s] = 0;
    __syncthreads();
    const short* col = codes_cm + (long long)c * n;
    for (long long i = tid; i < n; i += nt) {
        atomicAdd(&sh[(int)col[i] * Ky + (int)y[i]], 1);
    }
    __syncthreads();
    if (tid == 0) {
        {ACC} mi = 0.0;
        for (int xx = 0; xx < Kx; ++xx) {
            long long rx = 0;
            for (int yy = 0; yy < Ky; ++yy) rx += sh[xx * Ky + yy];
            if (rx == 0) continue;
            {ACC} px = ({ACC})rx * inv_n;
            for (int yy = 0; yy < Ky; ++yy) {
                long long nxy = sh[xx * Ky + yy];
                if (nxy == 0) continue;
                long long ry = 0;
                for (int xx2 = 0; xx2 < Kx; ++xx2) ry += sh[xx2 * Ky + yy];
                {ACC} pxy = ({ACC})nxy * inv_n;
                {ACC} py = ({ACC})ry * inv_n;
                mi += pxy * log(pxy / (px * py));
            }
        }
        mi_out[c] = mi > 0.0 ? mi : 0.0;
    }
}
""")
_MI_FROM_CODES_V2_KERNELS = None


def _get_mi_from_codes_v2_kernels():
    """Lazy-compiled column-major int16 RawKernel of the coalesced v2 path, or None when disabled
    (MLFRAME_MI_FROM_CODES_V2=0) / compile failure (the row-major one-kernel form is the automatic fallback)."""
    global _MI_FROM_CODES_V2_KERNELS
    if os.environ.get("MLFRAME_MI_FROM_CODES_V2", "0").strip().lower() not in ("1", "true", "on", "yes"):
        return None
    if _MI_FROM_CODES_V2_KERNELS is None:
        import cupy as cp
        try:
            _MI_FROM_CODES_V2_KERNELS = cp.RawKernel(_MI_FROM_CODES_V2_SRC, "mi_from_codes_cm_i16")
        except Exception as exc:
            # Latch only a PERMANENT compile failure. A syntax or arch problem will fail identically every time
            # and latching saves repeated compile attempts; a transient nvrtc/DLL fault -- a mode this repo
            # documents elsewhere -- is indistinguishable here, and latching it cost the fused path for the whole
            # run on a debug line. Warn, and leave the sentinel unset so the next call retries the compile.
            log_throttle(
                logger,
                "mi_from_codes_v2_compile_failed",
                logging.WARNING,
                "mi_from_codes v2 RawKernel compile failed (%s: %s); using the one-kernel fallback for this call "
                "and retrying the compile next time rather than latching the slower path for the process.",
                type(exc).__name__,
                exc,
            )
            return None
    return _MI_FROM_CODES_V2_KERNELS or None


def _get_mi_from_codes_kernel():
    """Lazy-compile + cache the ``mi_from_codes`` RawKernel (plug-in MI over a pre-binned (n,K) code matrix)."""
    global _MI_FROM_CODES_KERNEL
    if _MI_FROM_CODES_KERNEL is None:
        import cupy as cp
        _MI_FROM_CODES_KERNEL = cp.RawKernel(_MI_FROM_CODES_SRC, "mi_from_codes")
    return _MI_FROM_CODES_KERNEL


def binned_mi_from_codes_gpu(code_cols: Any, y_codes: Any, kx_per_col: Any = None, ky: int = 0, codes_trusted: bool = False, as_device: bool = False) -> Any:
    """Plug-in MI(col_k; y) for EVERY column of ``code_cols`` (n,K) in ONE fused RawKernel launch.

    Drop-in for ``_wavelet_basis_fe_batched.batched_binned_mi_gpu`` (same plain plug-in MI, no MM bias).
    Falls back to that cupy path when the (Kx*Ky) shared tile would exceed the shared-memory cap. Returns
    a host (K,) double array by default (``as_device=False``, unchanged contract for existing callers).
    Accepts a host ndarray or a resident cupy code matrix.

    ``as_device=True`` (2026-07-20, GPU-residency pass): skip the ``cp.asnumpy`` sync and return the
    cupy device array directly. For a caller doing several of these calls back-to-back per generation
    (e.g. one per candidate binary-function in ``_cupy_polynom_optimizer``'s search loop) and only
    needing the host values ONCE at the end (to pick a winner), forcing a device->host sync on every
    individual call was measured (nsys, wellbore 100k GPU-strict trace) to multiply
    cudaStreamSynchronize/cudaMemcpyAsync call counts well beyond what the algorithm needs - letting
    the caller batch its own single sync at the point it actually needs host values removes that.

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
        out = batched_binned_mi_gpu(C, y, kx_per_col=kx_per_col, ky=Ky)
        return cp.asarray(out) if as_device else out
    mi_out = cp.empty(K, dtype=cp.float64)  # Intentional float64: GPU MI output accumulator
    G = _mi_from_codes_tile_width(Kx * Ky)
    if G >= 1:
        tile = _get_mi_from_codes_tile_kernel()
        try:
            # 512 threads: fastest at 100000x100 (1.25 vs 1.78 ms at 256; block_size_for's 96 gives 4.02 ms), within 3% of 256 at 30000x930 (4.01 vs 3.89 ms). Interleaved sweep.
            tile(((K + G - 1) // G,), (512,),(C.ravel(), y, np.int64(n), np.int32(K), np.int32(G), np.int32(Kx), np.int32(Ky),  # launch-ok: 512 threads measured fastest of 96/128/256/512 at 100000 x 100 and 30000 x 930 (interleaved sweep: 1.25 vs 4.02 ms at 96, the block_size_for choice)
                                                   np.float64(1.0 / float(max(1, n))), mi_out),  # Intentional float64: GPU MI kernel scalar argument (float64 normalisation)
                 shared_mem=G * Kx * Ky * 4)
            return mi_out if as_device else np.asarray(cp.asnumpy(mi_out))
        except Exception:
            logger.debug("mi_from_codes tiled launch failed; falling back to the row-major kernel", exc_info=True)
    v2 = _get_mi_from_codes_v2_kernels()
    if v2 is not None and Kx < 32768 and Ky < 32768:
        try:
            from ._gpu_resident_select import transpose_codes_to_cm
            codes_cm = transpose_codes_to_cm(cp.ascontiguousarray(C.astype(cp.int16, copy=False)))  # (K, n)
            y16 = y.astype(cp.int16, copy=False)
            threads = block_size_for(v2, dyn_smem=Kx * Ky * 4, work_items=n)
            v2((K,), (threads,), (codes_cm, y16, np.int64(n), np.int32(K), np.int32(Kx), np.int32(Ky),
                                  np.float64(1.0 / float(max(1, n))), mi_out), shared_mem=Kx * Ky * 4)  # Intentional float64: GPU MI kernel scalar argument (float64 normalisation)
            return mi_out if as_device else np.asarray(cp.asnumpy(mi_out))
        except Exception:
            logger.debug("mi_from_codes v2 launch failed; one-kernel fallback", exc_info=True)
    self_kernel = _get_mi_from_codes_kernel()
    threads = block_size_for(self_kernel, dyn_smem=Kx * Ky * 4, work_items=n)
    self_kernel((K,), (threads,), (C.ravel(), y, np.int64(n), np.int32(K), np.int32(Kx), np.int32(Ky),
                                   np.float64(1.0 / float(max(1, n))), mi_out),  # Intentional float64: GPU MI kernel scalar argument (float64 normalisation)
                shared_mem=Kx * Ky * 4)
    return mi_out if as_device else np.asarray(cp.asnumpy(mi_out))


# Quantile binning lives in ``_fe_batched_qbin`` (file-size carve); re-exported so callers keep importing from here.
from ._fe_batched_qbin import _get_qbin_coder_kernel, batched_quantile_bin_gpu  # noqa: F401


def cmi_device_argmax(mi_d: Any) -> tuple[int, float]:
    """First-max argmax of a RESIDENT (K,) cupy CMI vector, returning ONLY ``(int best_idx, float best_val)``
    via one tiny scalar D2H - NOT the (K,) vector.

    Tiebreak matches ``np.argmax`` (LOWEST index among ties): ``cp.argmax`` already returns the first
    occurrence of the maximum on a contiguous 1-D array, and that value is byte-identical to the host vector
    the kernel would have produced (same buffer), so ``np.argmax(mi_d.get())`` and this agree exactly. We pull
    the scalar index (int64, 8 B) and gather its value (double, 8 B) - two sub-``BULK_BYTES`` D2Hs - instead
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

    ``x_cols`` (n,K) int codes - a host ndarray OR an already-resident cupy array (born-on-device codes
    from ``batched_quantile_bin_gpu``, no code H2D); ``y`` (n,) int codes (host OR resident cupy - the
    FIT-CONSTANT label, uploaded ONCE via the resident-operand cache when host); ``z`` (n,) int codes or None
    (marginal MI; host OR resident cupy - the ROUND-CONSTANT conditioning support, also resident-cached).
    Returns a host (K,) double array. Matches ``_mi_greedy_cmi_fe._cmi_from_binned`` per column
    (selection-equivalent).

    ``return_device`` (default False -> byte-identical to today's ``cp.asnumpy`` return): keep the (K,)
    double ``mi`` RESIDENT and return the cupy device array instead of D2H-ing the whole vector. The greedy
    loop then takes the argmax on-device (:func:`cmi_device_argmax`) so only the 2 winning scalars cross back
    rather than the (K,) vector. With ``return_cards`` the card arrays are likewise returned resident.

    ``return_cards`` (conditional path only): also return the occupied-cell cardinalities the analytic
    CMI-null df needs - ``(cmi[K], k_z, k_xz[K], k_yz, k_xyz[K])`` - computed in the SAME workload (they
    are already produced internally by ``_rows_entropy_and_k`` + the shared y/z terms). Lets the gate score
    the analytic floor/df of ALL round candidates from ONE call instead of a per-candidate
    ``joint_cardinalities_cupy``. The cell counts equal the per-candidate path's (same occupied-cell
    definition) -> df bit-identical.
    """
    import cupy as cp
    from ._fe_resident_operands import resident_operand

    if isinstance(x_cols, cp.ndarray):
        X = x_cols if x_cols.dtype == cp.uint8 else x_cols.astype(cp.int64, copy=False)  # a byte matrix stays bytes: the histogram kernels read it with an eighth of the traffic
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
    # ``kx`` / ``ky`` (scalar-sync kill): the histogram WIDTH upper bound. The codes are 0-based
    # equi-frequency bins in [0, nbins-1] and the labels in [0, n_classes-1], so the caller knows the width
    # (nbins / n_classes) - pass it to SKIP the ``int(X.max())`` / ``int(dy.max())`` blocking device syncs (each
    # drains the GPU queue ~ms; the kernel-timeline gap analysis put ~4,900 such scalar D2H as the dominant
    # remaining GPU-idle source). A width >= the true occupied max is SELECTION-IDENTICAL: the extra trailing
    # bins are always empty -> 0 count -> 0 entropy contribution, and the occupied-cell df (k_x) is counted
    # separately (nnz), unchanged. kx/ky<=0 -> the original .max() sync (unknown-cardinality callers).
    Kx = int(kx) if kx and kx > 0 else (int(X.max()) + 1 if X.size else 1)
    _assert_codes_in_range(X, Kx, "batched_cmi_gpu X codes", codes_trusted)

    if z is None or (hasattr(z, "size") and (z.size if isinstance(z, cp.ndarray) else size_of(z)) == 0):
        # Marginal MI(x_k; y), MM-corrected:  H(x)+H(y)-H(x,y) - (k_x+k_y-k_xy-1)/2n
        Ky = int(ky) if ky and ky > 0 else (int(dy.max()) + 1 if dy.size else 1)
        _assert_codes_in_range(dy, Ky, "batched_cmi_gpu y codes", codes_trusted)
        cnt_xy = _batched_joint_counts2(X, dy, Kx, Ky)  # (K, Kx*Ky) int32; the reduce casts in-register
        h_xy, k_xy = _rows_entropy_and_k(cnt_xy, inv_n)
        cnt_x = _batched_marginal_counts(X, Kx)  # (K, Kx) int32
        h_x, k_x = _rows_entropy_and_k(cnt_x, inv_n)
        h_y, k_y = joint_entropy_gpu([dy], [Ky], inv_n)  # H(y): fused hist+entropy in ONE launch (same 2-launch fallback if it won't fit shared)
        # FUSED assembly: max(h_x - h_xy + h_y + (k_x - k_xy + (k_y-1))/2n, 0) in one (K,) launch (the Miller-Madow term is added, so inv2n is negated).
        mi = _cmi_assemble(h_x, h_xy, float(h_y), k_x, k_xy, float(k_y - 1), -1.0 / (2.0 * nf))
        # return_device: keep the (K,) vector resident (caller device-argmaxes -> 2 scalars cross back, not K).
        if return_device:
            return mi
        return cp.asnumpy(mi)

    # z is ROUND-CONSTANT: when host, resident-cache it (one H2D per round, reused across candidate batches);
    # when already resident, use as-is. The shape+content fingerprint distinguishes successive rounds' z.
    # ``precomp_yz`` (perm-null chunk hoist): the column-invariant y/z terms
    # ``(dz, Kz, h_z, k_z, yz, Kyz, h_yz, k_yz)`` computed by a PRIOR call over the SAME (y, z). The perm-null's
    # VRAM-chunked driver calls this per perm-chunk (down to 1 perm/chunk at the gate's huge joints), and each
    # chunk re-derived the identical z entropies + yz key + the .max() syncs - up to ~25 recomputes per null.
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
        # the 1D joint won't fit shared) - bit-identical to _ent_nnz_1d(joint_counts_gpu(...)).
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
# ``_fe_batched_mi_cmi`` (carve-wave). Importers use ``from ._fe_batched_mi import X``;
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
from mlframe.feature_selection.filters._lazy_host_codes import size_of
