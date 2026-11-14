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
// coalescing-ok: measured reject (tracker GPU11-45, GPU11-47): the flat grid-stride rewrite ran at 0.44x and the column-major int16 variant at 0.52x; the blocks stream the same rows in lockstep so the strided reads are served from L2, and the limiter is the shared histogram atomics every variant pays
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
from ._fe_batched_mi_mm_values import (  # noqa: F401  -- carved: the Miller-Madow MI from raw values
    _MI_FROM_CODES_MAX_SHARED,
    _get_dedup_edges_kernel,
    _launch_dedup_edges,
    _get_mi_mm_from_values_nek_kernel,
    _get_mi_mm_from_values_nek_f32_kernel,
    _get_mi_mm_from_values_kernel,
    _launch_mm_nek_tile,
    _column_min_max,
    binned_mm_mi_from_values_gpu,
    _MI_MM_FROM_VALUES_KERNEL,
    _DEDUP_EDGES_SRC,
    _DEDUP_EDGES_KERNELS,
    _KERNEL_CACHE_LOCK,
    _MI_MM_FROM_VALUES_SRC,
    _MI_MM_FROM_VALUES_NEK_TEMPLATE,
    _MI_MM_FROM_VALUES_NEK_SRC,
    _MI_MM_FROM_VALUES_NEK_KERNEL,
    _MI_MM_FROM_VALUES_NEK_F32_SRC,
    _MI_MM_FROM_VALUES_NEK_F32_KERNEL,
    _MM_NEK_TILE_TEMPLATE,
    _MM_NEK_TILE_COLS,
    _MM_NEK_TILE_THREADS,
    _MM_NEK_TILE_KERNELS,
    _COL_MINMAX_TEMPLATE,
    _COL_MINMAX_THREADS,
    _COL_MINMAX_KERNELS,
)


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


def _batched_cmi_gpu_once(x_cols: Any, y: np.ndarray, z: Any = None, return_cards: bool = False, codes_trusted: bool = False,
                    return_device: bool = False, precomp_yz: Any = None, kx: int = 0, ky: int = 0, precomp_xz: Any = None) -> Any:
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
    # ``precomp_xz`` (h, k): the (x, z) joint entropy and occupied-cell count shared by every column, for columns that are within-stratum shuffles of one candidate (the permutation null): a
    # shuffle inside the strata of z leaves the (x, z) joint unchanged, so one count serves all of them.
    if precomp_xz is not None:
        h_xz = cp.full(X.shape[1], float(precomp_xz[0]), dtype=cp.float64)  # Intentional float64: entropy terms of the MI statistic
        k_xz = cp.full(X.shape[1], int(precomp_xz[1]), dtype=cp.int64)
    else:
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
from ._fe_batched_mi_blocks import batched_cmi_gpu  # noqa: F401  -- carved: the column-block driver around ``_batched_cmi_gpu_once``
from mlframe.feature_selection.filters._lazy_host_codes import size_of
