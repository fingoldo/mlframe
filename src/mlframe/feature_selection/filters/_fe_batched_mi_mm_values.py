"""Miller-Madow MI from raw values against the label codes, on the device: the dedup-edges kernel, the per-column edge-count (NEK) kernels in float and double, the 4-column tile and the
column min/max pass, behind ``binned_mm_mi_from_values_gpu``.

Carved out of ``_fe_batched_mi`` (which re-exports every name here) to keep that module under the line budget.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from pyutilz.performance.gpu_hw_launch import block_size_for

from mlframe.utils.log_throttle import log_throttle

from ._fe_batched_mi_guards import _assert_codes_in_range
from ._kernel_types import render

logger = logging.getLogger("mlframe.feature_selection.filters._fe_batched_mi")

_MI_FROM_CODES_MAX_SHARED = 44000  # bytes; stay under the 48KB default shared cap (Kx*Ky*4 must fit)


_MI_MM_FROM_VALUES_TEMPLATE = r"""
// coalescing-ok: measured reject of the column-major read (tracker UC11-07, bench_mi_mm_from_values_layout.py): the block is built per call and used once, so the transpose is paid every time and eats the 30 percent the kernel alone gains at K = 256; production runs the tiled mm_nek_tile and col_minmax twins
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
                      {ACC}* __restrict__ out, int* __restrict__ ne_out) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;   // one thread per column
    if (c >= K) return;
    // de[0] is cmin; emit de[1:] (everything strictly AFTER the first distinct value) into out[*, c].
    {ACC} prev = ({ACC})cmin[c];
    int w = 0;
    for (int e = 0; e < ne; ++e) {
        {ACC} v = ({ACC})edges[(long long)e * K + c];
        if (v != prev) { out[(long long)w * K + c] = v; w++; prev = v; }
    }
    {ACC} hv = ({ACC})cmax[c];
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
            kern = cp.RawKernel(_DEDUP_EDGES_SRC.replace("{ACC}", "double").replace("{REAL}", real_t), "dedup_njit_edges")  # Intentional float64: edges and bounds are deduplicated in double for every edge type
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
void mm_nek_tile(const {REAL}* __restrict__ X, const {ACC}* __restrict__ edges,
                 const long long* __restrict__ y, const long long n, const int K, const int nbins,
                 const int Ky, const int G, const {ACC} inv_n, const {ACC} h_y, const int k_y,
                 const int* __restrict__ ne_k, {ACC}* __restrict__ mi_out) {
    extern __shared__ char smem[];
    int M = nbins * Ky;
    int* sh = (int*)smem;                        // (G, M) joint histograms
    {ACC}* se = ({ACC}*)(smem + G * M * 4);    // (G, 256) cached interior edges of each column
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
        {ACC} v = X[i * (long long)K + c];
        int ne = ne_k[c];
        bool cached = ne <= 256;
        int lo = 0, hi = ne;                     // upper_bound == cp.searchsorted side='right'
        if (v != v) lo = hi;                     // NaN sorts last: the last bin, as cp.searchsorted does
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            {ACC} em = cached ? se[j * 256 + mid] : edges[(long long)mid * K + c];
            if (em <= v) lo = mid + 1; else hi = mid;
        }
        atomicAdd(&sh[j * M + lo * Ky + (int)y[i]], 1);
    }
    __syncthreads();
    if (tid < g) {
        int c = c0 + tid;
        int* t = sh + tid * M;
        {ACC} hx = 0.0, hxy = 0.0;
        int kx_occ = 0, kxy = 0;
        for (int xx = 0; xx < nbins; ++xx) {
            long long rx = 0;
            for (int yy = 0; yy < Ky; ++yy) { int nxy = t[xx * Ky + yy]; rx += nxy; if (nxy > 0) kxy++; }
            if (rx == 0) continue;
            kx_occ++;
            {ACC} px = ({ACC})rx * inv_n; hx -= px * log(px);
            for (int yy = 0; yy < Ky; ++yy) { int nxy = t[xx * Ky + yy]; if (nxy > 0) { {ACC} pxy = ({ACC})nxy * inv_n; hxy -= pxy * log(pxy); } }
        }
        {ACC} mi = hx + h_y - hxy + (({ACC})(kx_occ + k_y - kxy - 1)) * 0.5 * inv_n;
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
        with _KERNEL_CACHE_LOCK:
            kern = _MM_NEK_TILE_KERNELS.get(bool(is_f32))
            if kern is None:
                kern = cp.RawKernel(render(_MM_NEK_TILE_TEMPLATE, real=np.float32 if is_f32 else np.float64), "mm_nek_tile")
                _MM_NEK_TILE_KERNELS[bool(is_f32)] = kern
        kern(((K + G - 1) // G,), (_MM_NEK_TILE_THREADS,), (Xc, Ec, yv, np.int64(n), np.int32(K), np.int32(nbins), np.int32(Ky), np.int32(G),  # launch-ok: the tile's edge cache and per-column reduction are written for a fixed 256-lane block (the loader strides by blockDim / G)
                                                             np.float64(1.0 / float(max(1, n))), np.float64(float(h_y)), np.int32(int(k_y)), ne_k, mi_out),
             shared_mem=G * (4 * M + 256 * 8))
    except Exception as exc:
        log_throttle(
            logger, "mm_nek_tile_launch_failed", logging.WARNING,
            "mm_nek tiled launch failed (%s: %s); falling back to the one-column-per-block kernel", type(exc).__name__, exc, exc_info=True,
        )
        return False
    return True


# Per-column min and max of a row-major (n, K) matrix in ONE coalesced pass. cupy's axis-0 reductions cost 4.4 ms at
# 100000 x 100 (more than the MM kernel they feed); here a block owns 8 columns, thread t reads column t % 8 of rows
# t / 8 + k * (blockDim / 8), so each warp reads a contiguous run of a row. A NaN in a column makes its bound NaN, as
# cupy's min and max do (fmin/fmax would skip it).
_COL_MINMAX_TEMPLATE = r"""
extern "C" __global__
void col_minmax(const {REAL}* __restrict__ X, const long long n, const int K,
                {ACC}* __restrict__ cmin, {ACC}* __restrict__ cmax) {
    const int G = 8;
    __shared__ {ACC} smin[256], smax[256];
    __shared__ int snan[256];
    int c0 = blockIdx.x * G;
    int g = min(G, K - c0);
    int tid = threadIdx.x, nt = blockDim.x;
    int j = tid % G;
    int rg = threadIdx.x / G;                    // row group of the 8 columns: a warp reads whole 64-byte runs of four rows
    {ACC} lo = __longlong_as_double(0x7ff0000000000000LL), hi = __longlong_as_double(0xfff0000000000000LL);
    int nan = 0;
    if (j < g) {
        for (long long i = rg; i < n; i += nt / G) {
            {ACC} v = ({ACC})X[i * (long long)K + c0 + j];
            if (v != v) nan = 1;
            else { if (v < lo) lo = v; if (v > hi) hi = v; }
        }
    }
    smin[tid] = lo; smax[tid] = hi; snan[tid] = nan;
    __syncthreads();
    if (tid < g) {
        {ACC} a = __longlong_as_double(0x7ff0000000000000LL), b = __longlong_as_double(0xfff0000000000000LL);
        int z = 0;
        for (int t = tid; t < nt; t += G) { a = fmin(a, smin[t]); b = fmax(b, smax[t]); z |= snan[t]; }
        {ACC} nanv = __longlong_as_double(0x7ff8000000000000LL);
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
        with _KERNEL_CACHE_LOCK:
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
