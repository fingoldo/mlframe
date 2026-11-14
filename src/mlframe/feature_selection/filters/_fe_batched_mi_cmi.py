"""Batched device CMI count / entropy kernels carved out of ``_fe_batched_mi`` (carve-wave, 2026-06-28).

Pure code-movement sibling: the per-row / per-column joint-histogram + plug-in-entropy / occupied-cell
RawKernels and their lazy getters that the batched CMI / marginal-MI assembly in the parent
``_fe_batched_mi.batched_cmi_gpu`` consumes (``_rows_entropy_and_k``, the joint-hist + joint-entropy +
cmi-joint / marginal-mi entropy kernels, ``joint_counts_gpu`` / ``joint_entropy_gpu`` /
``joint_nnz_gpu`` / ``cmi_joint_entropies_gpu`` / ``marginal_mi_entropies_gpu`` / ``_cmi_assemble``).
Function and kernel-source bodies are byte-for-byte the originals - NO logic change. The parent
re-exports every public name for back-compat; importers continue to use ``from ._fe_batched_mi import X``.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from mlframe.utils.log_throttle import log_throttle
from ._kernel_types import render
from pyutilz.performance.gpu_hw_launch import block_size_for

from ._fe_batched_mi_cmi_lut import (  # noqa: F401 -- carved lookup-table entropy kernels, re-exported
    _ROWS_ENT_NNZ_LUT_SRC,
    _XLOGX_TABLES,
    _get_rows_ent_nnz_lut_kernels,
    _rows_entropy_lut,
    _rows_entropy_lut_split,
    _xlogx_table,
)
from ._fe_batched_mi_cmi_hist import (  # noqa: F401  -- carved: row entropies and joint counts
    _get_rows_ent_nnz_kernel,
    _get_rows_ent_nnz_i32_kernel,
    _get_rows_ent_nnz_split_kernels,
    _tree_block,
    _rows_entropy_and_k,
    _get_joint_hist_kernels,
    _get_batched_joint_hist_u8_kernels,
    _get_batched_joint_hist_kernels,
    _batched_joint_counts2,
    _hist_rows_per_block,
    _sm_count,
    _hist_smem_tile,
    _batched_marginal_counts,
    _XLOGX_ROWS_EK,
    _ROWS_ENT_NNZ_SRC,
    _ROWS_ENT_NNZ_KERNEL,
    _ROWS_ENT_NNZ_I32_KERNEL,
    _ROWS_ENT_NNZ_SPLIT_KERNELS,
    _JOINT_HIST_SRC,
    _JOINT_HIST_KERNELS,
    _JOINT_HIST_U8_SRC,
    _BATCHED_JOINT_HIST_U8_KERNELS,
    _BATCHED_JOINT_HIST_KERNELS,
    _HIST_SMEM_BYTES,
    _HIST_SMEM_MAX_CELLS,
    _HIST_SMEM_MAX_CELLS_U8,
    _HIST_BLOCKS_PER_SM,
    _SM_COUNT,
)

_BATCHED_JOINT_ENTROPY_SRC = render(r"""
extern "C" __global__
void batched_joint_entropy2(const {XT}* __restrict__ X, const long long* __restrict__ b,
                            const int Kx, const int Kb, const long long n, const int K, const int M, const int G,
                            const {ACC} inv_n, {ACC}* __restrict__ out_h, long long* __restrict__ out_k) {
    // One block owns G columns: a flat loop over rows x G reads X[i*K + c0 + j] with consecutive threads on consecutive j,
    // so a warp's loads are one contiguous run of the row instead of a K-element stride. Each column keeps its own
    // (Kx*Kb) histogram in shared memory; the per-column reduction keeps the original tree order, so the sums match bit for bit.
    extern __shared__ unsigned int hist[];      // (G, M)
    __shared__ {ACC} sh_h[256]; __shared__ {ACC} sh_k[256];
    int c0 = blockIdx.x * G;
    int g = min(G, K - c0);
    int tid = threadIdx.x, nt = blockDim.x;
    for (int t = tid; t < g * M; t += nt) hist[t] = 0u;
    __syncthreads();
    for (long long e = tid; e < n * g; e += nt) {
        long long i = e / g;
        int j = (int)(e - i * g);
        long long xv = X[i * (long long)K + c0 + j];
        atomicAdd(&hist[j * M + (unsigned int)(xv * Kb + b[i])], 1u);
    }
    __syncthreads();
    for (int j = 0; j < g; ++j) {
        {ACC} hloc = 0.0, kloc = 0.0;
        unsigned int* h = hist + j * M;
        for (int c = tid; c < M; c += nt) { unsigned int v = h[c]; if (v > 0u) { {ACC} p = ({ACC})v * inv_n; hloc += p * log(p); kloc += 1.0; } }
        sh_h[tid] = hloc; sh_k[tid] = kloc; __syncthreads();
        for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; } __syncthreads(); }
        if (tid == 0) { out_h[c0 + j] = -sh_h[0]; out_k[c0 + j] = (long long)(sh_k[0] + 0.5); }
        __syncthreads();
    }
}
extern "C" __global__
void batched_joint_entropy2_g(const {XT}* __restrict__ X, const long long* __restrict__ b,
                              const int Kx, const int Kb, const long long n, const int K, const int M, const int G,
                              const {ACC} inv_n, unsigned int* __restrict__ scratch, {ACC}* __restrict__ out_h, long long* __restrict__ out_k) {
    // Global-scratch twin of batched_joint_entropy2: a block walks groups of G columns; its scratch slice holds the G
    // histograms, and the reads are the same coalesced row-group pattern (consecutive threads take consecutive columns).
    unsigned int* hist = scratch + (long long)blockIdx.x * G * M;
    int tid = threadIdx.x, nt = blockDim.x;
    __shared__ {ACC} sh_h[256]; __shared__ {ACC} sh_k[256];
    int groups = (K + G - 1) / G;
    for (int cg = blockIdx.x; cg < groups; cg += gridDim.x) {
        int c0 = cg * G;
        int g = min(G, K - c0);
        for (int t = tid; t < g * M; t += nt) hist[t] = 0u;
        __syncthreads();
        int rstride = nt / g;                         // rows per pass; threads past rstride * g (a partial last group) sit out, so no row is counted twice
        int rg = threadIdx.x / g;                     // row group: consecutive threads take consecutive columns of one row
        int jl = threadIdx.x % g;
        if (rg < rstride) {
            for (long long i = rg; i < n; i += rstride) {
                long long xv = X[i * (long long)K + c0 + jl];
                atomicAdd(&hist[jl * M + (unsigned int)(xv * Kb + b[i])], 1u);
            }
        }
        __syncthreads();
        for (int j = 0; j < g; ++j) {
            {ACC} hloc = 0.0, kloc = 0.0;
            unsigned int* h = hist + j * M;
            for (int c = tid; c < M; c += nt) { unsigned int v = h[c]; if (v > 0u) { {ACC} p = ({ACC})v * inv_n; hloc += p * log(p); kloc += 1.0; } }
            sh_h[tid] = hloc; sh_k[tid] = kloc; __syncthreads();
            for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; } __syncthreads(); }
            if (tid == 0) { out_h[c0 + j] = -sh_h[0]; out_k[c0 + j] = (long long)(sh_k[0] + 0.5); }
            __syncthreads();
        }
    }
}

// coalescing-ok: kept for joints above 6144 cells, where the untiled kernel measured faster than the tiled one (tracker GPU11-54: 23.4 against 26.6 ms at 15000 cells); the tiled batched_joint_entropy2_g serves the smaller joints
extern "C" __global__
void batched_joint_entropy2_g_wide(const {XT}* __restrict__ X, const long long* __restrict__ b,
                              const int Kx, const int Kb, const long long n, const int K, const int M,
                              const {ACC} inv_n, unsigned int* __restrict__ scratch, {ACC}* __restrict__ out_h, long long* __restrict__ out_k) {
    unsigned int* hist = scratch + (long long)blockIdx.x * M;      // this block's slice, reused for every column it owns
    int tid = threadIdx.x, nt = blockDim.x;
    __shared__ {ACC} sh_h[256]; __shared__ {ACC} sh_k[256];
    for (int col = blockIdx.x; col < K; col += gridDim.x) {
        for (int i = tid; i < M; i += nt) hist[i] = 0u;
        __syncthreads();
        for (long long i = tid; i < n; i += nt) {
            long long xv = X[i * (long long)K + col];
            atomicAdd(&hist[(unsigned int)(xv * Kb + b[i])], 1u);
        }
        __syncthreads();
        {ACC} hloc = 0.0, kloc = 0.0;
        for (int c = tid; c < M; c += nt) { unsigned int v = hist[c]; if (v > 0u) { {ACC} p = ({ACC})v * inv_n; hloc += p * log(p); kloc += 1.0; } }
        sh_h[tid] = hloc; sh_k[tid] = kloc; __syncthreads();
        for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; } __syncthreads(); }
        if (tid == 0) { out_h[col] = -sh_h[0]; out_k[col] = (long long)(sh_k[0] + 0.5); }
        __syncthreads();
    }
}
""")
_BATCHED_JOINT_ENTROPY_KERNEL: dict = {}  # code type ('long long' or 'unsigned char') -> the shared-memory kernel
_BATCHED_JOINT_ENTROPY_G_KERNEL: dict = {}  # code type -> [tiled global-scratch kernel, one-column-per-block kernel]


_JOINT_ENTROPY_TILE_MAX_COLS = 8  # columns per block for both fused joint-entropy kernels; measured fastest at 8 (see the profile record)
# global-scratch route: the tiled kernel up to 6144 cells (measured faster at 5000: 11.9 vs 14.8 ms; at 8000 it ties and at 15000 it loses 26.6 vs 23.3 ms),
# eight columns per block only up to 1024 cells (measured fastest at 1000; one column per block above)
_JOINT_ENTROPY_GLOBAL_TILE_MAX_CELLS = 6144
_JOINT_ENTROPY_GLOBAL_G8_MAX_CELLS = 1024
_JOINT_ENTROPY_GLOBAL_MAX_BLOCK_MULT = 2  # global-scratch kernel: blocks per SM when the scratch fits a quarter of L2 (4 measured slower at M = 1000)


def batched_joint_entropy_global(X: Any, b: Any, Kx: int, Kb: int, inv_n: float, blocks: int) -> "tuple[Any, Any]":
    """The fused per-column joint entropy with each block's histogram in global scratch (``batched_joint_entropy2_g``), for joints over the shared-memory
    limit: ``(h (K,), k (K,))``, the fused kernel's loop and reduction with ``blocks`` blocks walking the columns."""
    import cupy as cp

    xt = _code_type(X)
    if xt not in _BATCHED_JOINT_ENTROPY_G_KERNEL:
        src = _entropy_source(xt)
        _BATCHED_JOINT_ENTROPY_G_KERNEL[xt] = [cp.RawKernel(src, "batched_joint_entropy2_g"), cp.RawKernel(src, "batched_joint_entropy2_g_wide")]
    g_kernels = _BATCHED_JOINT_ENTROPY_G_KERNEL[xt]
    n, K = int(X.shape[0]), int(X.shape[1])
    M = int(Kx) * int(Kb)
    if M > _JOINT_ENTROPY_GLOBAL_TILE_MAX_CELLS:
        # large joints: one column per block (the untiled kernel measured faster here; the column groups only pay for small histograms)
        g = max(1, min(int(blocks), K))
        scratch = cp.empty(g * M, dtype=cp.uint32)
        out_h = cp.empty(K, dtype=cp.float64)  # Intentional float64: entropies, as the fused kernel writes them
        out_k = cp.empty(K, dtype=cp.int64)
        g_threads = _tree_block(g_kernels[1], 0, n)
        g_kernels[1](
            (g,),
            (g_threads,),
            (cp.ascontiguousarray(X), b, np.int32(int(Kx)), np.int32(int(Kb)), np.int64(n), np.int32(K), np.int32(M), float(inv_n), scratch, out_h, out_k),
        )
        return out_h, out_k
    # Blocks and columns per block are sized so that all the blocks' scratch slices stay within L2: the caller admitted ``blocks`` (one per SM)
    # against that budget, so extra blocks are taken only while the total still fits. More blocks lift the parallelism of the global atomics.
    l2 = int(cp.cuda.Device().attributes.get("L2CacheSize", 0))
    if l2 <= 0:
        l2 = 1 << 20  # L2 size unreported: assume 1 MiB
    # the second block layer only while its doubled scratch stays in a quarter of L2: at M = 15000 the full-L2 rule measured 29 ms against 23 ms with one per SM
    mult = _JOINT_ENTROPY_GLOBAL_MAX_BLOCK_MULT if int(blocks) * _JOINT_ENTROPY_GLOBAL_MAX_BLOCK_MULT * M * 4 * 4 <= l2 else 1
    # column groups only for small joints: G = 8 measured fastest at M = 1000, while G > 1 at M = 5000 and 15000 was slower than one column per block
    G = _JOINT_ENTROPY_TILE_MAX_COLS if M <= _JOINT_ENTROPY_GLOBAL_G8_MAX_CELLS else 1
    groups = (K + G - 1) // G
    g = max(1, min(int(blocks) * mult, groups))
    scratch = cp.empty(g * G * M, dtype=cp.uint32)
    out_h = cp.empty(K, dtype=cp.float64)  # Intentional float64: entropies, as the fused kernel writes them
    out_k = cp.empty(K, dtype=cp.int64)
    g_threads = _tree_block(g_kernels[0], 0, n)
    g_kernels[0](
        (g,),
        (g_threads,),
        (
            cp.ascontiguousarray(X),
            b,
            np.int32(int(Kx)),
            np.int32(int(Kb)),
            np.int64(n),
            np.int32(K),
            np.int32(M),
            np.int32(G),
            float(inv_n),
            scratch,
            out_h,
            out_k,
        ),
    )
    return out_h, out_k


_BATCHED_JOINT_ENTROPY_SH_LIMIT = None


def _code_type(X) -> str:
    """The CUDA type the fused entropy kernels read the code matrix as: bytes stay bytes (an eighth of the traffic, no widening copy), everything else is 64-bit."""
    return "unsigned char" if str(X.dtype) == "uint8" else "long long"


def _entropy_source(xt: str) -> str:
    """The fused entropy kernels' source with the code type ``xt`` filled in."""
    return _BATCHED_JOINT_ENTROPY_SRC.replace("{XT}", xt)


def _get_batched_joint_entropy_kernel(cp, xt: str = "long long"):
    """Lazily compile and cache the ``batched_joint_entropy2`` RawKernel for the code type ``xt``, and probe+cache the device's max shared memory per block (used to decide when the kernel's histogram fits in shared memory vs global scratch)."""
    global _BATCHED_JOINT_ENTROPY_SH_LIMIT
    if xt not in _BATCHED_JOINT_ENTROPY_KERNEL:
        _BATCHED_JOINT_ENTROPY_KERNEL[xt] = cp.RawKernel(_entropy_source(xt), "batched_joint_entropy2")
        try:
            _BATCHED_JOINT_ENTROPY_SH_LIMIT = int(cp.cuda.Device().attributes.get("MaxSharedMemoryPerBlock", 48 * 1024))
        # best-effort: a smaller shared-memory limit only moves the dispatch to the global-memory kernel, which gives the same counts
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug("querying MaxSharedMemoryPerBlock failed, using the 48KiB default: %s", e)
            _BATCHED_JOINT_ENTROPY_SH_LIMIT = 48 * 1024
    return _BATCHED_JOINT_ENTROPY_KERNEL[xt]


def _joint_entropy_tile_cols(M: int) -> int:
    """Columns per block for ``batched_joint_entropy2``: at most 8, and G histograms of M uint32 plus the 4 KB of static reduction scratch fit the device limit."""
    limit = _BATCHED_JOINT_ENTROPY_SH_LIMIT
    if limit is None:
        limit = 48 * 1024
    return max(1, min(8, (int(limit) - 4096) // (4 * int(M))))


def _batched_joint_entropy_and_k2(X, b, Kx, Kb, inv_n):
    """Fused per-column joint entropy+nnz of (n,K) codes ``X`` with (n,) code ``b`` -> (h (K,) f64, k (K,) int64).
    ONE launch (block/column, shared hist + tree-reduce). Same occupied cells as
    ``_rows_entropy_and_k(_batched_joint_counts2(X,b,Kx,Kb).astype(f64), inv_n)`` and entropies within a few ULP (another summation order; relative <= 4.4e-16 measured). Returns None when the per-
    column hist won't fit shared memory (caller uses the unfused two-launch path) or on any kernel error.

    Default ``auto``: the fused kernel runs where it is faster, for joints of at least ``_FUSED_ENTROPY_MIN_M`` cells (per-host kernel-tuning cache over (n, M));
    ``MLFRAME_FE_GPU_FUSE_CMI_ENTROPY=1`` / ``0`` forces it on / off. 2026-11-03 (n=30000, K=930, Kx=10, GTX 1050 Ti, min of 5): unfused / fused ms at M=40 3.6 / 21.8,
    M=400 36.6 / 22.9, M=1000 59.9 / 21.7, M=4000 73.3 / 11.5, M=10000 78.7 / 15.8. The large joints of the conditional CMI (60-78 ms per unfused launch in
    the 100k fit) are the ones the June measurement below called rare."""
    # bench-attempt-rejected: count+reduce -> 1 fused launch. nsys: removed ~18-36 cuLaunchKernel/
    # fit (total launch APIs 159->135 across the combined A/B). BIT-IDENTICAL (h/k maxdiff 0; full batched_cmi
    # + return_cards maxdiff 0). REJECTED on GPU KERNEL TIME: the one-block-per-column shared-hist kernel runs at
    # a near-CONSTANT ~36ms (n=100k, K=384, GTX 1050 Ti) regardless of M - it serializes n shared-mem atomicAdds
    # per column across only K blocks x 256 threads (Pascal shared-atomic + low-occupancy bound). The unfused
    # count+reduce is GLOBAL-atomic over n*K threads (massively parallel) + a cheap reduce: 4.5ms at the joint
    # sizes the conditional CMI path actually hits (Kx*Kz, Kx*Kyz ~ 40-500 since Ky/Kz are small). Measured
    # fused/unfused: M=40 -> 8.05x SLOWER, M=100 -> 8.24x SLOWER, M=400 -> ~par, M=1000 -> 0.50x (only wins for
    # rare huge joints). Net GPU time far worse at production cardinalities despite the launch-count win; nsys
    # made batched_joint_entropy2 62% of GPU time (284ms/8) vs hist2 17% (44ms/8). Kept gated for the rare
    # large-M case + as a documented bench artifact; default OFF.
    import cupy as cp

    _mode = os.environ.get("MLFRAME_FE_GPU_FUSE_CMI_ENTROPY", "auto").strip().lower()
    if _mode in ("0", "false", "no", "off"):
        return None
    try:
        n, K = int(X.shape[0]), int(X.shape[1])
        M = int(Kx) * int(Kb)
        from ._fe_cmi_fused_backend import fused_entropy_backend

        if _mode not in ("1", "true", "yes", "on") and fused_entropy_backend(n, M) != "fused":
            return None
        if X.dtype != cp.uint8 and X.dtype != cp.int64:
            X = X.astype(cp.int64)  # a byte matrix is read as bytes by the fused kernels; any other dtype is widened to 64-bit
        ker = _get_batched_joint_entropy_kernel(cp, _code_type(X))  # also probes the shared-memory limit for the tile width below
        assert _BATCHED_JOINT_ENTROPY_SH_LIMIT is not None  # populated by _get_batched_joint_entropy_kernel above
        # shared = M uint32 hist + 2*256 f64 reduction scratch (the sh_h/sh_k arrays are static, not counted here);
        # gate the dynamic hist against the device per-block limit (leave headroom for the static doubles).
        if M <= 0:
            return None
        if M * 4 + 4096 > int(_BATCHED_JOINT_ENTROPY_SH_LIMIT):
            # over shared memory: one block per SM with its histogram in global scratch, while all the slices fit L2 together (n=30000, K=930:
            # M=15000 77.9 -> 24.0 ms, M=30000 79.6 -> 47.8 ms; M=100000 (2.4 MB of slices) 86.1 -> 125.4 ms, so count+reduce keeps those)
            attrs = cp.cuda.Device().attributes
            sms, l2 = int(attrs.get("MultiProcessorCount", 1)), int(attrs.get("L2CacheSize", 0))
            if sms * M * 4 > l2:
                return None
            return batched_joint_entropy_global(X, b, Kx, Kb, inv_n, sms)
        Xc = cp.ascontiguousarray(X)
        # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
        out_h = cp.empty(K, dtype=cp.float64)
        out_k = cp.empty(K, dtype=cp.int64)
        G = _joint_entropy_tile_cols(M)
        ker(  # launch-ok: the per-column tree reduction runs over static 256-entry scratch (sh_h, sh_k), so the block is fixed at 256 threads
            ((K + G - 1) // G,),
            (256,),
            (Xc, b, np.int32(int(Kx)), np.int32(int(Kb)), np.int64(n), np.int32(K), np.int32(M), np.int32(G), float(inv_n), out_h, out_k),
            shared_mem=G * M * 4,
        )
        return out_h, out_k
    except Exception:
        import logging

        logging.getLogger(__name__).debug("fused batched joint entropy failed; count+reduce fallback", exc_info=True)
        return None


# FUSED 1D scalar entropy + occupied-cell (launch-reduction). The column-invariant H(z)/H(y,z)/
# H(y) terms in batched_cmi_gpu were cp.bincount (scan+cub) -> boolean-mask getitem -> *inv_n -> log -> mul
# -> sum (~6-7 cuLaunchKernel) plus a separate (c>0).sum() nnz. Computing the histogram with the in-kernel
# atomicAdd joint_counts_gpu (one launch, no bincount) and reducing it with this grid-stride kernel (one
# launch, out is cp.zeros(2)=memset) collapses each scalar term to two launches. Same float64 plug-in
# entropy / occupied-cell -> selection-equivalent.
_ENT_NNZ_1D_SRC = render(r"""
extern "C" __global__
void ent_nnz_1d(const long long* __restrict__ c, const {ACC} inv_n, const long long M,
                {ACC}* __restrict__ out) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    {ACC} hloc = 0.0, kloc = 0.0;
    for (long long i = t; i < M; i += stride) {
        long long ci = c[i];
        if (ci > 0) { {ACC} p = ({ACC})ci * inv_n; hloc += p * log(p); kloc += 1.0; }
    }
    __shared__ {ACC} sh_h[256];
    __shared__ {ACC} sh_k[256];
    int tid = threadIdx.x;
    sh_h[tid] = hloc; sh_k[tid] = kloc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { atomicAdd(&out[0], sh_h[0]); atomicAdd(&out[1], sh_k[0]); }
}
""")
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
        # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
        out = cp.zeros(2, dtype=cp.float64)  # cudaMemsetAsync, not a cuLaunchKernel
        threads = _tree_block(_ENT_NNZ_1D_KERNEL, 0, M)
        blocks = min(1024, max(1, (M + threads - 1) // threads))
        _ENT_NNZ_1D_KERNEL((blocks,), (threads,), (c, float(inv_n), np.int64(M), out))
        h_k = cp.asnumpy(out)
        return float(-h_k[0]), round(h_k[1])
    except Exception as e:
        log_throttle(
            logging.getLogger(__name__),
            "batched_mi_cmi.nnz_1d",
            logging.WARNING,
            "batched CMI: the fused nnz-1d entropy kernel failed (%s), using the elementwise path",
            e,
        )
        # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
        cf = c.astype(cp.float64) if c.dtype != cp.float64 else c
        p = cf[cf > 0] * float(inv_n)
        return float(-(p * cp.log(p)).sum()), int((cf > 0).sum())


def joint_counts_gpu(codes: Any, cards: Any) -> Any:
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
    h1, h2, h3 = _get_joint_hist_kernels()
    threads = block_size_for(h1 if len(codes) == 1 else h2 if len(codes) == 2 else h3, work_items=n)
    blocks = (n + threads - 1) // threads
    if len(codes) == 1:
        h1((blocks,), (threads,), (codes[0], np.int64(n), counts))
    elif len(codes) == 2:
        h2((blocks,), (threads,), (codes[0], codes[1], np.int32(int(cards[1])), np.int64(n), counts))
    else:
        h3((blocks,), (threads,), (codes[0], codes[1], codes[2], np.int32(int(cards[1])), np.int32(int(cards[2])), np.int64(n), counts))
    return counts


# FUSED joint histogram + plug-in entropy in ONE launch (launch-reduction). The per-candidate
# CMI path computed each term as joint_counts_gpu (atomicAdd hist) + _ent_from_counts (entropy reduce) = TWO
# launches. When the joint cell count M = prod(cards) fits in shared memory, ONE block builds the histogram in
# SHARED via atomicAdd, then reduces the plug-in entropy (-sum p log p) + occupied-cell count in the same
# launch. out[0] = sum xlogx (h = -out[0]), out[1] = nnz. Same float64 plug-in entropy / occupied-cell
# definition -> selection-equivalent (exact integer counts; reduction-order ~1e-14). Falls back to the
# two-kernel path (joint_counts_gpu + ent_nnz_1d) when M is over the shared-memory budget.
_JOINT_ENTROPY_SRC = render(r"""
extern "C" __global__
void joint_entropy1(const long long* __restrict__ a, const long long n, const int M,
                    const {ACC} inv_n, {ACC}* __restrict__ out) {
    extern __shared__ unsigned int hist[];
    int tid = threadIdx.x, nt = blockDim.x;
    for (int i = tid; i < M; i += nt) hist[i] = 0u;
    __syncthreads();
    for (long long i = tid; i < n; i += nt) atomicAdd(&hist[a[i]], 1u);
    __syncthreads();
    {ACC} hloc = 0.0, kloc = 0.0;
    for (int c = tid; c < M; c += nt) { unsigned int v = hist[c]; if (v > 0u) { {ACC} p = ({ACC})v * inv_n; hloc += p * log(p); kloc += 1.0; } }
    __shared__ {ACC} sh_h[256]; __shared__ {ACC} sh_k[256];
    sh_h[tid] = hloc; sh_k[tid] = kloc; __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; } __syncthreads(); }
    if (tid == 0) { out[0] = sh_h[0]; out[1] = sh_k[0]; }
}
extern "C" __global__
void joint_entropy2(const long long* __restrict__ a, const long long* __restrict__ b, const int Kb,
                    const long long n, const int M, const {ACC} inv_n, {ACC}* __restrict__ out) {
    extern __shared__ unsigned int hist[];
    int tid = threadIdx.x, nt = blockDim.x;
    for (int i = tid; i < M; i += nt) hist[i] = 0u;
    __syncthreads();
    for (long long i = tid; i < n; i += nt) atomicAdd(&hist[a[i] * Kb + b[i]], 1u);
    __syncthreads();
    {ACC} hloc = 0.0, kloc = 0.0;
    for (int c = tid; c < M; c += nt) { unsigned int v = hist[c]; if (v > 0u) { {ACC} p = ({ACC})v * inv_n; hloc += p * log(p); kloc += 1.0; } }
    __shared__ {ACC} sh_h[256]; __shared__ {ACC} sh_k[256];
    sh_h[tid] = hloc; sh_k[tid] = kloc; __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; } __syncthreads(); }
    if (tid == 0) { out[0] = sh_h[0]; out[1] = sh_k[0]; }
}
extern "C" __global__
void joint_entropy3(const long long* __restrict__ a, const long long* __restrict__ b,
                    const long long* __restrict__ c, const int Kb, const int Kc, const long long n,
                    const int M, const {ACC} inv_n, {ACC}* __restrict__ out) {
    extern __shared__ unsigned int hist[];
    int tid = threadIdx.x, nt = blockDim.x;
    for (int i = tid; i < M; i += nt) hist[i] = 0u;
    __syncthreads();
    for (long long i = tid; i < n; i += nt) atomicAdd(&hist[(a[i] * Kb + b[i]) * Kc + c[i]], 1u);
    __syncthreads();
    {ACC} hloc = 0.0, kloc = 0.0;
    for (int cc = tid; cc < M; cc += nt) { unsigned int v = hist[cc]; if (v > 0u) { {ACC} p = ({ACC})v * inv_n; hloc += p * log(p); kloc += 1.0; } }
    __shared__ {ACC} sh_h[256]; __shared__ {ACC} sh_k[256];
    sh_h[tid] = hloc; sh_k[tid] = kloc; __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; } __syncthreads(); }
    if (tid == 0) { out[0] = sh_h[0]; out[1] = sh_k[0]; }
}
""")
_JOINT_ENTROPY_KERNELS = None
_JOINT_ENTROPY_SH_LIMIT = None


def _get_joint_entropy_kernels():
    """Lazily compile and cache the single-pair (1-, 2-, 3-way) joint-entropy RawKernels, and probe+cache the device's max shared memory per block for the fused-histogram fast path."""
    global _JOINT_ENTROPY_KERNELS, _JOINT_ENTROPY_SH_LIMIT
    if _JOINT_ENTROPY_KERNELS is None:
        import cupy as cp

        mod = cp.RawModule(code=_JOINT_ENTROPY_SRC)
        _JOINT_ENTROPY_KERNELS = (mod.get_function("joint_entropy1"), mod.get_function("joint_entropy2"), mod.get_function("joint_entropy3"))
        try:
            _JOINT_ENTROPY_SH_LIMIT = int(cp.cuda.Device().attributes.get("MaxSharedMemoryPerBlock", 48 * 1024))
        # best-effort: a smaller shared-memory limit only moves the dispatch to the global-memory kernel, which gives the same counts
        except Exception as e:
            import logging

            logging.getLogger(__name__).debug("querying MaxSharedMemoryPerBlock failed, using the 48KiB default: %s", e)
            _JOINT_ENTROPY_SH_LIMIT = 48 * 1024
    return _JOINT_ENTROPY_KERNELS


def joint_entropy_gpu(codes: Any, cards: Any, inv_n: float) -> tuple[float, int]:
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
    assert _JOINT_ENTROPY_SH_LIMIT is not None  # populated by _get_joint_entropy_kernels above
    # static shared (sh_h+sh_k = 256*8*2 = 4096 B) + dynamic hist (M*4 B) must clear the per-block limit.
    if M * 4 + 4096 <= _JOINT_ENTROPY_SH_LIMIT:
        try:
            n = int(codes[0].size)
            # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
            out = cp.zeros(2, dtype=cp.float64)
            shmem = M * 4
            tree_threads = _tree_block(kers[min(len(codes), 3) - 1], shmem, n)
            if len(codes) == 1:
                kers[0]((1,), (tree_threads,), (codes[0], np.int64(n), np.int32(M), float(inv_n), out), shared_mem=shmem)
            elif len(codes) == 2:
                kers[1]((1,), (tree_threads,), (codes[0], codes[1], np.int32(int(cards[1])), np.int64(n), np.int32(M), float(inv_n), out), shared_mem=shmem)
            else:
                kers[2](
                    (1,),
                    (tree_threads,),
                    (codes[0], codes[1], codes[2], np.int32(int(cards[1])), np.int32(int(cards[2])), np.int64(n), np.int32(M), float(inv_n), out),
                    shared_mem=shmem,
                )
            hk = cp.asnumpy(out)
            return float(-hk[0]), round(hk[1])
        except Exception as e:  # nosec B110 - best-effort/optional path, no module logger
            import logging

            logging.getLogger(__name__).debug("fused nnz-1d entropy path failed, falling back to joint_counts_gpu: %s", e)
    return _ent_nnz_1d(joint_counts_gpu(codes, cards), inv_n)  # type: ignore[no-any-return]


# FUSED four-joint conditional-CMI entropies in ONE launch (launch-reduction). The conditional
# CMI(x;y|z) needs the plug-in entropy + occupied-cell count of FOUR joints - (z), (x,z), (y,z), (x,y,z) -
# which _cmi_from_binned_cupy computed as four separate joint_entropy_gpu launches (the #1 cuLaunchKernel
# source on the F2 STRICT redundancy gate after the analytic-null card reuse removed joint_cardinalities).
# This kernel runs all four as FOUR BLOCKS of one grid (block b builds joint b's histogram from the shared
# x/y/z code arrays into its own dynamic-shared hist, then the same tree-reduction over the block's threads
# emits (entropy, nnz) for that joint). All threads in a block share blockIdx -> no branch divergence. The
# dynamic shared is sized to the largest joint (x,y,z); engages only when that fits the per-block limit (the
# SAME condition under which joint_entropy_gpu took its fast single-launch path for x,y,z), else the caller
# uses the per-joint path. Same f64 plug-in entropy, same occupied-cell (c>0) definition, same cell-index
# reduction order as joint_entropy1/2/3 -> the assembled CMI is bit-identical.
_CMI_JOINT_ENTROPIES_SRC = render(r"""
extern "C" __global__
void cmi_joint_entropies(const long long* __restrict__ x, const long long* __restrict__ y,
                         const long long* __restrict__ z, const int Kx, const int Ky, const int Kz,
                         const long long n, const {ACC} inv_n, {ACC}* __restrict__ out) {
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
    {ACC} hloc = 0.0, kloc = 0.0;
    for (int c = tid; c < M; c += nt) { unsigned int v = hist[c]; if (v > 0u) { {ACC} p = ({ACC})v * inv_n; hloc += p * log(p); kloc += 1.0; } }
    __shared__ {ACC} sh_h[256]; __shared__ {ACC} sh_k[256];
    sh_h[tid] = hloc; sh_k[tid] = kloc; __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; } __syncthreads(); }
    if (tid == 0) { out[blk * 2] = -sh_h[0]; out[blk * 2 + 1] = sh_k[0]; }
}
""")
_CMI_JOINT_ENTROPIES_KERNEL = None


def cmi_joint_entropies_gpu(dx: Any, dy: Any, dz: Any, Kx: int, ky: int, kz: int, inv_n: float) -> Any:
    """The four conditional-CMI joint (plug-in entropy, occupied-cell count) terms - (z), (x,z), (y,z),
    (x,y,z) - in ONE launch. Returns ((h_z,k_z),(h_xz,k_xz),(h_yz,k_yz),(h_xyz,k_xyz)), or None when the
    largest (x,y,z) joint won't fit the per-block shared limit (caller falls back to four joint_entropy_gpu
    launches). Bit-identical to four separate joint_entropy_gpu calls."""
    import cupy as cp

    global _CMI_JOINT_ENTROPIES_KERNEL
    _get_joint_entropy_kernels()  # ensures _JOINT_ENTROPY_SH_LIMIT is populated
    assert _JOINT_ENTROPY_SH_LIMIT is not None
    Mmax = int(Kx) * int(ky) * int(kz)
    if Mmax <= 0 or Mmax * 4 + 4096 > int(_JOINT_ENTROPY_SH_LIMIT):
        return None
    try:
        if _CMI_JOINT_ENTROPIES_KERNEL is None:
            _CMI_JOINT_ENTROPIES_KERNEL = cp.RawKernel(_CMI_JOINT_ENTROPIES_SRC, "cmi_joint_entropies")
        n = int(dz.size)
        # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
        out = cp.zeros(8, dtype=cp.float64)
        _CMI_JOINT_ENTROPIES_KERNEL(
            (4,),
            (_tree_block(_CMI_JOINT_ENTROPIES_KERNEL, Mmax * 4, n),),
            (dx, dy, dz, np.int32(int(Kx)), np.int32(int(ky)), np.int32(int(kz)), np.int64(n), float(inv_n), out),
            shared_mem=Mmax * 4,
        )
        hk = cp.asnumpy(out)
        return ((float(hk[0]), round(hk[1])), (float(hk[2]), round(hk[3])), (float(hk[4]), round(hk[5])), (float(hk[6]), round(hk[7])))
    except Exception:
        import logging

        logging.getLogger(__name__).debug("cmi_joint_entropies_gpu fused kernel failed; caller falls back to four joint_entropy_gpu launches", exc_info=True)
        return None


# bench-attempt-rejected: a two-block fused (x,z)+(x,y,z) kernel for the FIXED-yz greedy CMI path
# (cmi_from_binned_fixed_yz_cupy, where H(z)/H(y,z) are precomputed so only xz/xyz remain) was BIT-IDENTICAL
# to the two separate joint_entropy_gpu calls on random codes (10/10 A/B) yet FLIPPED the full-MRMR selection
# on test_gpu_cpu_mi_selection_equivalence[reg_two_pairs] and [adv_wide_p60] - a real divergence the random
# A/B did not surface (interaction with the precomputed-yz terms at a card combo). The four-block conditional
# fusion (cmi_joint_entropies_gpu) is SAFE there (all four joints come from the one kernel -> self-consistent;
# full suite green); the fixed-yz two-block split mixing fused xz/xyz with precomputed h_z/h_yz is not. Kept
# the fixed-yz path on the two per-joint joint_entropy_gpu launches.


# FUSED three-joint marginal-MI entropies in ONE launch - the marginal MI(x;y) path (the seed anchor and the
# redundancy gate's per-raw marginal anchor) needs H(x), H(y), H(x,y), three separate joint_entropy_gpu
# launches per call. Three blocks of one grid (block 0: x, M=Kx; block 1: y, M=Ky; block 2: xy, M=Kx*Ky),
# each the same per-block histogram + tree reduction. All three terms come from THIS kernel (the marginal MI
# combines only its own h_x/h_y/h_xy, no precomputed cross-path term), so it is self-consistent like the
# four-block conditional fusion - not the rejected two-block fixed-yz split. Engages when the (x,y) joint
# fits the per-block shared limit (it is tiny: Kx*Ky). Same f64 plug-in entropy, occupied-cell definition,
# and cell-index reduction order -> bit-identical.
_MARGINAL_MI_ENTROPIES_SRC = render(r"""
extern "C" __global__
void marginal_mi_entropies(const long long* __restrict__ x, const long long* __restrict__ y,
                           const int Kx, const int Ky, const long long n, const {ACC} inv_n,
                           {ACC}* __restrict__ out) {
    extern __shared__ unsigned int hist[];
    int blk = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    int M = (blk == 0) ? Kx : (blk == 1) ? Ky : (Kx * Ky);
    for (int i = tid; i < M; i += nt) hist[i] = 0u;
    __syncthreads();
    for (long long i = tid; i < n; i += nt) {
        // width-ok: idx is a histogram cell below M = Kx * Ky, which the caller bounds by the per-block shared-memory limit
        int idx = (blk == 0) ? (int)x[i] : (blk == 1) ? (int)y[i] : (int)(x[i] * Ky + y[i]);
        atomicAdd(&hist[idx], 1u);
    }
    __syncthreads();
    {ACC} hloc = 0.0, kloc = 0.0;
    for (int c = tid; c < M; c += nt) { unsigned int v = hist[c]; if (v > 0u) { {ACC} p = ({ACC})v * inv_n; hloc += p * log(p); kloc += 1.0; } }
    __shared__ {ACC} sh_h[256]; __shared__ {ACC} sh_k[256];
    sh_h[tid] = hloc; sh_k[tid] = kloc; __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) { if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; } __syncthreads(); }
    if (tid == 0) { out[blk * 2] = -sh_h[0]; out[blk * 2 + 1] = sh_k[0]; }
}
""")
_MARGINAL_MI_ENTROPIES_KERNEL = None


def marginal_mi_entropies_gpu(dx: Any, dy: Any, Kx: int, ky: int, inv_n: float) -> Any:
    """The three marginal-MI joint terms - (x), (y), (x,y) - in ONE launch. Returns
    ((h_x,k_x),(h_y,k_y),(h_xy,k_xy)), or None when the (x,y) joint won't fit the per-block shared limit
    (caller falls back to three joint_entropy_gpu launches). Bit-identical to the three separate calls."""
    import cupy as cp

    global _MARGINAL_MI_ENTROPIES_KERNEL
    _get_joint_entropy_kernels()  # ensures _JOINT_ENTROPY_SH_LIMIT is populated
    assert _JOINT_ENTROPY_SH_LIMIT is not None
    Mmax = int(Kx) * int(ky)
    if Mmax <= 0 or Mmax * 4 + 4096 > int(_JOINT_ENTROPY_SH_LIMIT):
        return None
    try:
        if _MARGINAL_MI_ENTROPIES_KERNEL is None:
            _MARGINAL_MI_ENTROPIES_KERNEL = cp.RawKernel(_MARGINAL_MI_ENTROPIES_SRC, "marginal_mi_entropies")
        n = int(dx.size)
        # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
        out = cp.zeros(6, dtype=cp.float64)
        _MARGINAL_MI_ENTROPIES_KERNEL(
            (3,),
            (_tree_block(_MARGINAL_MI_ENTROPIES_KERNEL, Mmax * 4, n),),
            (dx, dy, np.int32(int(Kx)), np.int32(int(ky)), np.int64(n), float(inv_n), out),
            shared_mem=Mmax * 4,
        )
        hk = cp.asnumpy(out)
        return ((float(hk[0]), round(hk[1])), (float(hk[2]), round(hk[3])), (float(hk[4]), round(hk[5])))
    except Exception:
        import logging

        logging.getLogger(__name__).debug("marginal_mi_entropies_gpu fused kernel failed; caller falls back to three joint_entropy_gpu launches", exc_info=True)
        return None


# FUSED final CMI/MI assembly (launch-reduction). batched_cmi_gpu assembled the per-column result
# with a cupy chain over (K,) arrays: cmi = h_xz + h_yz - h_z - h_xyz; bias = (k_xyz + k_z - k_xz - k_yz)/2n;
# max(cmi - bias, 0) - ~7 launches. Both the conditional and marginal forms reduce to
#   out[i] = max( he1[i] - he2[i] + hc - (kc1[i] - kc2[i] + kc) * inv2n , 0 )
# so one (K,) kernel emits the result. Same f64 ops -> bit-identical.
_CMI_ASSEMBLE_SRC = render(r"""
extern "C" __global__
void cmi_assemble(const {ACC}* __restrict__ he1, const {ACC}* __restrict__ he2, const {ACC} hc,
                  const long long* __restrict__ kc1, const long long* __restrict__ kc2, const {ACC} kc,
                  const {ACC} inv2n, const int K, {ACC}* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= K) return;
    {ACC} v = (he1[i] - he2[i] + hc) - (({ACC})(kc1[i] - kc2[i]) + kc) * inv2n;
    out[i] = v > 0.0 ? v : 0.0;
}
""")
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
        # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
        out = cp.empty(K, dtype=cp.float64)
        threads = block_size_for(_CMI_ASSEMBLE_KERNEL, work_items=K)
        _CMI_ASSEMBLE_KERNEL(
            ((K + threads - 1) // threads,),
            (threads,),
            # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
            (
                cp.ascontiguousarray(he1.astype(cp.float64, copy=False)),
                # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
                cp.ascontiguousarray(he2.astype(cp.float64, copy=False)),
                float(hc),
                cp.ascontiguousarray(kc1.astype(cp.int64, copy=False)),
                cp.ascontiguousarray(kc2.astype(cp.int64, copy=False)),
                float(kc),
                float(inv2n),
                np.int32(K),
                out,
            ),
        )
        return out
    except Exception:
        import logging

        logging.getLogger(__name__).debug("_cmi_assemble RawKernel failed; falling back to the cupy chain", exc_info=True)
        return cp.maximum((he1 - he2 + hc) - ((kc1 - kc2) + kc) * inv2n, 0.0)


_JOINT_NNZ_KERNELS = None


def _get_joint_nnz_kernels():
    """Lazily compile and cache the joint-histogram RawKernels' nnz-only entry points, reusing the same source module as ``_get_joint_hist_kernels``."""
    global _JOINT_NNZ_KERNELS
    if _JOINT_NNZ_KERNELS is None:
        import cupy as cp

        mod = cp.RawModule(code=_JOINT_HIST_SRC)
        _JOINT_NNZ_KERNELS = (mod.get_function("joint_hist_nnz1"), mod.get_function("joint_hist_nnz2"), mod.get_function("joint_hist_nnz3"))
    return _JOINT_NNZ_KERNELS


def joint_nnz_gpu(codes: Any, cards: Any) -> int:
    """Occupied-cell COUNT (cardinality) of the joint of 1-3 device code arrays in ONE launch - the
    histogram atomicAdd and the nonzero count fuse via the atomicAdd-returns-old 0->1 trick, so no separate
    cp.bincount / _nnz reduction. ``codes`` cupy int64 (n,); ``cards`` the matching cardinalities. Returns a
    Python int. The count is exact (integer) -> identical to ``_nnz_from_counts(joint_counts_gpu(...))``."""
    import cupy as cp

    n = int(codes[0].size)
    M = 1
    for kc in cards:
        M *= int(kc)
    counts = cp.zeros(int(max(M, 1)), dtype=cp.int64)  # cudaMemsetAsync, not a cuLaunchKernel
    nnz = cp.zeros(1, dtype=cp.uint64)  # cudaMemsetAsync, not a cuLaunchKernel
    h1, h2, h3 = _get_joint_nnz_kernels()
    threads = block_size_for(h1 if len(codes) == 1 else h2 if len(codes) == 2 else h3, work_items=n)
    blocks = (n + threads - 1) // threads
    if len(codes) == 1:
        h1((blocks,), (threads,), (codes[0], np.int64(n), counts, nnz))
    elif len(codes) == 2:
        h2((blocks,), (threads,), (codes[0], codes[1], np.int32(int(cards[1])), np.int64(n), counts, nnz))
    else:
        h3((blocks,), (threads,), (codes[0], codes[1], codes[2], np.int32(int(cards[1])), np.int32(int(cards[2])), np.int64(n), counts, nnz))
    return int(nnz.get()[0])
