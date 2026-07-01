"""Batched device CMI count / entropy kernels carved out of ``_fe_batched_mi`` (carve-wave, 2026-06-28).

Pure code-movement sibling: the per-row / per-column joint-histogram + plug-in-entropy / occupied-cell
RawKernels and their lazy getters that the batched CMI / marginal-MI assembly in the parent
``_fe_batched_mi.batched_cmi_gpu`` consumes (``_rows_entropy_and_k``, the joint-hist + joint-entropy +
cmi-joint / marginal-mi entropy kernels, ``joint_counts_gpu`` / ``joint_entropy_gpu`` /
``joint_nnz_gpu`` / ``cmi_joint_entropies_gpu`` / ``marginal_mi_entropies_gpu`` / ``_cmi_assemble``).
Function and kernel-source bodies are byte-for-byte the originals -- NO logic change. The parent
re-exports every public name for back-compat; importers continue to use ``from ._fe_batched_mi import X``.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np


# bench-attempt-rejected (2026-06-28): fit-amortised reuse of the joint-histogram atomicAdd count buffers in
# joint_counts_gpu (227 calls/fit, ~22M int64) and _batched_joint_counts2 (15 calls, up to ~320M int64) -- one
# grow-to-max int64 buffer per site, .fill(0) the live [:M] slice instead of cp.zeros (alloc+memset) -- to kill
# the 669 cp.zeros that cProfile reported as the #1 tottime (~29-34s) on the GPU-strict-resident F2 1M/200k fit.
# Bit-identical counts (4/1 F2 selection-equiv flag-on AND flag-off; gpu_cpu_mi_selection_equivalence + joint-hist
# identity all green). REJECTED on WALL: the 34s was a cProfile async-CUDA artifact (cp.zeros' memset is attributed
# tottime but the cost is the implicit sync draining queued kernels; eliminating it just moves the drain). A
# SYNCHRONIZED micro-bench gives the real per-call delta: ~2.7ms at 22M (227x -> ~0.6s, lost in +-2s wall noise)
# and ~221ms at 320M (15x -> ~3.3s). But the 320M reuse pins ~2.5GB for the whole fit, and on this 4GB GTX 1050 Ti
# that starves the resident mempool and REGRESSES wall (clean master 126.6/127.2s -> reuse 127.4/136.6s). The
# 22M-only variant is also within noise (127.4/129.4s). Net: no measurable wall win, and the only path with real
# savings (320M reuse) regresses. Kept cp.zeros so the large buffers return to the pool after each use. NEVER
# free_all_blocks the shared pool.
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
    # bench-attempt-rejected (2026-06-28): fit-amortised reuse buffer here grows to ~2.5GB (up to 320M int64 at
    # 1M rows) and stays pinned all fit -> on a 4GB GTX 1050 Ti it starves the resident mempool and REGRESSES
    # wall (~122s -> ~132s). The .fill(0) of 2.5GB also costs ~the same as cp.zeros' memset, so the only saving
    # was the (warm-pool-cheap) malloc. Keep cp.zeros so this large buffer is returned to the pool after use.
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


def cmi_joint_entropies_gpu(dx: Any, dy: Any, dz: Any, Kx: int, ky: int, kz: int, inv_n: float) -> Any:
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


def marginal_mi_entropies_gpu(dx: Any, dy: Any, Kx: int, ky: int, inv_n: float) -> Any:
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


def joint_nnz_gpu(codes: Any, cards: Any) -> int:
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
