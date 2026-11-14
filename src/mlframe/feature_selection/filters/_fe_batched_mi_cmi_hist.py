"""Row entropies and joint counts of the batched device CMI: the fused per-row entropy + occupied-cell kernels (``_rows_entropy_and_k``), the atomic and shared-memory joint-histogram kernels in
int64 and uint8 code types, ``_batched_joint_counts2`` and ``_batched_marginal_counts``.

Carved out of ``_fe_batched_mi_cmi`` (which re-exports every name here) to keep that module under the line budget.
"""

from __future__ import annotations

import logging

import numpy as np
from pyutilz.performance.gpu_hw_launch import block_size_for

from mlframe.utils.log_throttle import log_throttle

from ._kernel_types import render

# bench-attempt-rejected: fit-amortised reuse of the joint-histogram atomicAdd count buffers in
# joint_counts_gpu (227 calls/fit, ~22M int64) and _batched_joint_counts2 (15 calls, up to ~320M int64) - one
# grow-to-max int64 buffer per site, .fill(0) the live [:M] slice instead of cp.zeros (alloc+memset) - to kill
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


# FUSED per-row entropy + occupied-cell in ONE RawKernel (launch-reduction). The prior path was
# _XLOGX_ROWS_EK (1) + .sum(axis=1) (1) + (counts>0).sum(axis=1) (1) = 3 cuLaunchKernel per call; with 2-4
# calls per batched_cmi_gpu it was the residual #1 launch source. One block per row (candidate) grid-strides
# its M cells accumulating (sum xlogx, nnz), a shared-mem tree reduction folds the block, and thread 0 writes
# out_h[row] (negated) + out_k[row]. ONE launch; out_h/out_k are cp.empty (cudaMalloc, not a cuLaunchKernel).
# One-block-per-row -> deterministic tree-order sum (no cross-block atomics); same float64 plug-in entropy /
# occupied-cell definition -> selection-equivalent.
_ROWS_ENT_NNZ_SRC = render(r"""
extern "C" __global__
void rows_ent_nnz(const {ACC}* __restrict__ counts, const {ACC} inv_n, const int K, const long long M,
                  {ACC}* __restrict__ out_h, long long* __restrict__ out_k) {
    int row = blockIdx.x;
    if (row >= K) return;
    const {ACC}* cr = counts + (long long)row * M;
    int tid = threadIdx.x;
    {ACC} hloc = 0.0, kloc = 0.0;
    for (long long i = tid; i < M; i += blockDim.x) {
        {ACC} ci = cr[i];
        if (ci > 0.0) { {ACC} p = ci * inv_n; hloc += p * log(p); kloc += 1.0; }
    }
    __shared__ {ACC} sh_h[256];
    __shared__ {ACC} sh_k[256];
    sh_h[tid] = hloc; sh_k[tid] = kloc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { out_h[row] = -sh_h[0]; out_k[row] = (long long)(sh_k[0] + 0.5); }
}
// INT32-counts variant (bandwidth fix, 2026-07-02): nsys at the 1M fit put rows_ent_nnz at 9.35s (34% of
// GPU time) reading {ACC} precision cells of a huge mostly-zero counts matrix, PLUS 2.2s of cupy_copy for the
// int64->{ACC} precision .astype the callers did first. Counts are exact small ints (<= n < 2^31): read them as
// int32 (4B/cell, no astype copy at all) and cast in-register - identical entropy/nnz values, ~6x less
// memory traffic (astype read8+write8 + reduce read8 -> reduce read4).
extern "C" __global__
void rows_ent_nnz_i32(const int* __restrict__ counts, const {ACC} inv_n, const int K, const long long M,
                      {ACC}* __restrict__ out_h, long long* __restrict__ out_k) {
    int row = blockIdx.x;
    if (row >= K) return;
    const int* cr = counts + (long long)row * M;
    int tid = threadIdx.x;
    {ACC} hloc = 0.0, kloc = 0.0;
    for (long long i = tid; i < M; i += blockDim.x) {
        int ci = cr[i];
        if (ci > 0) { {ACC} p = ({ACC})ci * inv_n; hloc += p * log(p); kloc += 1.0; }
    }
    __shared__ {ACC} sh_h[256];
    __shared__ {ACC} sh_k[256];
    sh_h[tid] = hloc; sh_k[tid] = kloc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { out_h[row] = -sh_h[0]; out_k[row] = (long long)(sh_k[0] + 0.5); }
}
// SPLIT-K variant (SM-occupancy fix, 2026-07-02): one-block-per-row leaves the GPU nearly EMPTY when the
// perm chunk is narrow (K = 2-8 rows -> 2-8 blocks on a 6-SM card with ~48 block slots). Split each row's M
// cells across gridDim.y segment blocks; each block writes its partial (sum, nnz) to a (K, S) scratch, and
// the tiny finish kernel folds the S partials per row. Fixed S -> deterministic summation order (partials
// folded in segment order) -> same ~1e-15 fp class as any tree-order change: selection-equivalent.
extern "C" __global__
void rows_ent_nnz_i32_split(const int* __restrict__ counts, const {ACC} inv_n, const int K,
                            const long long M, const int S,
                            {ACC}* __restrict__ part_h, {ACC}* __restrict__ part_k) {
    int row = blockIdx.x;
    int seg = blockIdx.y;
    if (row >= K || seg >= S) return;
    long long seg_len = (M + S - 1) / S;
    long long lo = (long long)seg * seg_len;
    long long hi = lo + seg_len; if (hi > M) hi = M;
    const int* cr = counts + (long long)row * M;
    int tid = threadIdx.x;
    {ACC} hloc = 0.0, kloc = 0.0;
    for (long long i = lo + tid; i < hi; i += blockDim.x) {
        int ci = cr[i];
        if (ci > 0) { {ACC} p = ({ACC})ci * inv_n; hloc += p * log(p); kloc += 1.0; }
    }
    __shared__ {ACC} sh_h[256];
    __shared__ {ACC} sh_k[256];
    sh_h[tid] = hloc; sh_k[tid] = kloc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) { sh_h[tid] += sh_h[tid + s]; sh_k[tid] += sh_k[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) { part_h[row * S + seg] = sh_h[0]; part_k[row * S + seg] = sh_k[0]; }
}
// coalescing-ok: a finishing pass: one thread per row sums its S block partials part_h[row * S + s], with S the number of row chunks of the counting pass; K * S values are read once against the n * K counting pass
extern "C" __global__
void rows_ent_nnz_finish(const {ACC}* __restrict__ part_h, const {ACC}* __restrict__ part_k,
                         const int K, const int S,
                         {ACC}* __restrict__ out_h, long long* __restrict__ out_k) {
    int row = (int)((long long)blockIdx.x * blockDim.x + threadIdx.x);
    if (row >= K) return;
    {ACC} h = 0.0, k = 0.0;
    for (int s = 0; s < S; ++s) { h += part_h[row * S + s]; k += part_k[row * S + s]; }
    out_h[row] = -h; out_k[row] = (long long)(k + 0.5);
}
""")
_ROWS_ENT_NNZ_KERNEL = None
_ROWS_ENT_NNZ_I32_KERNEL = None
_ROWS_ENT_NNZ_SPLIT_KERNELS = None


def _get_rows_ent_nnz_kernel(cp):
    """Lazily compile (once per process) and cache the double precision ``rows_ent_nnz`` RawKernel; NVCC compilation is a ~30ms one-off, so callers pay it only on the first invocation."""
    global _ROWS_ENT_NNZ_KERNEL
    if _ROWS_ENT_NNZ_KERNEL is None:
        _ROWS_ENT_NNZ_KERNEL = cp.RawKernel(_ROWS_ENT_NNZ_SRC, "rows_ent_nnz")
    return _ROWS_ENT_NNZ_KERNEL


def _get_rows_ent_nnz_i32_kernel(cp):
    """Lazily compile and cache the int32-count variant ``rows_ent_nnz_i32`` RawKernel (same source module, different entry point)."""
    global _ROWS_ENT_NNZ_I32_KERNEL
    if _ROWS_ENT_NNZ_I32_KERNEL is None:
        _ROWS_ENT_NNZ_I32_KERNEL = cp.RawKernel(_ROWS_ENT_NNZ_SRC, "rows_ent_nnz_i32")
    return _ROWS_ENT_NNZ_I32_KERNEL


def _get_rows_ent_nnz_split_kernels(cp):
    """Lazily compile and cache the two-stage split/finish kernel pair (partial-sum + reduction) used when the row width exceeds what a single-block reduction can hold."""
    global _ROWS_ENT_NNZ_SPLIT_KERNELS
    if _ROWS_ENT_NNZ_SPLIT_KERNELS is None:
        mod = cp.RawModule(code=_ROWS_ENT_NNZ_SRC)
        _ROWS_ENT_NNZ_SPLIT_KERNELS = (mod.get_function("rows_ent_nnz_i32_split"), mod.get_function("rows_ent_nnz_finish"))
    return _ROWS_ENT_NNZ_SPLIT_KERNELS


from ._fe_batched_mi_cmi_lut import (  # noqa: F401 -- carved lookup-table entropy kernels, re-exported
    _ROWS_ENT_NNZ_LUT_SRC,
    _XLOGX_TABLES,
    _get_rows_ent_nnz_lut_kernels,
    _rows_entropy_lut,
    _rows_entropy_lut_split,
    _xlogx_table,
)


def _tree_block(kernel, dyn_smem: int, work: int) -> int:
    """Threads per block for a kernel whose block reduction halves ``blockDim.x`` down to 1 over 256-wide static scratch: a power of two, at most 256, derived from the device."""
    return block_size_for(kernel, dyn_smem=int(dyn_smem), work_items=int(work), pow2=True, max_width=256)


def _rows_entropy_and_k(counts, inv_n):
    """Per-row (per-candidate) plug-in entropy + occupied-cell count from a (K, M) device count matrix.

    Fused into ONE RawKernel (one block per row, shared-mem reduction) -> a single cuLaunchKernel replacing
    the elementwise + two axis sums. ``counts`` is double precision (K, M). Returns (h (K,) double precision, k (K,) int64),
    selection-equivalent to the prior path (deterministic per-row tree-order sum). Falls back to the
    ElementwiseKernel + axis sums on any kernel error."""
    import cupy as cp

    global _XLOGX_ROWS_EK
    try:
        # int32 counts: fused in-register cast (rows_ent_nnz_i32) - no float64 astype copy, 4B/cell reads.
        if counts.dtype == cp.int32:
            c = cp.ascontiguousarray(counts)
            _ker = _get_rows_ent_nnz_i32_kernel(cp)
        else:
            # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
            c = cp.ascontiguousarray(counts if counts.dtype == cp.float64 else counts.astype(cp.float64))
            _ker = _get_rows_ent_nnz_kernel(cp)
        K = int(c.shape[0])
        M = int(c.shape[1]) if c.ndim > 1 else int(c.size)
        # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
        out_h = cp.empty(K, dtype=cp.float64)
        out_k = cp.empty(K, dtype=cp.int64)
        # SPLIT-K when the one-block-per-row grid cannot fill the SMs (narrow perm chunk, huge M): a 6-SM card
        # runs ~48 resident blocks, so K < ~48 with a multi-million-cell M leaves most SMs idle for the whole
        # 15-30ms reduction. Split each row's M cells across S segment blocks (K*S ~ a few hundred blocks) +
        # a tiny finish fold. Only for the int32 huge-M perm-null case; small-M / large-K keeps one launch.
        if counts.dtype == cp.int32 and K < 48 and M >= (1 << 20):
            S = max(2, min(64, (48 + K - 1) // max(1, K)))
            # the lookup-table form first (c * ln(c) from a table instead of a double precision log per cell: 5 to 11x on this card at 8 to 26 rows of 2M to 8M cells, entropy within 3e-14,
            # occupied-cell counts identical); the double precision split kernel stays for a row count that is not a whole number or MLFRAME_ENT_LUT=0
            _lut_split = _rows_entropy_lut_split(cp, c, inv_n, S)
            if _lut_split is not None:
                return _lut_split
            _split, _finish = _get_rows_ent_nnz_split_kernels(cp)
            threads = _tree_block(_split, 0, M)
            # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
            part_h = cp.empty(K * S, dtype=cp.float64)
            # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
            part_k = cp.empty(K * S, dtype=cp.float64)
            _split((K, S), (threads,), (c, float(inv_n), np.int32(K), np.int64(M), np.int32(S), part_h, part_k))
            finish_threads = block_size_for(_finish, work_items=K)
            _finish(((K + finish_threads - 1) // finish_threads,), (finish_threads,), (part_h, part_k, np.int32(K), np.int32(S), out_h, out_k))
            return out_h, out_k
        if counts.dtype == cp.int32:
            _lut = _rows_entropy_lut(cp, c, inv_n)
            if _lut is not None:
                return _lut
        threads = _tree_block(_ker, 0, M)
        _ker((K,), (threads,), (c, float(inv_n), np.int32(K), np.int64(M), out_h, out_k))
        return out_h, out_k
    except Exception as e:
        log_throttle(
            logging.getLogger(__name__),
            "batched_mi_cmi.xlogx_rows",
            logging.WARNING,
            "batched CMI: the fused xlogx-rows kernel failed (%s), using the elementwise kernel",
            e,
        )
        if _XLOGX_ROWS_EK is None:
            # Intentional float64: njit/CUDA kernel dtype contract (signatures and device buffers are float64 by design); kernel refactor deferred.
            _XLOGX_ROWS_EK = cp.ElementwiseKernel("T c, float64 invn", "float64 o", "o = c > 0 ? (c * invn) * log(c * invn) : 0.0", "mrmr_xlogx_rows_ek")
        h = -_XLOGX_ROWS_EK(counts, float(inv_n)).sum(axis=1)
        k = cp.sum(counts > 0, axis=1)
        return h, k


# FUSED joint-histogram RawKernels (launch-reduction). The per-call cupy CMI built each joint
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
// INT32-counts twins (bandwidth fix, 2026-07-02): the counts are exact cell frequencies <= n < 2^31, so
// int32 atomics halve both the memset (cp.zeros) and the entropy-reduce traffic of the huge mostly-zero
// (K, Kx*Kb) buffer, and HALVE its VRAM -> the perm-null fits 2x the perms per chunk. Identical counts.
extern "C" __global__
void batched_joint_hist2_i32(const long long* __restrict__ X, const long long* __restrict__ b,
                             const int Kx, const int Kb, const long long n, const int K,
                             int* __restrict__ counts) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (t >= total) return;
    int col = (int)(t % (long long)K);
    long long row = t / (long long)K;
    long long xv = X[row * (long long)K + col];
    atomicAdd(&counts[(col * (long long)Kx + xv) * Kb + b[row]], 1);
}
// SHARED-MEMORY twin of batched_joint_hist2_i32: a block owns a tile of columns and a chunk of rows, counts into a shared (tile, Kx*Kb) histogram with shared
// atomics (a few cycles, no contention across blocks) and adds its non-zero cells to the global counts once at the end. The global-atomics kernel issues one
// contended global atomic per (row, column): n * K of them on a (K, Kx*Kb) buffer whose cells are few and hot. Same counts.
extern "C" __global__
void batched_joint_hist2_i32_smem(const long long* __restrict__ X, const long long* __restrict__ b,
                                  const int Kx, const int Kb, const long long n, const int K,
                                  const int tile, const long long rows_per_block, int* __restrict__ counts) {
    extern __shared__ int hist[];
    const int cells = Kx * Kb;
    const int c0 = blockIdx.y * tile;
    const int tc = min(tile, K - c0);
    for (int i = threadIdx.x; i < tc * cells; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    const long long r0 = (long long)blockIdx.x * rows_per_block;
    const long long r1 = min(r0 + rows_per_block, n);
    const long long span = (r1 - r0) * (long long)tc;
    for (long long t = threadIdx.x; t < span; t += blockDim.x) {
        const int c = (int)(t % tc);
        const long long row = r0 + t / tc;
        atomicAdd(&hist[c * cells + (int)X[row * (long long)K + c0 + c] * Kb + (int)b[row]], 1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < tc * cells; i += blockDim.x) {
        const int v = hist[i];
        if (v) atomicAdd(&counts[(long long)c0 * cells + i], v);
    }
}
extern "C" __global__
void batched_joint_hist1_i32(const long long* __restrict__ X, const int Kx, const long long n, const int K,
                             int* __restrict__ counts) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (t >= total) return;
    int col = (int)(t % (long long)K);
    long long row = t / (long long)K;
    long long xv = X[row * (long long)K + col];
    atomicAdd(&counts[col * (long long)Kx + xv], 1);
}
// OCCUPIED-CELL count in the SAME pass as the histogram: atomicAdd returns the OLD value, so a cell's
// 0 -> 1 transition (first sample landing there) is detected race-free and bumps a single global nnz
// counter. ONE launch yields the cardinality joint_cardinalities_cupy needs - no separate _nnz reduction.
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
    """Lazily compile and cache the single-pair (1-, 2-, 3-way) joint-histogram RawKernels from the shared source module."""
    global _JOINT_HIST_KERNELS
    if _JOINT_HIST_KERNELS is None:
        import cupy as cp

        mod = cp.RawModule(code=_JOINT_HIST_SRC)
        _JOINT_HIST_KERNELS = (mod.get_function("joint_hist1"), mod.get_function("joint_hist2"), mod.get_function("joint_hist3"))
    return _JOINT_HIST_KERNELS


# BYTE-CODE twins of the batched joint-histogram kernels: the same counts from a uint8 candidate matrix (codes < 256), one eighth of the int64 matrix's read traffic. The CMIM scorer
# calls the CMI about 600 times per fit on the same candidate block, so the block is kept as bytes and these kernels read it as such.
_JOINT_HIST_U8_SRC = r"""
extern "C" __global__
void batched_joint_hist2_i32_u8(const unsigned char* __restrict__ X, const long long* __restrict__ b,
                                const int Kx, const int Kb, const long long n, const int K,
                                int* __restrict__ counts) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (t >= total) return;
    int col = (int)(t % (long long)K);
    long long row = t / (long long)K;
    int xv = X[row * (long long)K + col];
    atomicAdd(&counts[((long long)col * Kx + xv) * Kb + b[row]], 1);
}
extern "C" __global__
void batched_joint_hist2_i32_smem_u8(const unsigned char* __restrict__ X, const long long* __restrict__ b,
                                     const int Kx, const int Kb, const long long n, const int K,
                                     const int tile, const long long rows_per_block, int* __restrict__ counts) {
    extern __shared__ int hist[];
    const int cells = Kx * Kb;
    const int c0 = blockIdx.y * tile;
    const int tc = min(tile, K - c0);
    for (int i = threadIdx.x; i < tc * cells; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    const long long r0 = (long long)blockIdx.x * rows_per_block;
    const long long r1 = min(r0 + rows_per_block, n);
    const long long span = (r1 - r0) * (long long)tc;
    for (long long t = threadIdx.x; t < span; t += blockDim.x) {
        const int c = (int)(t % tc);
        const long long row = r0 + t / tc;
        atomicAdd(&hist[c * cells + (int)X[row * (long long)K + c0 + c] * Kb + (int)b[row]], 1);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < tc * cells; i += blockDim.x) {
        const int v = hist[i];
        if (v) atomicAdd(&counts[(long long)c0 * cells + i], v);
    }
}
// DIVISION-FREE layout of the shared-histogram kernel: a block is (tile columns) x (blockDim.x / tile row lanes); thread (lane, c) takes rows lane, lane + R, ... of the block's row range
// and counts column c, so the 64-bit ``t % tc`` and ``t / tc`` per element of the version above are gone and a warp reads ``tile`` consecutive bytes of one row (a full 32-byte sector for
// tile = 32). ``tile`` must be a power of two (``tile_shift`` = log2).
extern "C" __global__
void batched_joint_hist2_i32_smem_u8_v2(const unsigned char* __restrict__ X, const long long* __restrict__ b,
                                        const int Kx, const int Kb, const long long n, const int K,
                                        const int tile, const int tile_shift, const long long rows_per_block, int* __restrict__ counts) {
    extern __shared__ int hist[];
    const int cells = Kx * Kb;
    const int c0 = blockIdx.y * tile;
    const int tc = min(tile, K - c0);
    for (int i = threadIdx.x; i < tc * cells; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    const int c = threadIdx.x & (tile - 1);
    const int lane = threadIdx.x >> tile_shift;
    const int lanes = blockDim.x >> tile_shift;
    const long long r0 = (long long)blockIdx.x * rows_per_block;
    const long long r1 = min(r0 + rows_per_block, n);
    if (c < tc) {
        const unsigned char* col = X + c0 + c;
        int* h = hist + c * cells;
        for (long long row = r0 + lane; row < r1; row += lanes) {
            // Warp-coalesced despite the per-row stride: within one warp every thread shares the same `lane` (hence the same `row`
            // at this iteration) and differs only in `c` (the low bits of threadIdx.x), so the warp reads `tile` CONSECUTIVE bytes of
            // this one row, not one byte each `K` apart. Named outside the subscript so it reads as the single scalar it is.
            const long long row_off = row * (long long)K;
            atomicAdd(&h[(int)col[row_off] * Kb + (int)b[row]], 1);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < tc * cells; i += blockDim.x) {
        const int v = hist[i];
        if (v) atomicAdd(&counts[(long long)c0 * cells + i], v);
    }
}
extern "C" __global__
void batched_joint_hist1_i32_u8(const unsigned char* __restrict__ X, const int Kx, const long long n, const int K,
                                int* __restrict__ counts) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = n * (long long)K;
    if (t >= total) return;
    int col = (int)(t % (long long)K);
    long long row = t / (long long)K;
    int xv = X[row * (long long)K + col];
    atomicAdd(&counts[(long long)col * Kx + xv], 1);
}
"""
_BATCHED_JOINT_HIST_U8_KERNELS = None


def _get_batched_joint_hist_u8_kernels():
    """Lazily compile and cache the byte-code batched joint-histogram RawKernels ``(hist1, hist2, hist2_smem)``."""
    global _BATCHED_JOINT_HIST_U8_KERNELS
    if _BATCHED_JOINT_HIST_U8_KERNELS is None:
        import cupy as cp

        mod = cp.RawModule(code=_JOINT_HIST_U8_SRC)
        _BATCHED_JOINT_HIST_U8_KERNELS = (
            mod.get_function("batched_joint_hist1_i32_u8"),
            mod.get_function("batched_joint_hist2_i32_u8"),
            mod.get_function("batched_joint_hist2_i32_smem_u8"),
            mod.get_function("batched_joint_hist2_i32_smem_u8_v2"),
        )
    return _BATCHED_JOINT_HIST_U8_KERNELS


_BATCHED_JOINT_HIST_KERNELS = None


def _get_batched_joint_hist_kernels():
    """Lazily compile and cache the batched (int64 + int32-count) joint-histogram RawKernels used when many candidate columns are histogrammed against one target in a single launch."""
    global _BATCHED_JOINT_HIST_KERNELS
    if _BATCHED_JOINT_HIST_KERNELS is None:
        import cupy as cp

        mod = cp.RawModule(code=_JOINT_HIST_SRC)
        _BATCHED_JOINT_HIST_KERNELS = (
            mod.get_function("batched_joint_hist1"),
            mod.get_function("batched_joint_hist2"),
            mod.get_function("batched_joint_hist1_i32"),
            mod.get_function("batched_joint_hist2_i32"),
            mod.get_function("batched_joint_hist2_i32_smem"),
        )
    return _BATCHED_JOINT_HIST_KERNELS


def _batched_joint_counts2(X, b, Kx, Kb):
    """Per-column joint counts of (n,K) int codes ``X`` with a single (n,) code ``b`` -> (K, Kx*Kb) int64,
    via ONE in-kernel-flat-key atomicAdd launch. Identical counts to
    ``cp.bincount((X*Kb + b[:,None] + col_off*(Kx*Kb)).ravel(), minlength=K*Kx*Kb).reshape(K, Kx*Kb)``."""
    import cupy as cp

    n, K = int(X.shape[0]), int(X.shape[1])
    # bench-attempt-rejected: fit-amortised reuse buffer here grows to ~2.5GB (up to 320M int64 at
    # 1M rows) and stays pinned all fit -> on a 4GB GTX 1050 Ti it starves the resident mempool and REGRESSES
    # wall (~122s -> ~132s). The .fill(0) of 2.5GB also costs ~the same as cp.zeros' memset, so the only saving
    # was the (warm-pool-cheap) malloc. Keep cp.zeros so this large buffer is returned to the pool after use.
    # int32 counts: exact frequencies <= n < 2^31; halves the memset + the
    # entropy-reduce traffic + the buffer VRAM (2x perms per perm-null chunk). Identical counts.
    counts = cp.zeros(int(K) * int(Kx) * int(Kb), dtype=cp.int32)
    u8 = X.dtype == cp.uint8
    tile = _hist_smem_tile(int(Kx) * int(Kb), K, u8)
    if tile:
        # one block per (row chunk, column tile): shared histograms, one global add per occupied cell per block
        col_tiles = (K + tile - 1) // tile
        rows_per_block = _hist_rows_per_block(n, col_tiles)
        grid = ((n + rows_per_block - 1) // rows_per_block, col_tiles)
        shared = tile * int(Kx) * int(Kb) * 4
        # the lane layout is blockDim.x / tile lanes over the tile's columns, so any warp multiple works; the width is the device's occupancy answer for the kernel (256 on the GTX 1050 Ti)
        if u8:
            tile_kernel = _get_batched_joint_hist_u8_kernels()[3]
            tile_kernel(
                grid,
                (block_size_for(tile_kernel, dyn_smem=shared, work_items=rows_per_block * tile),),
                (
                    X,
                    b,
                    np.int32(int(Kx)),
                    np.int32(int(Kb)),
                    np.int64(n),
                    np.int32(K),
                    np.int32(tile),
                    np.int32(tile.bit_length() - 1),
                    np.int64(rows_per_block),
                    counts,
                ),
                shared_mem=shared,
            )
            return counts.reshape(K, int(Kx) * int(Kb))
        tile_kernel = _get_batched_joint_hist_kernels()[4]
        tile_kernel(
            grid,
            (block_size_for(tile_kernel, dyn_smem=shared, work_items=rows_per_block * tile),),
            (X, b, np.int32(int(Kx)), np.int32(int(Kb)), np.int64(n), np.int32(K), np.int32(tile), np.int64(rows_per_block), counts),
            shared_mem=shared,
        )
        return counts.reshape(K, int(Kx) * int(Kb))
    h2 = _get_batched_joint_hist_u8_kernels()[1] if u8 else _get_batched_joint_hist_kernels()[3]
    threads = block_size_for(h2, work_items=n * K)
    blocks = (n * K + threads - 1) // threads
    h2((blocks,), (threads,), (X, b, np.int32(int(Kx)), np.int32(int(Kb)), np.int64(n), np.int32(K), counts))
    return counts.reshape(K, int(Kx) * int(Kb))


_HIST_SMEM_BYTES = 40 * 1024  # shared bytes one block may spend on its histograms (the 48 KB static limit with room for the launch)
# Measured on the dev GPU (GTX Pascal, 100000-300000 rows, 20-1944 columns, shared against global atomics): 20 cells x1.5, 40 cells x1.3, 100 cells x1.1, 288 cells x0.7, 2400 cells x0.7:
# with many cells the global atomics are spread over the buffer and already cheap, and the shared histogram's zeroing and flush dominate. Counts are equal in every case.
_HIST_SMEM_MAX_CELLS = 128


_HIST_SMEM_MAX_CELLS_U8 = (
    256  # the byte-code kernel reads an eighth of the traffic, so shared histograms win up to 256 cells (200 cells, 30000 x 1000: 9.3 -> 7.8 ms per CMI call)
)


def _hist_rows_per_block(n: int, col_tiles: int) -> int:
    """Rows one block of the shared-memory joint histogram counts: enough blocks to fill the device, so a narrow matrix is not left to a handful of blocks.

    The fixed 16384 rows per block gave a grid of (7, 2) = 14 blocks at 100000 rows x 25 columns on a 6-SM card: achieved occupancy 0.25, 75% of the issue stalls on memory dependency, 9 GB/s of DRAM
    (nvprof). The target is ``_HIST_BLOCKS_PER_SM`` blocks per SM; each block adds one flush of ``tile * cells`` global atomics, small against its 1024 or more rows per column.
    """
    sms = _sm_count()
    target = max(1, _HIST_BLOCKS_PER_SM * sms)
    per_chunk = max(1, (target + col_tiles - 1) // col_tiles)
    rows = -(-int(n) // per_chunk)
    rows = ((rows + 255) // 256) * 256
    return int(max(1024, min(int(n), 16384, rows)))


_HIST_BLOCKS_PER_SM = 8
_SM_COUNT: list = []


def _sm_count() -> int:
    """Multiprocessors of the current device (cached; 16 when it cannot be read)."""
    if not _SM_COUNT:
        try:
            import cupy as cp

            _SM_COUNT.append(int(cp.cuda.Device().attributes["MultiProcessorCount"]))
        except Exception:
            _SM_COUNT.append(16)
    return int(_SM_COUNT[0])


def _hist_smem_tile(cells: int, n_cols: int, u8: bool = False) -> int:
    """Columns per block of the shared-memory joint histogram: the largest of 32/16/8/4/2/1 whose ``tile * cells`` int32 histogram fits ``_HIST_SMEM_BYTES`` (and
    does not exceed the column count), or 0 when one column does not fit or ``MLFRAME_HIST_SMEM=0`` or ``cells`` exceeds ``_HIST_SMEM_MAX_CELLS``."""
    import os

    if os.environ.get("MLFRAME_HIST_SMEM", "1").strip() == "0" or cells > (_HIST_SMEM_MAX_CELLS_U8 if u8 else _HIST_SMEM_MAX_CELLS):
        return 0
    for tile in (32, 16, 8, 4, 2, 1):
        if tile <= max(1, n_cols) and tile * cells * 4 <= _HIST_SMEM_BYTES:
            return tile
    return 0


def _batched_marginal_counts(X, Kx):
    """Per-column marginal counts of (n,K) int codes ``X`` -> (K, Kx) int64, via ONE atomicAdd launch.
    Identical counts to ``cp.bincount((X + col_off*Kx).ravel(), minlength=K*Kx).reshape(K, Kx)``."""
    import cupy as cp

    n, K = int(X.shape[0]), int(X.shape[1])
    # int32 counts: same bandwidth fix as _batched_joint_counts2 (identical counts, half the traffic).
    counts = cp.zeros(int(K) * int(Kx), dtype=cp.int32)
    h1 = _get_batched_joint_hist_u8_kernels()[0] if X.dtype == cp.uint8 else _get_batched_joint_hist_kernels()[2]
    threads = block_size_for(h1, work_items=n * K)
    blocks = (n * K + threads - 1) // threads
    h1((blocks,), (threads,), (X, np.int32(int(Kx)), np.int64(n), np.int32(K), counts))
    return counts.reshape(K, int(Kx))


# LAUNCH-FUSION: per-column joint ENTROPY+NNZ in ONE launch - the per-column analogue of the
# single-joint ``joint_entropy2``. The conditional CMI path fired _batched_joint_counts2 (atomicAdd launch ->
# global (K, Kx*Kb) f64 counts) THEN _rows_entropy_and_k (a second launch reducing that matrix) for BOTH the
# (x,z) and (x,y*z) joints (4 launches + 2 large f64 intermediates / fit). This kernel runs ONE BLOCK PER
# COLUMN: build that column's (Kx*Kb) histogram in SHARED uint, then fused shared tree-reduce -> (h_k, k_k).
# Counts are identical (same flat key (col,x,b)->x*Kb+b): the same nnz, and entropies within a few ULP of the count-then-reduce path (the cells are summed in
# another order; relative <= 4.4e-16 measured). Over the shared-memory limit ``batched_joint_entropy2_g`` keeps the same loop and reduction with each
# block's histogram in a global scratch slice (G x M instead of the K x M count buffer), reused column after column so it stays in L2.
