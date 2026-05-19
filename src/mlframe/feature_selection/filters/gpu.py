"""GPU permutation testing via CuPy raw kernels.

Public entry: ``mi_direct_gpu``. The CUDA kernels live as module attributes (``compute_joint_hist_cuda``, ``compute_mi_from_classes_cuda``, plus the batched
variants) populated by ``init_kernels()``. Lazy initialisation goes through ``_ensure_kernels_inited()`` so the first call from any thread or process gets a
consistent module state without importing CuPy at package import time.

The kernel-init lock is a ``multiprocessing.Lock`` (not ``threading.Lock``) so Windows ``spawn`` workers don't all initialise the kernels concurrently. CuPy
seeding is guarded by ``ImportError`` rather than ``NameError``.

The kernels are written into the module's ``__dict__`` via ``sys.modules[__name__]`` so callers can refer to them as plain free names at call time even though
they don't exist at module-import time.
"""
from __future__ import annotations

import logging
import math
import multiprocessing
import sys
from typing import Any

import numpy as np

from ._internals import GPU_MAX_BLOCK_SIZE
from .info_theory import compute_mi_from_classes, merge_vars

logger = logging.getLogger(__name__)

# Module-level placeholders. ``_ensure_kernels_inited`` populates them.
compute_joint_hist_cuda: Any = None
compute_mi_from_classes_cuda: Any = None
compute_joint_hist_batched_cuda: Any = None
compute_joint_hist_batched_shared_cuda: Any = None


# Persistent device-buffer pool. Reuses a preallocated set of CuPy arrays across ``mi_direct_gpu`` calls so the inner permutation loop does not pay the H2D-copy
# + alloc cost on every iteration. Buffers grow monotonically (never shrink) so repeated calls with similar shapes hit the cached allocation.
class _GpuBufferPool:
    def __init__(self):
        self.classes_x = None        # CuPy int32 vector, length >= n
        self.classes_y = None        # CuPy int32 vector, length >= n
        self.freqs_x = None          # CuPy float64 vector, length >= nbins_x
        self.freqs_y = None          # CuPy float64 vector, length >= nbins_y
        self.joint_counts = None     # CuPy int32, shape >= (nbins_x, nbins_y)
        self.totals = None           # CuPy float64 length 1
        self.cap_n = 0
        self.cap_nbins_x = 0
        self.cap_nbins_y = 0

    def ensure(self, n: int, nbins_x: int, nbins_y: int):
        import cupy as cp
        # Grow each buffer monotonically to the max ever requested.
        if self.cap_n < n:
            self.classes_x = cp.empty(n, dtype=cp.int32)
            self.classes_y = cp.empty(n, dtype=cp.int32)
            self.cap_n = n
        if self.cap_nbins_x < nbins_x:
            self.freqs_x = cp.empty(nbins_x, dtype=cp.float64)
            self.cap_nbins_x = nbins_x
        if self.cap_nbins_y < nbins_y:
            self.freqs_y = cp.empty(nbins_y, dtype=cp.float64)
            self.cap_nbins_y = nbins_y
        if (self.joint_counts is None
                or self.joint_counts.shape[0] < nbins_x
                or self.joint_counts.shape[1] < nbins_y):
            self.joint_counts = cp.empty((nbins_x, nbins_y), dtype=cp.int32)
        if self.totals is None:
            self.totals = cp.zeros(1, dtype=cp.float64)


_GPU_POOL = _GpuBufferPool()

# Cross-process safe lock. Constructed on first import; safe under multiprocessing because the lock is a primitive type that picks up the host process's mutex on spawn.
_KERNEL_INIT_LOCK = multiprocessing.Lock()


def init_kernels() -> None:
    """Build the CuPy ``RawKernel`` objects and attach them to this module's namespace. Idempotent under the lock.

    Kernels:
    * ``compute_joint_hist_cuda`` -- single-permutation joint histogram (used by ``mi_direct_gpu``).
    * ``compute_mi_from_classes_cuda`` -- single-permutation MI from joint histogram.
    * ``compute_joint_hist_batched_cuda`` -- builds joint histograms for ``batch`` permutations of ``classes_y`` in a single launch. ``perms_y`` has shape
      ``(batch, n)``; ``joint_counts_batch`` has shape ``(batch, nbins_x * nbins_y)``. Cuts kernel-launch overhead from ``npermutations`` to 1.
    * ``compute_joint_hist_multi_pair_cuda`` -- builds joint histograms for ``n_pairs`` (a, b) column pairs in a single launch via a 3D grid (rows along X, pairs
      along Y).
    """
    import cupy as cp

    module = sys.modules[__name__]

    # Multi-GPU: pin the process to the best available device BEFORE building
    # any RawKernel. CuPy compiles RawKernels per-device; binding here once
    # makes subsequent kernel launches free of device-switch overhead.
    # ``select_best_gpu`` returns the device id with the best
    # ``free_vram * compute_capability`` score (per pyutilz "auto" strategy).
    # Falls back to device 0 (the CUDA default) if pyutilz / GPU probe is
    # unavailable, preserving the legacy behaviour.
    try:
        from pyutilz.system.gpu_dispatch import select_best_gpu
        _best = select_best_gpu(strategy="auto")
        if _best is not None:
            global _BEST_DEVICE_ID  # noqa: PLW0603 - module-level state by design
            _BEST_DEVICE_ID = int(_best)
            if int(_best) != cp.cuda.Device().id:
                cp.cuda.Device(int(_best)).use()
                logger.info("mlframe.filters.gpu: pinned to CUDA device %d", _best)
    except Exception as _exc:
        logger.debug(
            "mlframe.filters.gpu: select_best_gpu unavailable (%s); "
            "using current CUDA device", _exc,
        )

    module.compute_joint_hist_cuda = cp.RawKernel(
        r"""
    extern "C" __global__
    void compute_joint_hist_cuda(const int *classes_x, const int *classes_y, int *joint_counts, int n, int nbins_y) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid<n){
            atomicAdd(&joint_counts[classes_x[tid]*nbins_y+classes_y[tid]],1);
        }
    }
    """,
        "compute_joint_hist_cuda",
    )

    # Joint-hist batched, global-atomic variant. Win axis: large joint_size
    # (>> shared-mem budget). Each thread directly atomically increments the
    # global-memory bin; correctness-simple but global atomics are 10-100x
    # slower than shared-memory atomics on cc 6.x. Kept available for
    # back-compat + fall-back when joint_size exceeds the shared-mem budget.
    module.compute_joint_hist_batched_cuda = cp.RawKernel(
        r"""
    extern "C" __global__
    void compute_joint_hist_batched_cuda(
        const int *classes_x,           // (n,)
        const int *perms_y,             // (batch, n) row-major
        int *joint_counts_batch,        // (batch, nbins_x * nbins_y) row-major
        int n,
        int nbins_x,
        int nbins_y
    ) {
        int batch_id = blockIdx.y;
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < n) {
            int cy = perms_y[batch_id * n + tid];
            int cx = classes_x[tid];
            atomicAdd(&joint_counts_batch[batch_id * (nbins_x * nbins_y) + cx * nbins_y + cy], 1);
        }
    }
    """,
        "compute_joint_hist_batched_cuda",
    )

    # Joint-hist batched, SHARED-MEM atomic variant. Win axis: small joint_size
    # (<= shared-mem budget, typical for MRMR nbins_x*nbins_y in 4-256 range).
    # Each block keeps a local histogram in shared memory (10-100x faster
    # atomic-add than global memory on cc 6.x); after the strided read of
    # n_samples the block reduces its shared hist into the global output once.
    # Shared-mem usage: joint_size * 4 bytes -- caller passes the size as
    # ``shared_mem`` launch arg (see ``mi_direct_gpu_batched`` dispatch).
    # See ``feedback_keep_all_kernel_versions``: ``compute_joint_hist_batched_cuda``
    # (the global-atomic version above) stays available; the dispatcher picks
    # this faster path only when joint_size fits.
    module.compute_joint_hist_batched_shared_cuda = cp.RawKernel(
        r"""
    extern "C" __global__
    void compute_joint_hist_batched_shared_cuda(
        const int *classes_x,           // (n,)
        const int *perms_y,             // (batch, n) row-major
        int *joint_counts_batch,        // (batch, nbins_x * nbins_y) row-major
        int n,
        int nbins_x,
        int nbins_y
    ) {
        extern __shared__ int sm_hist[];  // size: nbins_x * nbins_y * sizeof(int)
        int batch_id = blockIdx.y;
        int joint_size = nbins_x * nbins_y;
        int tid = threadIdx.x;
        int nthreads = blockDim.x;
        int gid_offset = blockIdx.x * nthreads;

        // Zero the shared histogram.
        for (int i = tid; i < joint_size; i += nthreads) {
            sm_hist[i] = 0;
        }
        __syncthreads();

        // Grid-strided read of n_samples (only the block's slice).
        int my_row = gid_offset + tid;
        if (my_row < n) {
            int cy = perms_y[batch_id * n + my_row];
            int cx = classes_x[my_row];
            atomicAdd(&sm_hist[cx * nbins_y + cy], 1);
        }
        __syncthreads();

        // Reduce: each thread merges its slice of sm_hist into the per-batch
        // global output. Block-level atomicAdd into global is still atomic
        // (many blocks may have this batch_id), so we use atomicAdd here too.
        for (int i = tid; i < joint_size; i += nthreads) {
            atomicAdd(&joint_counts_batch[batch_id * joint_size + i], sm_hist[i]);
        }
    }
    """,
        "compute_joint_hist_batched_shared_cuda",
    )
    # Multi-pair batched joint histogram kernel. Processes B pairs per launch via 3D grid (blocks along rows, blocks along pairs). Each (pair, row) thread atomically
    # adds into the pair-local joint hist.
    #   factors_data_T: (n_cols, n) row-major int32 -- transposed for coalesced reads when iterating over rows of a column.
    #   pairs_a, pairs_b: (n_pairs,) column indices into factors_data_T.
    #   nbins_a, nbins_b: (n_pairs,) per-pair cardinalities.
    #   joint_offsets: (n_pairs + 1,) prefix-sum into joint_counts_flat.
    #   joint_counts_flat: (sum(nbins_a*nbins_b),) flat output buffer.
    #   n_rows, n_pairs: grid dims.
    module.compute_joint_hist_multi_pair_cuda = cp.RawKernel(
        r"""
    extern "C" __global__
    void compute_joint_hist_multi_pair_cuda(
        const int *factors_data_T,    // (n_cols, n)
        const int *classes_y,          // (n,)  -- already merged target
        const int *pairs_a,            // (n_pairs,)
        const int *pairs_b,            // (n_pairs,)
        const int *nbins_a,            // (n_pairs,)
        const int *joint_offsets,      // (n_pairs + 1,) prefix sum into joint_counts_flat
        int *joint_counts_flat,        // sum(nbins_a[p]*nbins_b[p]*nbins_y)
        int n_rows,
        int n_pairs,
        int nbins_y
    ) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        int pid = blockIdx.y;
        if (pid >= n_pairs) return;
        if (row >= n_rows) return;

        int col_a = pairs_a[pid];
        int col_b = pairs_b[pid];
        int nba = nbins_a[pid];
        int va = factors_data_T[col_a * n_rows + row];
        int vb = factors_data_T[col_b * n_rows + row];
        int cy = classes_y[row];
        // Merged (a, b) code = va + vb * nba. Joint cell = merged * nbins_y + cy.
        int merged = va + vb * nba;
        atomicAdd(
            &joint_counts_flat[joint_offsets[pid] + merged * nbins_y + cy],
            1
        );
    }
    """,
        "compute_joint_hist_multi_pair_cuda",
    )

    # Shared-mem multi-pair variant. Win axis: small per-pair joint_size_y
    # (joint_size_y = nbins_a * nbins_b * nbins_y). Each block builds the
    # pair-local histogram in shared memory; one global reduce at block-end.
    # Caller passes ``max_joint_size_y`` (= max over all pairs in the launch)
    # as the dynamic-shared-mem size at kernel-launch time. Pairs with
    # smaller joint_size only use the first slice of sm_hist; the rest is
    # wasted shared-mem -- acceptable when the variance across pairs is
    # bounded (mlframe cat-FE typically has uniform nbins).
    module.compute_joint_hist_multi_pair_shared_cuda = cp.RawKernel(
        r"""
    extern "C" __global__
    void compute_joint_hist_multi_pair_shared_cuda(
        const int *factors_data_T,    // (n_cols, n)
        const int *classes_y,          // (n,)
        const int *pairs_a,            // (n_pairs,)
        const int *pairs_b,            // (n_pairs,)
        const int *nbins_a,            // (n_pairs,)
        const int *joint_offsets,      // (n_pairs + 1,) prefix sum
        int *joint_counts_flat,        // sum(nbins_a[p]*nbins_b[p]*nbins_y)
        int n_rows,
        int n_pairs,
        int nbins_y,
        int max_joint_size_y           // shared-mem allocation size in cells
    ) {
        extern __shared__ int sm_hist[];  // size: max_joint_size_y * sizeof(int)
        int pid = blockIdx.y;
        if (pid >= n_pairs) return;

        int pair_size = joint_offsets[pid + 1] - joint_offsets[pid];
        int tid = threadIdx.x;
        int nthreads = blockDim.x;

        // Zero only the pair's slice of shared (rest is unused / garbage).
        for (int i = tid; i < pair_size; i += nthreads) {
            sm_hist[i] = 0;
        }
        __syncthreads();

        int col_a = pairs_a[pid];
        int col_b = pairs_b[pid];
        int nba = nbins_a[pid];

        int row = blockIdx.x * nthreads + tid;
        if (row < n_rows) {
            int va = factors_data_T[col_a * n_rows + row];
            int vb = factors_data_T[col_b * n_rows + row];
            int cy = classes_y[row];
            int merged = va + vb * nba;
            atomicAdd(&sm_hist[merged * nbins_y + cy], 1);
        }
        __syncthreads();

        // Reduce shared -> per-pair global slot.
        int g_off = joint_offsets[pid];
        for (int i = tid; i < pair_size; i += nthreads) {
            atomicAdd(&joint_counts_flat[g_off + i], sm_hist[i]);
        }
    }
    """,
        "compute_joint_hist_multi_pair_shared_cuda",
    )

    module.compute_mi_from_classes_cuda = cp.RawKernel(
        r"""
    extern "C" __global__
    void compute_mi_from_classes_cuda(const int *classes_x, const double *freqs_x,const int *classes_y, const double *freqs_y, int *joint_counts, double *totals, int n,int nbins_x,int nbins_y) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid==0){
            double total = 0.0;
            for (int i=0;i<nbins_x;++i){
                float prob_x = freqs_x[i];
                for (int j=0;j<nbins_y;++j){
                    int jc = joint_counts[i*2+j];
                    if (jc>0){
                        float prob_y = freqs_y[j];
                        double jf=(float)jc/ (float)n;
                        total += jf* log(jf / (prob_x * prob_y));
                    }
                }
            }
            totals[0]=total;
        }
    }
    """,
        "compute_mi_from_classes_cuda",
    )


compute_joint_hist_multi_pair_cuda: Any = None
compute_joint_hist_multi_pair_shared_cuda: Any = None

# D2 fix (Critic 1): the multi-GPU pin in ``init_kernels`` only affects
# the CALLING thread (CUDA runtime is thread-local). Store the selected
# device id at module level so every kernel entry-point can re-pin its
# own thread via ``with cp.cuda.Device(_BEST_DEVICE_ID):``. None until
# ``init_kernels`` runs; remains None on single-GPU / probe-failed
# hosts (no per-call switching needed).
_BEST_DEVICE_ID: int | None = None


def _ensure_kernels_inited() -> None:
    """Double-checked init guard. Cheap on the hot path, race-free on the first call across any combination of threads and joblib workers.

    Private; external callers should use ``mi_direct_gpu`` or ``mi_direct_gpu_batched``, which call this internally. Direct invocation is only needed when
    pre-warming CUDA kernels before a parallel joblib run.
    """
    if (compute_joint_hist_cuda is not None
            and compute_mi_from_classes_cuda is not None
            and compute_joint_hist_batched_cuda is not None
            and compute_joint_hist_batched_shared_cuda is not None
            and compute_joint_hist_multi_pair_shared_cuda is not None):
        return
    with _KERNEL_INIT_LOCK:
        if (compute_joint_hist_cuda is None
                or compute_mi_from_classes_cuda is None
                or compute_joint_hist_batched_cuda is None
                or compute_joint_hist_batched_shared_cuda is None
                or compute_joint_hist_multi_pair_shared_cuda is None):
            init_kernels()


def mi_direct_gpu_batched(
    factors_data,
    x: tuple,
    y: tuple,
    factors_nbins: np.ndarray,
    npermutations: int = 100,
    batch_size: int = 64,
    dtype=np.int32,
    classes_y: np.ndarray = None,
    freqs_y: np.ndarray = None,
) -> tuple:
    """Batched GPU permutation MI test.

    Generates ``npermutations`` shuffled copies of ``classes_y`` once, chunks them into batches of ``batch_size``, and processes each batch in a SINGLE kernel
    launch (instead of one launch per permutation as in ``mi_direct_gpu``). Cuts kernel-launch overhead from O(npermutations) to O(npermutations / batch_size).

    Auto-fallback: if ``batch_size * n * 4 bytes`` would exceed half of free GPU memory, ``batch_size`` shrinks to fit. For small datasets the overhead of
    permutation matrix generation can outweigh the saved launches; ``mi_direct_gpu`` remains the right call for ``npermutations < 32``.

    Returns ``(original_mi, confidence)`` -- same contract as ``mi_direct_gpu``.
    """
    import cupy as cp

    _ensure_kernels_inited()

    classes_x, freqs_x, _ = merge_vars(
        factors_data=factors_data, vars_indices=x,
        var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
    )
    if classes_y is None:
        classes_y, freqs_y, _ = merge_vars(
            factors_data=factors_data, vars_indices=y,
            var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
        )

    original_mi = compute_mi_from_classes(
        classes_x=classes_x, freqs_x=freqs_x,
        classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
    )

    if original_mi <= 0 or npermutations <= 0:
        return original_mi, 0.0

    n = len(classes_x)
    nbins_x = len(freqs_x)
    nbins_y = len(freqs_y)

    # OOM guard: cap batch_size to half of available free GPU memory.
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    bytes_per_perm = n * 4  # int32 permutation row
    safe_batch = max(1, int(free_bytes // 2 // bytes_per_perm))
    if batch_size > safe_batch:
        batch_size = safe_batch

    classes_x_gpu = cp.asarray(classes_x.astype(np.int32))
    classes_y_gpu = cp.asarray(classes_y.astype(np.int32))
    freqs_x_gpu = cp.asarray(freqs_x).astype(cp.float64)
    freqs_y_gpu = cp.asarray(freqs_y).astype(cp.float64)

    # Joint-hist kernel dispatch via the per-host kernel_tuning_cache. The
    # cache is built on first use by a 30s sweep and persisted to
    # ``~/.mlframe/kernel_tuning/{hw_fingerprint}.json``. Subsequent calls
    # do an O(N_regions) dict lookup (N_regions <= ~10). Falls back to the
    # hand-tuned source-code default (shared, bs=512) on cache miss /
    # CUDA-unavailable / lookup error so production fits never block on
    # the cache.
    #
    # Both kernel variants stay alongside per ``feedback_keep_all_kernel_versions``:
    #   * compute_joint_hist_batched_shared_cuda -- shared-mem atomic;
    #     wins at most (n_samples, joint_size) combos.
    #   * compute_joint_hist_batched_cuda -- global-atomic; wins at large
    #     joint_size on certain (n_samples, block_size) combos that the
    #     auto-tune sweep discovered (e.g. n=1M, joint=400 on cc 6.1).
    joint_size = nbins_x * nbins_y
    try:
        from mlframe.feature_selection._benchmarks.kernel_tuning_cache.dispatch import (
            lookup_joint_hist,
        )
        _choice = lookup_joint_hist(n_samples=n, joint_size=joint_size)
        use_shared_hist = _choice["kernel_variant"] == "shared"
        block_size = int(_choice["block_size"])
    except Exception:
        # Hand-tuned source-code fallback (matches the pre-cache defaults).
        use_shared_hist = joint_size <= 4096
        block_size = 512 if use_shared_hist else GPU_MAX_BLOCK_SIZE

    shared_mem_bytes = joint_size * 4 if use_shared_hist else 0
    grid_x = (n + block_size - 1) // block_size

    # ``.get()`` coalescing v2: stage each batch's failure-count as a
    # CuPy 0-d array in a Python list, then sync ONCE at end-of-loop via
    # ``cp.stack([...]).sum().get()``. The previous per-batch
    # ``int((mi_batch >= original_mi).sum().get())`` pattern triggered a
    # cross-device sync on every iteration -- profile on 1M x 30,
    # fe_npermutations=50 showed ``cupy.ndarray.get`` at 5.2s of 13.2s fit
    # wall (39%) with 42 calls = ~124ms each (Windows WDDM driver fault
    # latency).
    #
    # An earlier attempt (commit 7319f11 noted in its log) using a
    # device-side scalar accumulator ``cp.zeros((), int64)`` with ``+=``
    # measured worse at n=10k because per-batch in-place add on a 0-d
    # array is itself a kernel launch. The list-of-arrays pattern below
    # avoids both pitfalls: no per-batch sync, no per-batch device-side
    # arithmetic; just queue compute, sync once.
    batch_failures = []  # list of CuPy 0-d arrays
    nchecked = 0
    remaining = npermutations
    while remaining > 0:
        b = min(batch_size, remaining)
        # ``cp.argsort(uniform((b, n)))`` is statistically equivalent to Fisher-Yates for permutation tests (argsort of distinct floats is a bijection).
        rand = cp.random.uniform(size=(b, n), dtype=cp.float64)
        perm_idx = cp.argsort(rand, axis=1)  # (b, n) int64
        # Gather classes_y at these indices -> shuffled copies.
        perms_y = classes_y_gpu[perm_idx].astype(cp.int32)

        joint_counts_batch = cp.zeros((b, nbins_x * nbins_y), dtype=cp.int32)
        if use_shared_hist:
            compute_joint_hist_batched_shared_cuda(
                (grid_x, b),
                (block_size,),
                (
                    classes_x_gpu, perms_y, joint_counts_batch,
                    np.int32(n), np.int32(nbins_x), np.int32(nbins_y),
                ),
                shared_mem=shared_mem_bytes,
            )
        else:
            compute_joint_hist_batched_cuda(
                (grid_x, b),
                (block_size,),
                (
                    classes_x_gpu, perms_y, joint_counts_batch,
                    np.int32(n), np.int32(nbins_x), np.int32(nbins_y),
                ),
            )

        # Compute MI per row of joint_counts_batch via vectorised math.
        # joint_freqs = joint_counts / n; mi = sum p_xy log(p_xy / (p_x p_y)).
        joint_freqs = joint_counts_batch.astype(cp.float64) / n
        joint_freqs = joint_freqs.reshape(b, nbins_x, nbins_y)
        # Broadcast p_x * p_y (shape (nbins_x, nbins_y)).
        outer_marginals = freqs_x_gpu[:, None] * freqs_y_gpu[None, :]
        # Mask the RATIO (not the log) so nan / inf never enter cp.log.
        # An ``outer_marginals`` cell of 0 (e.g. unobserved class) would
        # produce inf at jf>0 or nan at jf==0; cp.log of either is nan
        # which CuPy 8+ propagates through fma despite the where-mask.
        # Safe form: where(joint_freqs > 0, jf / outer_marginals, 1.0)
        # so the masked path computes log(1) = 0 with no nan / inf in flight.
        outer_safe = cp.where(outer_marginals[None, :, :] > 0, outer_marginals[None, :, :], 1.0)
        ratio = cp.where(joint_freqs > 0, joint_freqs / outer_safe, 1.0)
        log_ratio = cp.log(ratio)
        mi_batch = (joint_freqs * log_ratio).sum(axis=(1, 2))  # (b,)

        # Stage on device; defer cross-device sync.
        batch_failures.append((mi_batch >= original_mi).sum())
        nchecked += b
        remaining -= b

    # Single end-of-loop sync. cp.stack into 1-D array of length len(batch_failures),
    # sum along axis 0, .get() once. For small batch counts (e.g. n=10k -> 8 batches)
    # the stack is cheap; for large (e.g. 1M -> 8 batches per call, ~9 calls -> 72
    # total) we still pay only 1 sync per call instead of N per call.
    if batch_failures:
        nfailed = int(cp.stack(batch_failures).sum())
    else:
        nfailed = 0

    confidence = (1.0 - nfailed / nchecked) if nchecked > 0 else 0.0
    if nfailed >= int(npermutations * 0.05):  # rough min_nonzero_confidence=0.95 default
        original_mi = 0.0

    return original_mi, confidence


def mi_direct_gpu_batched_streamed(
    factors_data,
    x: tuple,
    y: tuple,
    factors_nbins: np.ndarray,
    npermutations: int = 100,
    batch_size: int = 64,
    dtype=np.int32,
    classes_y: np.ndarray = None,
    freqs_y: np.ndarray = None,
    n_streams: int = 2,
) -> tuple:
    """CUDA-streams variant of :func:`mi_direct_gpu_batched`.

    Same contract (``(original_mi, confidence)``) but the batch loop
    alternates iterations across ``n_streams`` non-blocking CuPy streams
    (default 2). This lets iter (i+1)'s ``cp.random.uniform`` + argsort
    queue concurrently with iter (i)'s histogram kernel + MI math,
    potentially filling SM idle gaps that the single-stream variant leaves.

    On cc 6.1 (GTX 1050 Ti, 6 SMs) the SMs are usually fully saturated by
    one kernel at a time so the overlap is minimal; bench expected
    speedup is 0-15%. On cc 8.x (Ampere, 100+ SMs) the overlap is larger.

    Per ``feedback_keep_all_kernel_versions``: kept ALONGSIDE the original
    ``mi_direct_gpu_batched``. The dispatcher (kernel_tuning_cache) can
    route to either via the ``variant`` field after a sweep finds the
    crossover.

    Caveats:
    * Streams share the GPU memory pool; over-subscribing batches can
      cause OOM at smaller n. ``batch_size`` is still subject to the
      memGetInfo-derived cap (same as the non-streamed variant).
    * The cupy legacy ``cp.random.uniform`` global RNG serialises through
      a Python-side lock; with 2 streams the lock contention is tiny but
      observable. Migrating to ``cp.random.Generator`` per stream would
      remove the lock at the cost of more state.
    """
    import cupy as cp

    _ensure_kernels_inited()

    classes_x, freqs_x, _ = merge_vars(
        factors_data=factors_data, vars_indices=x,
        var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
    )
    if classes_y is None:
        classes_y, freqs_y, _ = merge_vars(
            factors_data=factors_data, vars_indices=y,
            var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
        )

    original_mi = compute_mi_from_classes(
        classes_x=classes_x, freqs_x=freqs_x,
        classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
    )
    if original_mi <= 0 or npermutations <= 0:
        return original_mi, 0.0

    n = len(classes_x)
    nbins_x = len(freqs_x)
    nbins_y = len(freqs_y)

    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    bytes_per_perm = n * 4
    safe_batch = max(1, int(free_bytes // 2 // bytes_per_perm))
    if batch_size > safe_batch:
        batch_size = safe_batch

    classes_x_gpu = cp.asarray(classes_x.astype(np.int32))
    classes_y_gpu = cp.asarray(classes_y.astype(np.int32))
    freqs_x_gpu = cp.asarray(freqs_x).astype(cp.float64)
    freqs_y_gpu = cp.asarray(freqs_y).astype(cp.float64)

    joint_size = nbins_x * nbins_y
    use_shared_hist = joint_size <= 4096
    block_size = 512 if use_shared_hist else GPU_MAX_BLOCK_SIZE
    shared_mem_bytes = joint_size * 4 if use_shared_hist else 0
    grid_x = (n + block_size - 1) // block_size

    # B2 fix: ``classes_y_gpu``, ``freqs_x_gpu``, ``freqs_y_gpu`` were just
    # allocated on the default stream. Non-blocking child streams do NOT
    # implicitly wait for the default stream's H2D copies, so the first
    # iter could read garbage. Force a sync before launching child streams.
    cp.cuda.Stream.null.synchronize()

    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(max(1, n_streams))]
    # B1 fix: cp.random.uniform uses a singleton device-side RandomState
    # whose curandState advance is NOT safe under concurrent stream access.
    # One Generator per stream avoids the race; each generator's Philox
    # state advance stays serial within its own stream.
    rngs = [cp.random.default_rng() for _ in range(len(streams))]
    batch_failures = []
    nchecked = 0
    remaining = npermutations
    iter_idx = 0
    while remaining > 0:
        b = min(batch_size, remaining)
        stream = streams[iter_idx % len(streams)]
        rng = rngs[iter_idx % len(rngs)]
        with stream:
            # ``rng.random`` (per-stream) replaces the legacy global
            # ``cp.random.uniform`` to stay race-free across streams.
            rand = rng.random(size=(b, n), dtype=cp.float64)
            perm_idx = cp.argsort(rand, axis=1)
            perms_y = classes_y_gpu[perm_idx].astype(cp.int32)

            joint_counts_batch = cp.zeros((b, joint_size), dtype=cp.int32)
            if use_shared_hist:
                compute_joint_hist_batched_shared_cuda(
                    (grid_x, b), (block_size,),
                    (classes_x_gpu, perms_y, joint_counts_batch,
                     np.int32(n), np.int32(nbins_x), np.int32(nbins_y)),
                    shared_mem=shared_mem_bytes,
                )
            else:
                compute_joint_hist_batched_cuda(
                    (grid_x, b), (block_size,),
                    (classes_x_gpu, perms_y, joint_counts_batch,
                     np.int32(n), np.int32(nbins_x), np.int32(nbins_y)),
                )
            joint_freqs = joint_counts_batch.astype(cp.float64) / n
            joint_freqs = joint_freqs.reshape(b, nbins_x, nbins_y)
            outer_marginals = freqs_x_gpu[:, None] * freqs_y_gpu[None, :]
            # Same C3 fix as the non-streamed variant: mask the ratio
            # (not the log) to keep nan / inf out of cp.log entirely.
            outer_safe = cp.where(outer_marginals[None, :, :] > 0, outer_marginals[None, :, :], 1.0)
            ratio = cp.where(joint_freqs > 0, joint_freqs / outer_safe, 1.0)
            log_ratio = cp.log(ratio)
            mi_batch = (joint_freqs * log_ratio).sum(axis=(1, 2))
            batch_failures.append((mi_batch >= original_mi).sum())
        iter_idx += 1
        nchecked += b
        remaining -= b

    # Sync all streams before the host-side aggregation.
    for s in streams:
        s.synchronize()

    if batch_failures:
        nfailed = int(cp.stack(batch_failures).sum())
    else:
        nfailed = 0

    confidence = (1.0 - nfailed / nchecked) if nchecked > 0 else 0.0
    if nfailed >= int(npermutations * 0.05):
        original_mi = 0.0
    return original_mi, confidence


def mi_direct_gpu(
    factors_data,
    x: tuple,
    y: tuple,
    factors_nbins: np.ndarray,
    min_occupancy: int = None,
    dtype=np.int32,
    npermutations: int = 10,
    max_failed: int = None,
    min_nonzero_confidence: float = 0.95,
    classes_y: np.ndarray = None,
    classes_y_safe: np.ndarray = None,  # CuPy GPU array -- distinguished from the CPU classes_y_safe at the API boundary
    freqs_y: np.ndarray = None,
    freqs_y_safe: np.ndarray = None,
    use_gpu: bool = True,
) -> tuple:
    """GPU mutual-information + permutation test (CuPy).

    When ``npermutations >= 32`` and the caller did not pre-warm the
    device-side target buffers (``classes_y_safe is None and
    freqs_y_safe is None``), this function delegates to
    :func:`mi_direct_gpu_batched`, which runs ``batch_size=64``
    permutations per kernel launch. That cuts kernel-launch overhead
    from O(npermutations) to O(npermutations / 64) -- material on the
    screening hot path where the same single-pair MI evaluation is
    repeated for every candidate.

    Below 32 permutations the per-permutation generation overhead in
    the batched path dominates the launch saving; we keep the legacy
    per-iter loop. The 32 threshold comes from
    ``mi_direct_gpu_batched``'s own documented crossover.

    Return contract is identical (``(original_mi, confidence)``) in
    both branches; the only behavioural difference is that the
    batched path checks the ``nfailed >= max_failed`` early-stop
    condition at batch granularity, so up to ``batch_size - 1`` extra
    permutations may run before short-circuit (typically dominated by
    the saved launch overhead, but documented so callers depending
    on exact-perm-counts know).
    """
    import cupy as cp

    _ensure_kernels_inited()

    # Transparent fan-out to the batched permutation kernel when the
    # caller hasn't pre-warmed device buffers (those carry caller-
    # provided state we'd silently bypass) AND we have enough
    # permutations to amortise the batch-generation overhead.
    if (
        npermutations >= 32
        and classes_y_safe is None
        and freqs_y_safe is None
    ):
        return mi_direct_gpu_batched(
            factors_data=factors_data,
            x=x,
            y=y,
            factors_nbins=factors_nbins,
            npermutations=npermutations,
            batch_size=64,
            dtype=dtype,
            classes_y=classes_y,
            freqs_y=freqs_y,
        )

    classes_x, freqs_x, _ = merge_vars(
        factors_data=factors_data, vars_indices=x,
        var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
    )
    if classes_y is None:
        classes_y, freqs_y, _ = merge_vars(
            factors_data=factors_data, vars_indices=y,
            var_is_nominal=None, factors_nbins=factors_nbins, dtype=dtype,
        )

    original_mi = compute_mi_from_classes(
        classes_x=classes_x, freqs_x=freqs_x,
        classes_y=classes_y, freqs_y=freqs_y, dtype=dtype,
    )

    confidence = 0.0
    if original_mi > 0 and npermutations > 0:
        if not max_failed:
            max_failed = int(npermutations * (1 - min_nonzero_confidence))
            if max_failed <= 1:
                max_failed = 1

        # Persistent device buffers. Pool grows monotonically so back-to-back calls on similarly-sized inputs reuse the same allocations.
        n = len(classes_x)
        nbins_x = len(freqs_x)
        nbins_y = len(freqs_y)
        _GPU_POOL.ensure(n=n, nbins_x=nbins_x, nbins_y=nbins_y)

        # classes_y_safe override path (caller passed cached GPU arrays).
        if classes_y_safe is None:
            classes_y_safe = _GPU_POOL.classes_y[:n]
            classes_y_safe[:] = cp.asarray(classes_y, dtype=cp.int32)
        if freqs_y_safe is None:
            freqs_y_safe = _GPU_POOL.freqs_y[:nbins_y]
            freqs_y_safe[:] = cp.asarray(freqs_y)

        totals = _GPU_POOL.totals
        totals.fill(0)
        joint_counts = _GPU_POOL.joint_counts[:nbins_x, :nbins_y]

        # block_size via the per-host kernel_tuning_cache: lets fast GPUs
        # (cc 7+, more registers/warp scheduler) pick a smaller block size
        # when the per-perm kernel is dispatch-bound. Cache miss -> the
        # hardcoded ``GPU_MAX_BLOCK_SIZE`` (1024), unchanged.
        block_size = GPU_MAX_BLOCK_SIZE
        try:
            from pyutilz.system.kernel_tuning_cache import KernelTuningCache
            _ktc_entry = KernelTuningCache().lookup(
                "joint_hist_single_perm", n_samples=int(n),
            )
            if _ktc_entry is not None and "block_size" in _ktc_entry:
                block_size = int(_ktc_entry["block_size"])
        except Exception:
            pass
        grid_size = math.ceil(n / block_size)

        classes_x_gpu = _GPU_POOL.classes_x[:n]
        classes_x_gpu[:] = cp.asarray(classes_x, dtype=cp.int32)
        classes_x = classes_x_gpu
        freqs_x_gpu = _GPU_POOL.freqs_x[:nbins_x]
        freqs_x_gpu[:] = cp.asarray(freqs_x)
        freqs_x = freqs_x_gpu
        nfailed = 0
        _i = 0
        for _i in range(npermutations):
            cp.random.shuffle(classes_y_safe)
            joint_counts.fill(0)
            compute_joint_hist_cuda(
                (grid_size,),
                (block_size,),
                (classes_x, classes_y_safe, joint_counts, len(classes_x), len(freqs_y)),
            )
            compute_mi_from_classes_cuda(
                (1,),
                (1,),
                (classes_x, freqs_x, classes_y_safe, freqs_y_safe, joint_counts, totals, len(classes_x), len(freqs_x), len(freqs_y)),
            )

            mi = totals.get()[0]

            if mi >= original_mi:
                nfailed += 1
                if nfailed >= max_failed:
                    original_mi = 0.0
                    break
        confidence = 1 - nfailed / (_i + 1)

    return original_mi, confidence


# ============================================================================
# Cat-FE GPU dispatch: batched joint MI computation across many pairs, sharing classes_y on device. Use case is cat-FE pair search at N >= 200 cat cols and
# n >= 500k, where CPU prange is memory-bandwidth bound (~minutes per pair search); GPU drops it to seconds by amortising the H2D copy of classes_y across all pairs.
# ============================================================================


def mi_direct_gpu_batched_pairs(
    factors_data: np.ndarray,
    pairs_a: np.ndarray,
    pairs_b: np.ndarray,
    factors_nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    dtype=np.int32,
) -> np.ndarray:
    """Compute ``I(X_i, X_j; Y)`` for every ``(i, j) in zip(pairs_a, pairs_b)`` on GPU. Returns a 1-D ``float64`` array of joint MIs aligned with the input order.

    Kernel-level batching via 3D grid: ONE launch processes all pairs in parallel through ``compute_joint_hist_multi_pair_cuda``. Joint MI per pair is then
    computed on CPU (cheap relative to histogram aggregation). Cuts per-pair kernel-launch overhead (~30us) to zero -- at n_pairs=4950 that saves ~150ms vs the
    naive per-pair loop.

    Preconditions: CuPy installed and a GPU available; ``classes_y`` / ``freqs_y`` precomputed by the caller. Raises a clear error if CuPy is missing -- callers
    must resolve ``backend`` before calling.
    """
    try:
        import cupy as cp
    except ImportError as e:
        raise RuntimeError(
            "mi_direct_gpu_batched_pairs requires CuPy; install via "
            "`pip install cupy-cudaXX` matching your CUDA toolkit, "
            "or set CatFEConfig(backend='cpu')."
        ) from e

    _ensure_kernels_inited()
    n_pairs = len(pairs_a)
    n_rows = factors_data.shape[0]
    if n_pairs == 0:
        return np.zeros(0, dtype=np.float64)

    # E1 fix: CUDA gridDim.y maxes at 65535 on cc 6.x (still 65535 on cc 7+;
    # only x can go up to 2^31-1). At n_pairs > 65535 the launch fails with
    # an opaque InvalidConfiguration. Validate host-side with a clear
    # error so callers know to chunk.
    if n_pairs > 65535:
        raise ValueError(
            f"mi_direct_gpu_batched_pairs: n_pairs={n_pairs} exceeds CUDA "
            f"gridDim.y limit of 65535; chunk the call host-side."
        )

    # A5 fix (Critic 1): validate factors_data values are within their
    # declared bin range. Out-of-range values produce a ``merged`` index
    # that overflows the per-pair slice and silently corrupts the next
    # pair's histogram (still inside joint_counts_flat -- no segfault,
    # just wrong MI numbers). The CPU path validates via ``merge_vars``;
    # the GPU path didn't. Cheap host-side max-per-column scan.
    _ref_cols = np.unique(np.concatenate([np.asarray(pairs_a), np.asarray(pairs_b)]))
    for _c in _ref_cols:
        _c_int = int(_c)
        _col = factors_data[:, _c_int]
        _hi = int(_col.max()) if _col.size else -1
        _nb = int(factors_nbins[_c_int])
        if _hi >= _nb:
            raise ValueError(
                f"mi_direct_gpu_batched_pairs: factors_data[:, {_c_int}] "
                f"contains {_hi}, but factors_nbins[{_c_int}]={_nb} "
                f"(values must be in [0, nbins)). The kernel's "
                f"``merged = va + vb * nba`` arithmetic would overflow "
                f"the pair's joint slice."
            )
        if _col.size and int(_col.min()) < 0:
            raise ValueError(
                f"mi_direct_gpu_batched_pairs: factors_data[:, {_c_int}] "
                f"contains negative values; factor codes must be >= 0."
            )

    # Per-pair joint cardinalities ((nbins_a * nbins_b) cells for the MERGED dim) plus prefix-sum offsets. Each pair's joint table has shape (merged_size, nbins_y),
    # flattened row-major.
    nbins_a = factors_nbins[pairs_a].astype(np.int32)
    nbins_b = factors_nbins[pairs_b].astype(np.int32)
    nbins_y = int(np.asarray(freqs_y).shape[0])
    pair_merged_sizes = nbins_a.astype(np.int64) * nbins_b.astype(np.int64)
    pair_joint_sizes = pair_merged_sizes * nbins_y
    # A3 fix: int32 cumsum truncates at total_cells >= 2**31 (the 4 GB guard
    # below was only catching this by 1 bit of headroom). Promote
    # joint_offsets to int64 so the cumsum can never wrap.
    joint_offsets = np.zeros(n_pairs + 1, dtype=np.int64)
    joint_offsets[1:] = np.cumsum(pair_joint_sizes)
    total_cells = int(joint_offsets[-1])

    # GPU memory bound: total_cells int32. At total_cells = 1e7 that's 40 MB; at 1e8 that's 400 MB. Caller should constrain via ``max_combined_nbins`` to stay in bounds.
    if total_cells * 4 > 4 * 1024 * 1024 * 1024:  # > 4 GB
        raise MemoryError(
            f"mi_direct_gpu_batched_pairs: total joint cells {total_cells} "
            f"would require {total_cells*4/2**30:.1f} GB on GPU. Tighten "
            f"max_combined_nbins or run on CPU."
        )

    # Transposed factors_data for coalesced reads
    factors_data_T = np.ascontiguousarray(factors_data.T.astype(np.int32))
    factors_data_T_gpu = cp.asarray(factors_data_T)
    classes_y_gpu = cp.asarray(np.asarray(classes_y).astype(np.int32))
    pairs_a_gpu = cp.asarray(pairs_a.astype(np.int32))
    pairs_b_gpu = cp.asarray(pairs_b.astype(np.int32))
    nbins_a_gpu = cp.asarray(nbins_a)
    # Host-side joint_offsets is int64 (overflow-safe cumsum); the kernel
    # signature ``const int *joint_offsets`` (int32) is preserved by
    # narrowing here -- safe because the earlier 4 GB total_cells guard
    # enforces ``total_cells < 2**30 < 2**31``, which fits int32 exactly.
    joint_offsets_gpu = cp.asarray(joint_offsets.astype(np.int32))
    joint_counts_flat = cp.zeros(total_cells, dtype=cp.int32)

    # block_size via per-host kernel_tuning_cache (cache miss -> hand-tuned
    # 256, the original default for atomic-heavy multi-pair). cc 8+ devices
    # often prefer 512 here; the auto-tune sweep finds the win.
    block_size = min(GPU_MAX_BLOCK_SIZE, 256)
    try:
        from pyutilz.system.kernel_tuning_cache import KernelTuningCache
        _entry = KernelTuningCache().lookup(
            "joint_hist_multi_pair",
            n_rows=int(n_rows), n_pairs=int(n_pairs),
        )
        if _entry is not None and "block_size" in _entry:
            block_size = int(_entry["block_size"])
    except Exception:
        pass
    grid_x = (n_rows + block_size - 1) // block_size

    # Kernel-variant dispatch: shared-mem multi-pair kernel wins for small
    # per-pair joint_size_y (typical cat-FE nbins=5, ny=3 -> joint=15;
    # 5*5*3=75 cells × int32 = 300 bytes per pair). Cap conservatively at
    # 4096 cells (16 KB shared per block) to leave runtime headroom.
    max_joint_size_y = int(pair_joint_sizes.max()) if len(pair_joint_sizes) else 0
    _SHARED_MULTI_PAIR_MAX = 4096
    use_shared_multi_pair = (
        max_joint_size_y > 0 and max_joint_size_y <= _SHARED_MULTI_PAIR_MAX
    )

    # SINGLE kernel launch processes all pairs via 3D grid (rows along X, pairs along Y); per-pair launch overhead amortised to zero.
    if use_shared_multi_pair:
        compute_joint_hist_multi_pair_shared_cuda(
            (grid_x, n_pairs),
            (block_size,),
            (
                factors_data_T_gpu, classes_y_gpu, pairs_a_gpu, pairs_b_gpu,
                nbins_a_gpu, joint_offsets_gpu,
                joint_counts_flat, n_rows, n_pairs, nbins_y,
                np.int32(max_joint_size_y),
            ),
            shared_mem=max_joint_size_y * 4,
        )
    else:
        compute_joint_hist_multi_pair_cuda(
            (grid_x, n_pairs),
            (block_size,),
            (
                factors_data_T_gpu, classes_y_gpu, pairs_a_gpu, pairs_b_gpu,
                nbins_a_gpu, joint_offsets_gpu,
                joint_counts_flat, n_rows, n_pairs, nbins_y,
            ),
        )
    joint_counts_host = cp.asnumpy(joint_counts_flat)

    # Per-pair MI from joint counts (numpy). joint shape = (merged_size, nbins_y); marg_m = joint.sum(axis=1); marg_y = joint.sum(axis=0); total = n_rows.
    # MI = sum over non-zero cells of (jc/n) * log(jc * n / (marg_m * marg_y)), in nats.
    n_total = float(n_rows)
    joint_mi_out = np.zeros(n_pairs, dtype=np.float64)
    for k in range(n_pairs):
        off = int(joint_offsets[k])
        merged_size = int(pair_merged_sizes[k])
        joint_2d = joint_counts_host[off : off + merged_size * nbins_y].reshape(
            merged_size, nbins_y
        )
        marg_m = joint_2d.sum(axis=1)
        marg_y = joint_2d.sum(axis=0)
        # MI in nats
        mi = 0.0
        for m in range(merged_size):
            mm = marg_m[m]
            if mm == 0:
                continue
            for y in range(nbins_y):
                jc = joint_2d[m, y]
                if jc == 0:
                    continue
                my = marg_y[y]
                if my == 0:
                    continue
                jf = jc / n_total
                mi += jf * np.log(jc * n_total / (mm * my))
        joint_mi_out[k] = mi
    return joint_mi_out

