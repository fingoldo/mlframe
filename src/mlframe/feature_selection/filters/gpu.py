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
        # The CUDA joint-hist + MI kernels index ``joint_counts`` as a row-major
        # buffer with stride EXACTLY ``nbins_y`` (``cx*nbins_y+cy``). A pooled
        # buffer wider than the current request (``shape[1] > nbins_y``) would make
        # ``joint_counts[:nbins_x, :nbins_y]`` a NON-contiguous slice whose true row
        # stride is the allocated width, silently corrupting both the histogram
        # write and the MI read. Reallocate whenever the column width CHANGES (this
        # is a module-singleton pool that persists across fits, so a prior fit with
        # more target bins could leave a wider buffer); the row count may still grow
        # monotonically since slicing only the leading rows keeps each row contiguous.
        if (self.joint_counts is None
                or self.joint_counts.shape[0] < nbins_x
                or self.joint_counts.shape[1] != nbins_y):
            _rows = max(int(nbins_x), int(self.joint_counts.shape[0]) if self.joint_counts is not None else 0)
            self.joint_counts = cp.empty((_rows, nbins_y), dtype=cp.int32)
        if self.totals is None:
            self.totals = cp.zeros(1, dtype=cp.float64)

    def free(self) -> None:
        """Drop all device buffers + reset caps. Call while the CUDA context is still alive (e.g. a test
        session finalizer) so these persistent cupy allocations are released in a controlled order rather
        than during chaotic interpreter atexit teardown, where freeing them alongside torch/numba CUDA
        contexts has triggered heap corruption (0xc0000374) on multi-CUDA-library hosts."""
        self.classes_x = self.classes_y = self.freqs_x = self.freqs_y = None
        self.joint_counts = self.totals = None
        self.cap_n = self.cap_nbins_x = self.cap_nbins_y = 0


_GPU_POOL = _GpuBufferPool()

# Wave 27 P2 fix (2026-05-20): the prior docstring claimed
# "Cross-process safe ... picks up the host process's mutex on spawn".
# That is FALSE on Windows ``spawn`` (the joblib loky default): each
# child process re-imports the module and constructs its own
# ``Lock()`` -- the parent + children do NOT share the underlying
# ``SemLock`` handle. The lock only mutexes WITHIN one process.
# This is acceptable HERE because the lock body
# (``init_kernels``) is idempotent: duplicate CuPy NVRTC compile is
# wasted work but correctness is unaffected. Future contributors
# adding NON-idempotent init under this lock would silently break.
# Either pass via Pool(initializer=..., initargs=(lock,)) or use a
# Manager().Lock() if cross-process exclusion is ever required.
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
            // joint_counts is a (nbins_x, nbins_y) row-major histogram written by
            // compute_joint_hist_cuda with stride nbins_y (``cx*nbins_y+cy``). The
            // reduction MUST read it with the SAME stride. A pre-2026-06-11 bug
            // hardcoded ``i*2+j`` (stride 2), which only matched a binary target
            // (nbins_y==2); for any nbins_y!=2 (multi-class y, or quantile-binned
            // regression y with nbins_y up to 10) it read the WRONG cells, yielding
            // garbage permutation-null MIs that systematically over-rejected genuine
            // candidates -- silently diverging the GPU selection from the CPU path.
            // Use double throughout (no float32 narrowing) to match the bit-exact
            // CPU njit compute_mi_from_classes.
            double total = 0.0;
            for (int i=0;i<nbins_x;++i){
                double prob_x = freqs_x[i];
                for (int j=0;j<nbins_y;++j){
                    int jc = joint_counts[i*nbins_y+j];
                    if (jc>0){
                        double prob_y = freqs_y[j];
                        double jf=(double)jc/ (double)n;
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


def _pin_device_if_needed() -> None:
    """Per-thread idempotent pin to ``_BEST_DEVICE_ID``. Cheap no-op if
    already on the right device or if the pin is unset. CUDA runtime
    state is thread-local, so this MUST be called from every thread
    that runs an ``mi_direct_gpu*`` entry-point (joblib worker,
    asyncio task, etc.) -- the ``init_kernels`` pin only affected the
    init thread."""
    if _BEST_DEVICE_ID is None:
        return
    try:
        import cupy as cp
        if cp.cuda.Device().id != _BEST_DEVICE_ID:
            cp.cuda.Device(_BEST_DEVICE_ID).use()
    except Exception:
        pass


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


def _mi_from_counts_cupy(cp, counts_2d, n, nbins_x, nbins_y, outer_safe):
    """(b, joint_size) int counts -> (b,) MI in nats via the noise-gate reduction (masked-safe ratio; shared by both mi_direct_gpu_batched twins)."""
    jf = (counts_2d.astype(cp.float64) / n).reshape(-1, nbins_x, nbins_y)
    ratio = cp.where(jf > 0, jf / outer_safe, 1.0)
    return (jf * cp.log(ratio)).sum(axis=(1, 2))


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
    min_nonzero_confidence: float = 0.95,
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
    _pin_device_if_needed()  # D2 part 2: per-thread CUDA device pin

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

    # ABSOLUTE cushion guard (2026-07-05): decline the GPU entirely on a near-full / SHARED card BEFORE the
    # relative half-free batch cap below (which is computed only after the cupy pool may have already eaten the
    # device). On a cushion violation route to the exact CPU MI path (selection-equivalent, prefer_gpu=False so
    # it does not recurse back onto the GPU) instead of launching a kernel that would fault. Pure ADD -- tightens.
    try:
        from mlframe.feature_selection.filters._fe_gpu_vram import fe_gpu_has_vram_cushion
        _cushion_ok = fe_gpu_has_vram_cushion(n * 4)
    except Exception:  # noqa: BLE001  -- cushion module unavailable: leave existing guards in charge
        _cushion_ok = True
    if not _cushion_ok:
        from .permutation import mi_direct
        return mi_direct(
            factors_data, x=x, y=y, factors_nbins=factors_nbins,
            npermutations=npermutations, dtype=dtype,
            classes_y=classes_y, freqs_y=freqs_y,
            min_nonzero_confidence=min_nonzero_confidence, prefer_gpu=False,
        )
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
    # Loop-invariant marginal product + masked-safe form, hoisted out of the batch loop.
    outer_marginals = freqs_x_gpu[:, None] * freqs_y_gpu[None, :]
    outer_safe = cp.where(outer_marginals[None, :, :] > 0, outer_marginals[None, :, :], 1.0)

    # GATE REFERENCE (FP-consistency, 2026-06-22): the gate counts #{perm_mi >= observed_mi}; mi_batch is
    # GPU-reduced, so recompute the observed MI through the SAME kernel+reduction on the UNPERMUTED y (a
    # ~1e-15 order delta vs the CPU-njit original_mi flips a count). CPU original_mi stays the RETURNED value.
    _id_counts = cp.zeros((1, nbins_x * nbins_y), dtype=cp.int32)
    _id_perm = classes_y_gpu.reshape(1, n).astype(cp.int32)
    if use_shared_hist:
        compute_joint_hist_batched_shared_cuda(
            (grid_x, 1), (block_size,),
            (classes_x_gpu, _id_perm, _id_counts, np.int32(n), np.int32(nbins_x), np.int32(nbins_y)),
            shared_mem=shared_mem_bytes,
        )
    else:
        compute_joint_hist_batched_cuda(
            (grid_x, 1), (block_size,),
            (classes_x_gpu, _id_perm, _id_counts, np.int32(n), np.int32(nbins_x), np.int32(nbins_y)),
        )
    original_mi_gpu = _mi_from_counts_cupy(cp, _id_counts, n, nbins_x, nbins_y, outer_safe)  # (1,) device scalar, compared against mi_batch below

    batch_failures = []  # list of CuPy 0-d arrays
    nchecked = 0
    remaining = npermutations
    # Use the modern cupy Generator (XORWOW) rather than the legacy global cp.random.uniform: the legacy
    # host-API cuRAND generator fails to initialise (CURAND_STATUS_INITIALIZATION_FAILED) on some
    # driver/lib combos where the Generator API works fine -- and the streamed twin already uses default_rng.
    _rng = cp.random.default_rng()
    while remaining > 0:
        b = min(batch_size, remaining)
        # ``cp.argsort(random((b, n)))`` is statistically equivalent to Fisher-Yates for permutation tests (argsort of distinct floats is a bijection).
        rand = _rng.random((b, n), dtype=cp.float64)
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

        # MI per row via the SAME reduction as the gate reference (the masked-safe ratio avoids nan/inf in
        # cp.log; outer_safe hoisted above). joint_freqs = joint_counts / n; mi = sum p_xy log(p_xy/(p_x p_y)).
        mi_batch = _mi_from_counts_cupy(cp, joint_counts_batch, n, nbins_x, nbins_y, outer_safe)  # (b,)

        # Stage on device; defer cross-device sync. Compare against the GPU-reduced observed MI (same FP
        # order as mi_batch) -- NOT the CPU-njit original_mi -- so an equality near-tie cannot spuriously flip.
        batch_failures.append((mi_batch >= original_mi_gpu).sum())
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
    # 2026-05-30 Wave 9.1 fix (loop iter 4): respect caller's
    # min_nonzero_confidence instead of the previous hardcoded 0.05
    # threshold. The hardcode silently diverged CPU and GPU paths when the
    # caller (default MRMR ctor: min_nonzero_confidence=0.99) wanted a
    # tighter gate. Same mirror as permutation.py:344 CPU path. The
    # ``max(1, ...)`` clamp protects against degenerate ``npermutations``
    # auto-rejecting every candidate when the threshold rounds to 0.
    max_failed = int(npermutations * (1.0 - float(min_nonzero_confidence)))
    if max_failed <= 1:
        max_failed = 1
    if nfailed >= max_failed:
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
    min_nonzero_confidence: float = 0.95,
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
    _pin_device_if_needed()  # D2 part 2: per-thread CUDA device pin

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

    # ABSOLUTE cushion guard (2026-07-05): as in mi_direct_gpu_batched, decline the GPU on a near-full / shared
    # card BEFORE the relative half-free cap and route to the exact CPU MI path (selection-equivalent). Pure ADD.
    try:
        from mlframe.feature_selection.filters._fe_gpu_vram import fe_gpu_has_vram_cushion
        _cushion_ok = fe_gpu_has_vram_cushion(n * 4)
    except Exception:  # noqa: BLE001
        _cushion_ok = True
    if not _cushion_ok:
        from .permutation import mi_direct
        return mi_direct(
            factors_data, x=x, y=y, factors_nbins=factors_nbins,
            npermutations=npermutations, dtype=dtype,
            classes_y=classes_y, freqs_y=freqs_y,
            min_nonzero_confidence=min_nonzero_confidence, prefer_gpu=False,
        )

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
    # Wave 23 #1 fix (2026-05-20): mirror the sibling lookup at L466-475
    # so this streamed variant adapts to live HW. Pre-fix it open-coded
    # the same hand-tuned defaults from before kernel_tuning_cache landed
    # and silently fell behind on non-dev hardware.
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

    # B2 fix: ``classes_y_gpu``, ``freqs_x_gpu``, ``freqs_y_gpu`` were just
    # allocated on the default stream. Non-blocking child streams do NOT
    # implicitly wait for the default stream's H2D copies, so the first
    # iter could read garbage. Force a sync before launching child streams.
    cp.cuda.Stream.null.synchronize()

    # FP-consistency gate reference (mirrors the non-streamed twin, 2026-06-22): reduce the OBSERVED MI via
    # the same kernel+reduction on the unpermuted y (default stream, before child streams) so the >= gate
    # compares like-with-like. CPU original_mi stays the RETURNED value.
    outer_marginals = freqs_x_gpu[:, None] * freqs_y_gpu[None, :]
    outer_safe = cp.where(outer_marginals[None, :, :] > 0, outer_marginals[None, :, :], 1.0)

    _id_counts = cp.zeros((1, joint_size), dtype=cp.int32)
    _id_perm = classes_y_gpu.reshape(1, n).astype(cp.int32)
    if use_shared_hist:
        compute_joint_hist_batched_shared_cuda(
            (grid_x, 1), (block_size,),
            (classes_x_gpu, _id_perm, _id_counts, np.int32(n), np.int32(nbins_x), np.int32(nbins_y)),
            shared_mem=shared_mem_bytes,
        )
    else:
        compute_joint_hist_batched_cuda(
            (grid_x, 1), (block_size,),
            (classes_x_gpu, _id_perm, _id_counts, np.int32(n), np.int32(nbins_x), np.int32(nbins_y)),
        )
    cp.cuda.Stream.null.synchronize()
    original_mi_gpu = _mi_from_counts_cupy(cp, _id_counts, n, nbins_x, nbins_y, outer_safe)  # (1,) device scalar for the gate comparison below

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
            # Same reduction as the gate reference (outer_safe + closure hoisted above); mask the ratio (not
            # the log) to keep nan/inf out of cp.log entirely.
            mi_batch = _mi_from_counts_cupy(cp, joint_counts_batch, n, nbins_x, nbins_y, outer_safe)
            batch_failures.append((mi_batch >= original_mi_gpu).sum())
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
    # 2026-05-30 Wave 9.1 fix (loop iter 4): same fix as
    # ``mi_direct_gpu_batched`` above - respect caller's
    # min_nonzero_confidence; clamp to avoid degenerate auto-reject.
    max_failed = int(npermutations * (1.0 - float(min_nonzero_confidence)))
    if max_failed <= 1:
        max_failed = 1
    if nfailed >= max_failed:
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
    return_null_mean: bool = False,
) -> tuple:
    """GPU mutual-information + permutation test (CuPy).

    ``return_null_mean=True`` additionally returns the empirical permutation-null MEAN (the average MI of X
    against the y-shuffles this call already computes) and the add-one permutation p-value, so the caller can
    apply the SAME significance-gated relevance debiasing as the CPU ``mi_direct`` path -- without it, the GPU
    relevance was RAW plug-in MI (high-cardinality-biased, and selection differed from the CPU branch). Returns
    a 4-tuple ``(mi, confidence, null_mean, p_value)`` instead of the 2-tuple.

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
    _pin_device_if_needed()  # D2 part 2: per-thread CUDA device pin

    # Transparent fan-out to the batched permutation kernel when the
    # caller hasn't pre-warmed device buffers (those carry caller-
    # provided state we'd silently bypass) AND we have enough
    # permutations to amortise the batch-generation overhead.
    if (
        npermutations >= 32
        and classes_y_safe is None
        and freqs_y_safe is None
        and not return_null_mean  # the batched path does not accumulate the null mean; keep the per-iter loop
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
            min_nonzero_confidence=min_nonzero_confidence,
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

    # return_null_mean needs a STABLE null mean AND a usable permutation p-value, so mirror the CPU mi_direct path
    # (permutation.py:642): run a larger fixed null budget (_NULL_MEAN_MIN_PERMS, default 32) rather than the screen's
    # tiny exceedance budget. Without this the GPU ran only ``npermutations`` (as low as 2) shuffles, so the add-one
    # p-value floored at (1+0)/(2+1)=0.33 >= alpha for EVERY feature -- the null-mean debiasing was then applied even
    # to strong genuine signal, and GPU-present hosts selected differently from CPU-only hosts. max_failed is lifted
    # to the full budget so the early-stop cannot truncate the null (unbiased mean + full-resolution p-value).
    if return_null_mean:
        from .permutation import _NULL_MEAN_MIN_PERMS
        npermutations = max(int(npermutations), _NULL_MEAN_MIN_PERMS)
        max_failed = npermutations

    confidence = 0.0
    nfailed = 0            # defined outside the block so the return_null_mean path is valid when the block is skipped
    _null_sum = 0.0        # sum of per-permutation MIs -> empirical null mean = _null_sum / _nchecked
    _nchecked = 0
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
        # Module-singleton cache (see ._kernel_tuning); a fresh
        # KernelTuningCache() here would re-spawn nvidia-smi per call.
        block_size = GPU_MAX_BLOCK_SIZE
        from ._kernel_tuning import get_kernel_tuning_cache
        _ktc = get_kernel_tuning_cache()
        if _ktc is not None:
            try:
                _ktc_entry = _ktc.lookup("joint_hist_single_perm", n_samples=int(n))
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
        # Modern Generator (XORWOW) rather than legacy cp.random.shuffle: the legacy global cuRAND host
        # generator fails to init (CURAND_STATUS_INITIALIZATION_FAILED) on some driver/lib combos. cupy's
        # Generator has no .shuffle, so shuffle in-place via argsort(random) (a permutation of distinct
        # floats is a bijection) -- preserving the pooled buffer identity downstream consumers rely on.
        _shuf_rng = cp.random.default_rng()
        _shuf_n = classes_y_safe.shape[0]
        for _i in range(npermutations):
            classes_y_safe[:] = classes_y_safe[cp.argsort(_shuf_rng.random(_shuf_n))]
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
            _null_sum += float(mi)  # accumulate for the empirical null mean (return_null_mean path)

            if mi >= original_mi:
                nfailed += 1
                if nfailed >= max_failed:
                    original_mi = 0.0
                    break
        confidence = 1 - nfailed / (_i + 1)
        _nchecked = _i + 1

    if return_null_mean:
        # Mirror the CPU mi_direct(return_null_mean=True) contract: empirical null mean over the permutations
        # actually run, and the add-one Monte-Carlo p-value (budget-consistent under early-stop) so evaluation
        # applies the SAME significance-gated relevance debiasing on the GPU path as on the CPU path.
        from .permutation import _perm_pvalue
        _null_mean = (_null_sum / _nchecked) if _nchecked > 0 else 0.0
        _p_value = _perm_pvalue(nfailed, _nchecked, full_budget=npermutations) if _nchecked > 0 else 1.0
        return original_mi, confidence, _null_mean, _p_value
    return original_mi, confidence


# ============================================================================
# Cat-FE GPU dispatch: batched joint MI computation across many pairs, sharing classes_y on device. Use case is cat-FE pair search at N >= 200 cat cols and
# n >= 500k, where CPU prange is memory-bandwidth bound (~minutes per pair search); GPU drops it to seconds by amortising the H2D copy of classes_y across all pairs.


# Wave 99 (2026-05-21): mi_direct_gpu_batched_pairs (~250 lines) moved to
# sibling file _gpu_pairs.py to drop this file below the 1k-line monolith
# threshold. Re-exported below so existing callers
# (`from mlframe.feature_selection.filters.gpu import mi_direct_gpu_batched_pairs`)
# keep working.
from ._gpu_pairs import mi_direct_gpu_batched_pairs  # noqa: F401, E402

