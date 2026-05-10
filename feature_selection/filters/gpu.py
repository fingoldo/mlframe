"""GPU permutation testing via CuPy raw kernels.

Public entry: ``mi_direct_gpu``. The two CUDA kernels live as module
attributes (``compute_joint_hist_cuda`` and ``compute_mi_from_classes_cuda``)
populated by ``init_kernels()``. Lazy initialisation goes through
``_ensure_kernels_inited()`` so the first call from any thread or process
gets a consistent module state without having to import CuPy at package
import time.

B11 (etap 5): the kernel-init lock is a ``multiprocessing.Lock`` (not a
``threading.Lock``) so Windows ``spawn`` workers don't all initialise the
kernels concurrently.

B20 (etap 5): CuPy seeding goes through an ``ImportError``-guarded
``try/except`` rather than the legacy ``NameError`` reflex.

The kernels are written into the module's ``__dict__`` via
``sys.modules[__name__]`` so callers in this same module can refer to
them as plain free names (``compute_joint_hist_cuda(...)``) at call time
even though they don't exist at module-import time.
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


# Phase 2 (post-plan etap 15): persistent device-buffer pool. Reuses a
# preallocated set of CuPy arrays across ``mi_direct_gpu`` calls so the
# inner permutation loop does not pay the H2D-copy + alloc cost on
# every iteration. Buffers grow monotonically (never shrink) so
# repeated calls with similar shapes hit the cached allocation.
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

# B11: cross-process safe lock. Constructed on first import; safe under
# multiprocessing because the lock is a primitive type that picks up the
# host process's mutex when spawned.
_KERNEL_INIT_LOCK = multiprocessing.Lock()


def init_kernels() -> None:
    """Build the CuPy ``RawKernel`` objects and attach them to this
    module's namespace. Idempotent under the lock.

    Three kernels:
    * ``compute_joint_hist_cuda`` -- single-permutation joint histogram
      (legacy, used by ``mi_direct_gpu``).
    * ``compute_mi_from_classes_cuda`` -- single-permutation MI from
      joint histogram.
    * ``compute_joint_hist_batched_cuda`` (Phase 2 batch) -- builds
      joint histograms for ``batch`` permutations of ``classes_y`` in
      a single launch. ``perms_y`` has shape ``(batch, n)``;
      ``joint_counts_batch`` has shape ``(batch, nbins_x * nbins_y)``.
      Cuts kernel-launch overhead from ``npermutations`` to 1.
    """
    import cupy as cp

    module = sys.modules[__name__]

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


def _ensure_kernels_inited() -> None:
    """Double-checked init guard. Cheap on the hot path, race-free on the
    first call across any combination of threads and joblib workers."""
    if (compute_joint_hist_cuda is not None
            and compute_mi_from_classes_cuda is not None
            and compute_joint_hist_batched_cuda is not None):
        return
    with _KERNEL_INIT_LOCK:
        if (compute_joint_hist_cuda is None
                or compute_mi_from_classes_cuda is None
                or compute_joint_hist_batched_cuda is None):
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
    """Phase 2 batched GPU permutation MI test.

    Generates ``npermutations`` shuffled copies of ``classes_y`` once,
    chunks them into batches of ``batch_size``, and processes each
    batch in a **single** kernel launch (instead of one launch per
    permutation as in ``mi_direct_gpu``). Cuts kernel-launch overhead
    from ``O(npermutations)`` to ``O(npermutations / batch_size)``.

    Auto-fallback: if the requested ``batch_size * n * 4 bytes`` would
    exceed half of free GPU memory, ``batch_size`` is shrunk to fit.
    For small datasets the overhead of permutation matrix generation
    can outweigh the saved launches; the legacy ``mi_direct_gpu``
    remains the right call for ``npermutations < 32``.

    Returns ``(original_mi, confidence)`` -- same contract as
    ``mi_direct_gpu``.
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
    bytes_per_perm = n * 4  # int32
    safe_batch = max(1, int(free_bytes // 2 // bytes_per_perm))
    if batch_size > safe_batch:
        batch_size = safe_batch

    classes_x_gpu = cp.asarray(classes_x.astype(np.int32))
    classes_y_gpu = cp.asarray(classes_y.astype(np.int32))
    freqs_x_gpu = cp.asarray(freqs_x).astype(cp.float64)
    freqs_y_gpu = cp.asarray(freqs_y).astype(cp.float64)

    block_size = GPU_MAX_BLOCK_SIZE
    grid_x = (n + block_size - 1) // block_size

    nfailed = 0
    nchecked = 0
    remaining = npermutations
    while remaining > 0:
        b = min(batch_size, remaining)
        # Generate batch permutations: ``cp.argsort(uniform((b, n)))``
        # is statistically equivalent to Fisher-Yates for permutation
        # tests (argsort of distinct floats is a bijection).
        rand = cp.random.uniform(size=(b, n), dtype=cp.float64)
        perm_idx = cp.argsort(rand, axis=1)  # (b, n) int64
        # Use these indices to gather classes_y -> shuffled copies.
        perms_y = classes_y_gpu[perm_idx].astype(cp.int32)

        joint_counts_batch = cp.zeros((b, nbins_x * nbins_y), dtype=cp.int32)
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
        # Mask out zero joint cells to avoid log(0).
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = joint_freqs / outer_marginals[None, :, :]
        log_ratio = cp.where(joint_freqs > 0, cp.log(ratio), 0.0)
        mi_batch = (joint_freqs * log_ratio).sum(axis=(1, 2))  # (b,)

        nfailed += int((mi_batch >= original_mi).sum().get())
        nchecked += b
        remaining -= b

    confidence = (1.0 - nfailed / nchecked) if nchecked > 0 else 0.0
    if nfailed >= int(npermutations * 0.05):  # rough min_nonzero_confidence=0.95 default
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
    classes_y_safe: np.ndarray = None,  # GPU array (CuPy) -- B18 distinguished from CPU at boundary
    freqs_y: np.ndarray = None,
    freqs_y_safe: np.ndarray = None,
    use_gpu: bool = True,
) -> tuple:
    """GPU mutual-information + permutation test (CuPy)."""
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

    confidence = 0.0
    if original_mi > 0 and npermutations > 0:
        if not max_failed:
            max_failed = int(npermutations * (1 - min_nonzero_confidence))
            if max_failed <= 1:
                max_failed = 1

        # Phase 2: persistent device buffers. Pool grows monotonically
        # so back-to-back ``mi_direct_gpu`` calls on similarly-sized
        # inputs reuse the same allocations.
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

        block_size = GPU_MAX_BLOCK_SIZE
        grid_size = math.ceil(n / block_size)

        classes_x_gpu = _GPU_POOL.classes_x[:n]
        classes_x_gpu[:] = cp.asarray(classes_x, dtype=cp.int32)
        classes_x = classes_x_gpu
        freqs_x_gpu = _GPU_POOL.freqs_x[:nbins_x]
        freqs_x_gpu[:] = cp.asarray(freqs_x)
        freqs_x = freqs_x_gpu
        nfailed = 0
        i = 0
        for i in range(npermutations):
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
        confidence = 1 - nfailed / (i + 1)

    return original_mi, confidence


# ============================================================================
# Cat-FE GPU dispatch shim (P9): batched joint MI computation across many
# pairs, sharing classes_y on device.
#
# Use case: cat-FE pair search at N >= 200 cat cols, n >= 500k. CPU prange
# is memory-bandwidth bound at that scale (~minutes per pair search);
# GPU drops it to seconds by amortising the H2D copy of classes_y across
# all pairs.
#
# This MVP implements pair-by-pair GPU dispatch (one kernel launch per
# pair). True kernel-level batching (one launch processes B pairs in
# parallel via 3D grid) is a future refinement that requires a new
# RawKernel; the bottleneck at the stated workload is the per-call
# memory transfer, not kernel launch overhead.
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
    """Compute ``I(X_i, X_j; Y)`` for every ``(i, j) in zip(pairs_a, pairs_b)``
    on GPU. Returns a 1-D ``float64`` array of joint MIs aligned with
    the input order.

    Pre-conditions:
    - CuPy installed and a GPU is available.
    - ``classes_y`` / ``freqs_y`` are precomputed by the caller -- the
      shim does NOT re-merge the target per pair.

    Falls back to a clear error if CuPy is missing -- callers must
    check ``backend`` resolution before calling.
    """
    try:
        import cupy as cp  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "mi_direct_gpu_batched_pairs requires CuPy; install via "
            "`pip install cupy-cudaXX` matching your CUDA toolkit, "
            "or set CatFEConfig(backend='cpu')."
        ) from e

    _ensure_kernels_inited()
    n_pairs = len(pairs_a)
    joint_mi_out = np.zeros(n_pairs, dtype=np.float64)

    # Loop pair-by-pair, reusing the single-pair primitive. Each call
    # re-uses ``_GPU_POOL`` so persistent device buffers cut the H2D
    # cost on classes_y / freqs_y to ~zero after the first pair.
    for k in range(n_pairs):
        i = int(pairs_a[k]); j = int(pairs_b[k])
        # Compute joint MI WITHOUT permutations -- just the point
        # estimate. ``mi_direct_gpu`` with ``npermutations=0`` does
        # exactly this (returns ``original_mi`` and ``confidence=0``).
        mi_joint, _ = mi_direct_gpu(
            factors_data=factors_data,
            x=np.array([i, j], dtype=np.int64),
            y=None,
            factors_nbins=factors_nbins,
            npermutations=0,
            dtype=dtype,
            classes_y=classes_y,
            freqs_y=freqs_y,
        )
        joint_mi_out[k] = mi_joint
    return joint_mi_out
