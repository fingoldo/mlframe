"""Compute kernels for the batched FE-candidate MI + permutation noise-gate.

Carved verbatim out of ``batch_mi_noise_gate_gpu.py`` (Tier E LOC-budget split):
the optional GPU dependency probes, the cupy/numba.cuda kernel factories (the
RawKernel-equivalent source bodies) + their compile helpers, and the bit-exact
CPU njit reducers shared by both GPU backends. The parent module re-exports every
name below, so the public API and all import paths are unchanged. No kernel
source / numerics / dispatch behavior changed by the move.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

# Optional GPU deps -- mirror batch_pair_mi_gpu.py's probe order exactly.
try:
    from numba import cuda as _nb_cuda
except Exception:
    _nb_cuda = None

from ._internals import numba_cuda_can_compile as _numba_cuda_can_compile

try:
    from pyutilz.core.pythonlib import is_cuda_available as _pyutilz_is_cuda_available
    _CUDA_AVAIL = _pyutilz_is_cuda_available()
except Exception:
    try:
        _CUDA_AVAIL = bool(getattr(_nb_cuda, "is_available", lambda: False)()) if _nb_cuda is not None else False
    except Exception:
        _CUDA_AVAIL = False

# Device-presence alone is not enough: a GPU with a cudatoolkit/numba NVVM mismatch passes the
# probe above but raises NvvmSupportError on the first kernel launch. Require actual compilability
# so the dispatcher falls back to cupy/CPU instead of crashing.
_CUDA_AVAIL = _CUDA_AVAIL and _numba_cuda_can_compile()

try:
    import cupy as _cp
    _CUPY_AVAIL = True
except Exception:
    _cp = None
    _CUPY_AVAIL = False

# OPT-D (2026-06-07): cupy's public ``cupy.bincount`` runs TWO host-blocking
# synchronizations on EVERY call -- ``(x < 0).any()`` (non-negativity validation)
# and ``int(cupy.max(x))`` (output sizing) -- before the actual histogram kernel.
# In ``batch_mi_with_noise_gate_cupy`` both are pure overhead: the flat index is
# constructed non-negative (offsets + non-negative codes) and the output size is
# already known exactly (``rows*total_size`` = the ``minlength`` we pass). Those two
# syncs were the dominant cost of the scene MRMR ``bincount`` hotspot (~26% of fit
# wall in the sampler) -- NOT the count kernel. ``cupy._statistics.histogram._bincount_kernel``
# is the SAME ElementwiseKernel ``cupy.bincount`` dispatches into, so calling it
# directly into a pre-zeroed array of the known size is BYTE-IDENTICAL (verified:
# profiling/bench_cupy_bincount_sync.py, 3.0-5.4x faster, byte_identical=True on every
# scene FE-MI shape) while skipping both barriers. Probe the private symbol once at
# import; if a future cupy renames it we fall back to public ``cupy.bincount``.
try:
    from cupy._statistics.histogram import _bincount_kernel as _cupy_bincount_kernel
except Exception:
    _cupy_bincount_kernel = None


def _cupy_bincount_known_size(d_flat, size):
    """``cupy.bincount(d_flat, minlength=size)[:size]`` for the case where ``size`` is
    KNOWN exactly and ``d_flat`` is non-negative by construction -- skips cupy.bincount's
    ``(x<0).any()`` + ``cupy.max(x)`` host-sync barriers. BYTE-IDENTICAL output (same
    underlying ElementwiseKernel). Falls back to public ``cupy.bincount`` if the private
    kernel symbol is unavailable on this cupy build."""
    cp = _cp
    if _cupy_bincount_kernel is not None:
        import numpy as _np
        b = cp.zeros((size,), dtype=_np.intp)
        _cupy_bincount_kernel(d_flat, b)
        return b
    return cp.bincount(d_flat, minlength=size)[:size]


# ---------------------------------------------------------------------------
# Bit-exact CPU entropy-from-counts (shared by both GPU backends)
# ---------------------------------------------------------------------------


@njit(nogil=True, cache=True)
def _mi_from_counts_cpu(
    joint_counts: np.ndarray,   # (nbins_x, K_y) int64 -- integer joint histogram
    nbins_x: int,
    freqs_y: np.ndarray,        # (K_y,) float64
    n: int,
    use_su: bool,
) -> float:
    """MI (or SU) of one column against ``y`` from its INTEGER joint histogram,
    reproducing ``_relevance_from_dense`` to the bit.

    Iterates ``for i in nbins_x: for j in K_y`` in ascending bin-code order,
    skipping empty (count==0) cells -- exactly the CPU kernel's accumulation
    order. ``prob_x`` is ``fx/n`` (``fx`` = row sum), matching the CPU kernel's
    ``freqs_dense[k, i] = counts[c] / n`` (an int/int float division, NOT
    ``counts * (1/n)``), so the float result is identical bit-for-bit. Empty
    bins (pruned in the CPU dense remap) contribute 0 here too, so this FULL-bin
    pass matches the dense pass exactly.
    """
    K_y = freqs_y.shape[0]
    inv_n = 1.0 / n
    mi_xy = 0.0
    for i in range(nbins_x):
        fx = 0
        for j in range(K_y):
            fx += joint_counts[i, j]
        if fx == 0:
            continue
        # prob_x = counts/n via int/int division (bit-identical to the CPU dense path).
        prob_x = fx / n
        for j in range(K_y):
            jc = joint_counts[i, j]
            if jc != 0:
                prob_y = freqs_y[j]
                jf = jc * inv_n
                mi_xy += jf * math.log(jf / (prob_x * prob_y))

    if not use_su:
        return mi_xy

    h_x = 0.0
    for i in range(nbins_x):
        fx = 0
        for j in range(K_y):
            fx += joint_counts[i, j]
        if fx != 0:
            p = fx / n
            h_x -= p * math.log(p)
    h_y = 0.0
    for j in range(K_y):
        p = freqs_y[j]
        if p > 0:
            h_y -= p * math.log(p)
    denom = h_x + h_y
    if denom <= 1e-12:
        return 0.0
    return 2.0 * mi_xy / denom


@njit(nogil=True, cache=True)
def _mi_columns_from_counts_cpu(
    counts_flat: np.ndarray,    # (total_size,) int64 -- flat per-column joint histograms
    col_offsets: np.ndarray,    # (K,) int64 -- start offset of column k
    nbins_arr: np.ndarray,      # (K,) int64
    K_y: int,
    freqs_y: np.ndarray,        # (K_y,) float64
    n: int,
    use_su: bool,
    ref_mi: np.ndarray,         # (K,) float64 -- compute column k only when ref_mi[k] > 0
) -> np.ndarray:
    """Reduce MI for ALL K columns in ONE njit call instead of K separate Python->njit
    dispatches. Calls the SAME ``_mi_from_counts_cpu`` body per column (a compiled, not
    Python, call here), so the result is BIT-IDENTICAL to the per-column loop. The
    ``ref_mi`` mask reproduces the original loop's ``if original_mi[k] <= 0: continue``
    perm-skip (pass an all-positive array to compute every column, e.g. the original-y
    pass). Kills the ~1.9s the per-column ``_mi_from_counts_cpu`` dispatch cost across
    K*(npermutations+1) calls (224k calls at the canonical FE size)."""
    K = nbins_arr.shape[0]
    out = np.zeros(K, dtype=np.float64)
    for k in range(K):
        if ref_mi[k] <= 0.0:
            continue
        nb_k = nbins_arr[k]
        off = col_offsets[k]
        block = counts_flat[off: off + nb_k * K_y].reshape(nb_k, K_y)
        out[k] = _mi_from_counts_cpu(block, nb_k, freqs_y, n, use_su)
    return out


@njit(nogil=True, cache=True)
def _fisher_yates_shuffle(classes_y_safe: np.ndarray, base_seed: np.uint64, perm_index: int) -> np.ndarray:
    """Bit-identical copy of the CPU kernel's per-permutation Fisher-Yates shuffle
    (LCG seed ``base_seed*2654435761 + (i+1)`` then the PCG step). Returns a fresh
    shuffled copy of ``classes_y_safe``."""
    ny = classes_y_safe.shape[0]
    local = classes_y_safe.copy()
    state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(perm_index + 1)
    for j in range(ny - 1, 0, -1):
        state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        kk = int(state >> np.uint64(33)) % (j + 1)
        tmp = local[j]
        local[j] = local[kk]
        local[kk] = tmp
    return local


def _cuda_mi_from_counts_kernel_factory():
    """numba.cuda kernel: MI of every (column k, y-vector p) from the integer joint histograms, ON the
    GPU -- one thread per (k, p). Reproduces ``_mi_from_counts_cpu`` (use_su=False) reduction order
    EXACTLY (prob_x = fx/n int/int division; jf = jc/n; mi += jf*log(jf/(prob_x*prob_y))), so the result
    matches the CPU entropy to fp round-off (selection-equivalent -- the FE perf bar). This is the last
    CPU step of the noise gate; keeping it on-device lets the whole (orig + perms) gate run from resident
    counts with only the small (P, K) MI matrix coming back. ``ref_mi`` masks perm columns whose original
    MI is <= 0 (exactly the CPU loop's perm-skip)."""
    if not _CUDA_AVAIL:
        return None

    @_nb_cuda.jit
    def _kernel(
        counts_flat,   # (P*total_size,) int64 -- per-(p) per-column joint histograms
        col_offsets,   # (K,) int64
        nbins_col,     # (K,) int32
        freqs_y,       # (K_y,) float64
        n,
        K_y,
        ref_mi,        # (K,) float64 -- original_mi; perm columns with ref_mi<=0 are skipped (-> 0)
        total_size,
        P,
        out_mi,        # (P, K) float64 output
    ):
        K = nbins_col.shape[0]
        tid = _nb_cuda.blockIdx.x * _nb_cuda.blockDim.x + _nb_cuda.threadIdx.x
        if tid >= K * P:
            return
        p = tid // K
        k = tid - p * K
        if p > 0 and ref_mi[k] <= 0.0:
            out_mi[p, k] = 0.0
            return
        nb_k = nbins_col[k]
        off = col_offsets[k] + p * total_size
        inv_n = 1.0 / n
        mi = 0.0
        for i in range(nb_k):
            base = off + i * K_y
            fx = 0
            for j in range(K_y):
                fx += counts_flat[base + j]
            if fx == 0:
                continue
            prob_x = fx / n
            for j in range(K_y):
                jc = counts_flat[base + j]
                if jc != 0:
                    jf = jc * inv_n
                    mi += jf * math.log(jf / (prob_x * freqs_y[j]))
        out_mi[p, k] = mi

    return _kernel


def _gate_from_mi(
    original_mi: np.ndarray,
    perm_mis: list,          # list of (K,) float64 arrays, one per permutation
    npermutations: int,
    min_nonzero_confidence: float,
) -> np.ndarray:
    """Apply the EXACT noise-gate rejection rule of the CPU kernel given the
    original MIs and the per-permutation MIs. ``perm_mis[i][k]`` is column k's MI
    against the i-th shuffled y (or any value when ``original_mi[k] <= 0``)."""
    K = original_mi.shape[0]
    fe_mi = np.zeros(K, dtype=np.float64)
    if npermutations <= 0:
        for k in range(K):
            fe_mi[k] = original_mi[k]
        return fe_mi

    max_failed = int(npermutations * (1.0 - min_nonzero_confidence))
    if max_failed <= 1:
        max_failed = 1

    nfailed = np.zeros(K, dtype=np.int64)
    for i in range(npermutations):
        pm = perm_mis[i]
        for k in range(K):
            if original_mi[k] <= 0.0:
                continue
            if pm[k] >= original_mi[k]:
                nfailed[k] += 1

    for k in range(K):
        om = original_mi[k]
        if om > 0.0 and nfailed[k] >= max_failed:
            fe_mi[k] = 0.0
        else:
            fe_mi[k] = om
    return fe_mi


# Bit-identical host LCG used to fill a WHOLE (npermutations, n) shuffle matrix in
# one njit-prange pass -- the GPU-resident kernel uploads this matrix ONCE instead
# of one cp.asarray(shuffled) per permutation. Each row i reproduces
# ``_fisher_yates_shuffle(classes_y_safe, base_seed, i)`` EXACTLY.
# NOTE: ``prange`` MUST be imported from numba (see the module-level import); with
# ``parallel=True`` a bare ``prange`` is an undefined global -> TypingError at JIT
# compile, which previously crashed the GPU-resident permutation path for every
# npermutations>0 call (nperm=0 never builds the matrix, so it appeared to "work").
@njit(nogil=True, cache=True, parallel=True)
def _build_shuffle_matrix(classes_y_safe: np.ndarray, base_seed: np.uint64, npermutations: int) -> np.ndarray:
    """``out[i, :]`` = ``classes_y_safe`` Fisher-Yates-shuffled with the per-perm LCG
    seed ``base_seed*2654435761 + (i+1)`` -- bit-identical to the CPU kernel's stream
    for every permutation, materialised as one (npermutations, n) int matrix for a
    single H2D upload."""
    ny = classes_y_safe.shape[0]
    out = np.empty((npermutations, ny), dtype=classes_y_safe.dtype)
    for i in prange(npermutations):
        for t in range(ny):
            out[i, t] = classes_y_safe[t]
        state = np.uint64(base_seed) * np.uint64(2654435761) + np.uint64(i + 1)
        for j in range(ny - 1, 0, -1):
            state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            kk = int(state >> np.uint64(33)) % (j + 1)
            tmp = out[i, j]
            out[i, j] = out[i, kk]
            out[i, kk] = tmp
    return out


# ---------------------------------------------------------------------------
# numba.cuda kernel factories
# ---------------------------------------------------------------------------


def _cuda_hist_kernel_factory():
    """Build the numba.cuda joint-histogram kernel lazily (avoid CUDA driver
    lookup at import on a CPU-only host). One block per candidate column; threads
    stride over rows and atomically populate a per-column joint histogram in
    GLOBAL memory (no shared-mem cap on nbins -> works for any column cardinality).
    """
    if not _CUDA_AVAIL:
        return None

    @_nb_cuda.jit
    def _kernel(
        disc_2d,        # (n, K) int32
        col_offsets,    # (K,) int64 -- start offset of column k in counts_flat
        nbins_col,      # (K,) int32
        y_codes,        # (n,) int32 -- the (shuffled) target codes
        counts_flat,    # (total_size,) int64 -- output, zeroed by host
        n,
        K_y,
    ):
        k = _nb_cuda.blockIdx.x
        if k >= disc_2d.shape[1]:
            return
        off = col_offsets[k]
        tid = _nb_cuda.threadIdx.x
        nthreads = _nb_cuda.blockDim.x
        for r in range(tid, n, nthreads):
            cx = disc_2d[r, k]
            cy = y_codes[r]
            # global index: off + cx*K_y + cy
            _nb_cuda.atomic.add(counts_flat, off + cx * K_y + cy, 1)

    return _kernel


def _cuda_hist_kernel_batched_factory():
    """Batched joint-histogram kernel: ALL P=(npermutations+1) y-vectors in ONE launch. grid (K, P);
    block (k, p) bins column k against y_all[p] into counts_flat[p*total_size + off_k ...]. Lets the whole
    noise gate run from resident counts (paired with the GPU MI kernel) so only the (P, K) MI matrix
    leaves the device. Counts are integer + commutative -> identical to the per-perm kernel."""
    if not _CUDA_AVAIL:
        return None

    @_nb_cuda.jit
    def _kernel_b(disc_2d, col_offsets, y_all, counts_flat, n, K_y, total_size):
        # bench-attempt-rejected (2026-06-21): column-major (K,n) coalescing of this batched hist was
        # 0.79-1.05x (LOSS) -- the host transpose-copy cost outweighs any coalescing gain and the kernel
        # is not actually bandwidth-coalescing-bound here (consistent with the radix/noise-gate coalescing
        # washes). Kept row-major (n, K).
        k = _nb_cuda.blockIdx.x
        p = _nb_cuda.blockIdx.y
        if k >= disc_2d.shape[1]:
            return
        off = col_offsets[k] + p * total_size
        tid = _nb_cuda.threadIdx.x
        nthreads = _nb_cuda.blockDim.x
        for r in range(tid, n, nthreads):
            cx = disc_2d[r, k]
            cy = y_all[p, r]
            _nb_cuda.atomic.add(counts_flat, off + cx * K_y + cy, 1)

    return _kernel_b


def _cuda_hist_kernel_batched_shared_factory():
    """Shared-memory PRIVATIZED batched joint-histogram. Each block (k, p) accumulates column k's joint
    histogram for y_all[p] in DYNAMIC SHARED memory (shared atomics ~20x faster than global AND no
    cross-block contention), then flushes its disjoint ``counts_flat[p*total_size+off_k : ...]`` slice with
    PLAIN writes (one block owns that slice). Fixes the global-atomic-contention the elevated nvprof
    metrics exposed (atomic_transactions_per_request ~14.6, ~430M L2 atomic transactions on the
    global-atomic kernel). Counts are integer + commutative -> BIT-IDENTICAL. The caller gates on the
    per-column histogram (nb_k*K_y int32) fitting the shared budget; else the global-atomic kernel.

    bench-attempt-rejected (2026-06-21): column-major (K,n) coalescing of THIS shared kernel is 0.89x
    (loss) even with the transpose amortised over all P perms -- post-privatization the kernel is
    SHARED-ATOMIC-bound (n counts/block), not disc-load-bound (despite gld_efficiency ~13.8%), so fixing
    the load layout doesn't help and the transpose pass only adds work. Kept row-major (n, K)."""
    if not _CUDA_AVAIL:
        return None

    from numba import int32 as _i32

    @_nb_cuda.jit
    def _kernel_bs(disc_2d, col_offsets, nbins_col, y_all, counts_flat, n, K_y, total_size):
        k = _nb_cuda.blockIdx.x
        p = _nb_cuda.blockIdx.y
        if k >= disc_2d.shape[1]:
            return
        nb_k = nbins_col[k]
        hsize = nb_k * K_y
        sh = _nb_cuda.shared.array(0, _i32)  # dynamic; bytes set by launch config
        tid = _nb_cuda.threadIdx.x
        nthreads = _nb_cuda.blockDim.x
        i = tid
        while i < hsize:
            sh[i] = 0
            i += nthreads
        _nb_cuda.syncthreads()
        for r in range(tid, n, nthreads):
            cx = disc_2d[r, k]
            cy = y_all[p, r]
            _nb_cuda.atomic.add(sh, cx * K_y + cy, 1)
        _nb_cuda.syncthreads()
        off = col_offsets[k] + p * total_size
        i = tid
        while i < hsize:
            counts_flat[off + i] = sh[i]
            i += nthreads

    return _kernel_bs


_CUDA_SHARED_BYTES_PER_BLOCK: int = -1


def _cuda_shared_mem_per_block() -> int:
    """Device shared-memory-per-block budget in bytes (queried once, cached); 0 if unavailable."""
    global _CUDA_SHARED_BYTES_PER_BLOCK
    if _CUDA_SHARED_BYTES_PER_BLOCK >= 0:
        return _CUDA_SHARED_BYTES_PER_BLOCK
    val = 0
    try:
        val = int(getattr(_nb_cuda.get_current_device(), "MAX_SHARED_MEMORY_PER_BLOCK", 0)) or 0
    except Exception:
        val = 0
    _CUDA_SHARED_BYTES_PER_BLOCK = val
    return val
