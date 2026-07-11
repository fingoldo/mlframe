"""GPU variants of :func:`batch_pair_mi_prange` + a size-aware dispatcher.

Three backends are exposed:

* ``batch_pair_mi_njit_prange`` -- re-export of the CPU njit kernel from
  ``info_theory``. The reference implementation; numerical baseline.
* ``batch_pair_mi_cuda`` -- ``numba.cuda`` JIT kernel. One CUDA block per pair,
  threads inside the block share a joint-class histogram via shared memory
  before a single thread runs the MI reduction.
* ``batch_pair_mi_cupy`` -- pure CuPy implementation. One vectorised sweep
  per pair using ``cupy.bincount`` for the joint histogram + a manual MI
  reduction. Trades GPU-occupancy for code simplicity; benefits more on
  high-bin combinations where CuPy's elementwise kernels saturate the SMs.

The dispatcher (:func:`dispatch_batch_pair_mi`) picks the fastest backend
given the input shape and the available hardware. Crossover thresholds
were measured on a single benchmark machine (CPU: 4 physical cores, GPU:
GTX 1050 Ti compute_capability=6.1, 4GB VRAM); callers can override via the
``force_backend`` knob to lock a specific implementation.

Numerical equivalence vs the ``merge_vars + compute_mi_from_classes`` legacy
path is verified by ``tests/feature_selection/test_batch_pair_mi_prange.py``
(CPU) and ``test_batch_pair_mi_gpu.py`` (GPU variants when CUDA is
available; auto-skip otherwise).
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numba
import numpy as np

# Re-export CPU baseline for callers who want a single import point.
from .info_theory import batch_pair_mi_prange as batch_pair_mi_njit_prange

logger = logging.getLogger(__name__)

# Optional GPU deps. The dispatcher gracefully falls back to the CPU kernel
# when either is missing.
#
# ``_CUDA_AVAIL`` consults the central pyutilz probe (``is_cuda_available``)
# rather than re-running ``numba.cuda.is_available()`` inline. The numba
# module itself is still imported here because the ``@_nb_cuda.jit``
# decorator at ``_cuda_kernel_factory`` needs a binding; the probe is just
# how we DECIDE whether to take the GPU path.
try:
    from numba import cuda as _nb_cuda
except Exception:
    _nb_cuda = None

try:
    from pyutilz.core.pythonlib import is_cuda_available as _pyutilz_is_cuda_available
    _CUDA_AVAIL = _pyutilz_is_cuda_available()
except Exception:
    # Fallback to inline numba probe if pyutilz is not importable for some reason.
    try:
        _CUDA_AVAIL = bool(getattr(_nb_cuda, "is_available", lambda: False)()) if _nb_cuda is not None else False
    except Exception:
        _CUDA_AVAIL = False

# Require numba.cuda to actually compile+launch a kernel (not just device presence) so a
# cudatoolkit/NVVM mismatch routes to cupy/CPU instead of raising NvvmSupportError mid-dispatch.
from ._internals import numba_cuda_can_compile as _numba_cuda_can_compile
_CUDA_AVAIL = _CUDA_AVAIL and _numba_cuda_can_compile()

try:
    import cupy as _cp
    _CUPY_AVAIL = True
except Exception:
    _cp = None
    _CUPY_AVAIL = False


# ---------------------------------------------------------------------------
# numba.cuda variant
# ---------------------------------------------------------------------------

# Per-pair CUDA kernel: one block per pair, threads in the block populate a
# shared-memory joint-class histogram, then one thread reduces it to MI.
#
# The shared-memory footprint per block is:
#     MAX_JOINT_BINS_CUDA * MAX_Y_BINS_CUDA * 8  +  MAX_JOINT_BINS_CUDA * 8
# i.e. (joint_card * n_classes_y * int64) for the histogram, plus joint_card *
# int64 for the marginal freqs.
#
# We derive the safe caps from the live device's per-block shared-memory
# budget via ``pyutilz.system.gpu_dispatch.get_shared_mem_budget_per_block``.
# That probe returns the correct per-block ceiling for every shipped compute
# capability (cc 6.x = 48 KB, cc 7.0 Volta = 96 KB opt-in, cc 8.0 A100 = 163 KB
# opt-in, cc 9.0 Hopper = 227 KB opt-in; cf. pyutilz commit 8371ce1 for the
# full table). The cap is locked to MAX_Y_BINS_CUDA=16 (sufficient for 16-class
# multiclass targets) and MAX_JOINT_BINS_CUDA is the largest power-of-2 that
# fits the remaining budget.
#
# Fallback: if pyutilz is unavailable OR the probe fails, we use the cc 6.x
# safe defaults (128, 16) -> 17 KB.

MAX_Y_BINS_CUDA = 16  # supports up to 16-class multiclass targets


def _derive_max_joint_bins(max_y_bins: int) -> int:
    """Pick MAX_JOINT_BINS_CUDA from the live device's per-block shared-memory
    budget (via pyutilz). Solves
        joint * max_y_bins * 8 + joint * 8 <= budget
    for ``joint``, then rounds DOWN to the nearest power of 2 for kernel-launch
    safety. Caps at 1024 (kernel design guarantees correctness only up to
    that joint cardinality). Falls back to 128 (cc 6.x safe) on probe failure.
    """
    try:
        from pyutilz.system.gpu_dispatch import (
            gpu_capability_summary,
            get_shared_mem_budget_per_block,
        )
        summary = gpu_capability_summary(0) if _CUDA_AVAIL else None
        if summary is None:
            return 128
        budget = get_shared_mem_budget_per_block(
            summary["cc_major"], summary["cc_minor"], allow_opt_in=False,
        )
        # Solve joint * (max_y_bins + 1) * 8 <= budget.
        raw = budget // ((max_y_bins + 1) * 8)
        if raw < 16:
            return 16
        # Round down to nearest power of 2 to keep CUDA shared-mem alignment
        # predictable.
        joint = 1
        while joint * 2 <= raw and joint < 1024:
            joint *= 2
        return joint
    except Exception:
        return 128


MAX_JOINT_BINS_CUDA = _derive_max_joint_bins(MAX_Y_BINS_CUDA)


def _cuda_kernel_factory():
    """Build the CUDA kernel lazily so importing this module on a CPU-only
    host doesn't trigger numba.cuda's CUDA driver lookup (which can raise on
    machines without the CUDA toolkit installed even when numba is present).
    """
    if not _CUDA_AVAIL:
        return None

    @_nb_cuda.jit
    def _kernel(
        factors_data,  # (n_samples, n_features) int32
        pair_a,  # (n_pairs,) int64
        pair_b,  # (n_pairs,) int64
        nbins,  # (n_features,) int32
        classes_y,  # (n_samples,) int32
        freqs_y,  # (n_classes_y,) float64
        mi_out,  # (n_pairs,) float64
        n_samples,
        n_classes_y,
    ):
        """numba.cuda device kernel: one block per (feature-a, feature-b) pair, builds a shared-memory joint histogram against ``classes_y`` and writes each pair's MI into ``mi_out``."""
        # One block per pair. Threads within the block stride over n_samples
        # to populate a shared-memory joint-class histogram. Then thread 0
        # reduces to MI.
        p = _nb_cuda.blockIdx.x
        if p >= pair_a.shape[0]:
            return

        a = pair_a[p]
        b = pair_b[p]
        nb_a = nbins[a]
        nb_b = nbins[b]
        joint_card = nb_a * nb_b

        # Shared-memory joint histogram + freqs_x. Sized at compile time to
        # the worst-case bound; actual usage may be smaller.
        # Layout: joint_counts[joint_card * n_classes_y] then freqs_x[joint_card].
        sm_hist = _nb_cuda.shared.array(
            shape=(MAX_JOINT_BINS_CUDA, MAX_Y_BINS_CUDA), dtype=np.int64,
        )
        sm_fx = _nb_cuda.shared.array(shape=(MAX_JOINT_BINS_CUDA,), dtype=np.int64)

        tid = _nb_cuda.threadIdx.x
        nthreads = _nb_cuda.blockDim.x

        # Zero the shared-memory histogram.
        for i in range(tid, joint_card, nthreads):
            sm_fx[i] = 0
            for j in range(n_classes_y):
                sm_hist[i, j] = 0
        _nb_cuda.syncthreads()

        # Populate histogram: each thread strides over n_samples.
        for i in range(tid, n_samples, nthreads):
            va = factors_data[i, a]
            vb = factors_data[i, b]
            cls_x = va * nb_b + vb
            cls_y = classes_y[i]
            _nb_cuda.atomic.add(sm_hist, (cls_x, cls_y), 1)
            _nb_cuda.atomic.add(sm_fx, cls_x, 1)
        _nb_cuda.syncthreads()

        # Reduce on thread 0. MI = sum_{i,j} jf * log(jf / (px * py)).
        if tid == 0:
            total = 0.0
            inv_n = 1.0 / n_samples
            for i in range(joint_card):
                fx = sm_fx[i]
                if fx == 0:
                    continue
                prob_x = fx * inv_n
                for j in range(n_classes_y):
                    jc = sm_hist[i, j]
                    if jc == 0:
                        continue
                    jf = jc * inv_n
                    prob_y = freqs_y[j]
                    if prob_y > 0.0:
                        total += jf * math.log(jf / (prob_x * prob_y))
            mi_out[p] = total

    return _kernel


_CUDA_HIST_KERNEL: Any = None  # lazy-bound on first call, separate from _CUDA_KERNEL (different signature)


def _cuda_hist_kernel_factory():
    """Row-chunk-capable variant of :func:`_cuda_kernel_factory`'s kernel: instead of reducing straight to
    MI inside one launch (which needs the WHOLE ``factors_data`` resident on-device at once), this kernel
    only ACCUMULATES joint/marginal counts into a device-persistent GLOBAL-memory buffer via atomic adds.
    Calling it once per ROW-CHUNK of ``factors_data`` (each chunk small enough to fit free VRAM) and
    finalizing the MI reduction afterward on the accumulated counts is mathematically identical to the
    single-shot kernel: joint/marginal counts are a plain sum over rows, and integer-addition is
    order-independent, so the finalize step (same reduction formula, same iteration order) is bit-identical
    to :func:`batch_pair_mi_cuda`'s single-launch result -- see :func:`batch_pair_mi_cuda_row_chunked`.
    """
    if not _CUDA_AVAIL:
        return None

    @_nb_cuda.jit
    def _hist_kernel(
        factors_data,  # (chunk_rows, n_features) int32 -- ONE row-chunk slice, not the whole array
        pair_a,  # (n_pairs,) int64
        pair_b,  # (n_pairs,) int64
        nbins,  # (n_features,) int32
        classes_y,  # (chunk_rows,) int32 -- ONE row-chunk slice
        joint_counts,  # (n_pairs, max_joint, n_classes_y) int64 device-persistent ACCUMULATOR
        fx_counts,  # (n_pairs, max_joint) int64 device-persistent ACCUMULATOR
        chunk_rows,
    ):
        p = _nb_cuda.blockIdx.x
        if p >= pair_a.shape[0]:
            return
        a = pair_a[p]
        b = pair_b[p]
        nb_b = nbins[b]
        tid = _nb_cuda.threadIdx.x
        nthreads = _nb_cuda.blockDim.x
        for i in range(tid, chunk_rows, nthreads):
            va = factors_data[i, a]
            vb = factors_data[i, b]
            cls_x = va * nb_b + vb
            cls_y = classes_y[i]
            _nb_cuda.atomic.add(joint_counts, (p, cls_x, cls_y), 1)
            _nb_cuda.atomic.add(fx_counts, (p, cls_x), 1)

    return _hist_kernel


_CUDA_HIST_KERNEL_SHARED: Any = None  # lazy-bound, dynamic-shared-memory variant of _hist_kernel


def _cuda_hist_kernel_shared_factory():
    """Shared-memory-staged variant of :func:`_cuda_hist_kernel_factory`'s kernel.

    Found live (2026-07-10 wellbore 100k-row ncu profiling): the plain global-atomics kernel above
    achieved 99.2% occupancy but only 2.5% compute throughput -- ncu's Warp State Statistics attributed
    93.5% of the 703.7-cycle average stall between issued instructions to "LG Throttle" (the L1
    instruction queue for local/global memory operations staying full), with ncu's own recommendation
    reading "avoid redundant global memory accesses... combine multiple lower-width memory operations
    into fewer wider memory operations". Each thread issues 2 GLOBAL atomic adds PER ROW it processes
    (up to chunk_rows per block, e.g. 79,237 at the profiled shape) -- :func:`_cuda_kernel_factory`'s
    original non-chunked kernel already avoids exactly this by staging counts in per-block SHARED memory
    (fast, on-chip) and touching global memory only once per histogram cell. This kernel applies the same
    staging to the row-chunked path: a per-block dynamic shared-memory buffer (int32, safe up to a
    ~2.1e9-row chunk -- ``_choose_row_chunk_rows`` caps chunks at 5,000,000) accumulates the block's
    contribution, then ONE atomic-add-per-cell flushes it to the persistent global int64 accumulator --
    cutting global atomic traffic from ``O(chunk_rows)`` to ``O(max_joint * n_classes_y)`` per block
    (~17x fewer at the profiled shape: 441*20=8,820 cells vs up to 158,474 row-driven atomics).

    Numba.cuda's dynamic shared memory is one flat buffer per kernel (multiple ``cuda.shared.array(0,
    ...)`` calls alias the same base address) -- laid out here as ``[0:max_joint*n_classes_y)`` for the
    joint histogram followed by ``[max_joint*n_classes_y : +max_joint)`` for the marginal counts, with
    manual index arithmetic (``cls_x * n_classes_y + cls_y``) standing in for what would otherwise be a
    2D shared array. Callers must pass ``shared_mem_bytes=(max_joint*n_classes_y + max_joint) * 4`` in
    the launch config's 4th slot; see :func:`_hist_kernel_shared_fits_budget` for the gate that decides
    whether this variant applies (falls back to the global-only kernel above when it doesn't).
    """
    if not _CUDA_AVAIL:
        return None

    @_nb_cuda.jit
    def _hist_kernel_shared(
        factors_data,  # (chunk_rows, n_features) int32 -- ONE row-chunk slice, not the whole array
        pair_a,  # (n_pairs,) int64
        pair_b,  # (n_pairs,) int64
        nbins,  # (n_features,) int32
        classes_y,  # (chunk_rows,) int32 -- ONE row-chunk slice
        joint_counts,  # (n_pairs, max_joint, n_classes_y) int64 device-persistent ACCUMULATOR
        fx_counts,  # (n_pairs, max_joint) int64 device-persistent ACCUMULATOR
        chunk_rows,
        max_joint,
        n_classes_y,
    ):
        sh = _nb_cuda.shared.array(0, dtype=numba.int32)
        n_joint_cells = max_joint * n_classes_y
        total_cells = n_joint_cells + max_joint
        tid = _nb_cuda.threadIdx.x
        nthreads = _nb_cuda.blockDim.x

        for i in range(tid, total_cells, nthreads):
            sh[i] = 0
        _nb_cuda.syncthreads()

        p = _nb_cuda.blockIdx.x
        if p < pair_a.shape[0]:
            a = pair_a[p]
            b = pair_b[p]
            nb_b = nbins[b]
            for i in range(tid, chunk_rows, nthreads):
                va = factors_data[i, a]
                vb = factors_data[i, b]
                cls_x = va * nb_b + vb
                cls_y = classes_y[i]
                _nb_cuda.atomic.add(sh, cls_x * n_classes_y + cls_y, 1)
                _nb_cuda.atomic.add(sh, n_joint_cells + cls_x, 1)
        _nb_cuda.syncthreads()

        if p < pair_a.shape[0]:
            for i in range(tid, n_joint_cells, nthreads):
                v = sh[i]
                if v != 0:
                    jc = i // n_classes_y
                    jy = i - jc * n_classes_y
                    _nb_cuda.atomic.add(joint_counts, (p, jc, jy), v)
            for i in range(tid, max_joint, nthreads):
                v = sh[n_joint_cells + i]
                if v != 0:
                    _nb_cuda.atomic.add(fx_counts, (p, i), v)

    return _hist_kernel_shared


def _hist_kernel_shared_fits_budget(max_joint: int, n_classes_y: int) -> int:
    """Bytes needed for :func:`_cuda_hist_kernel_shared_factory`'s dynamic shared buffer, or 0 if it
    would exceed the device's per-block shared-memory budget (caller falls back to the global-only
    kernel in that case -- gating a fast path on its safe condition rather than risking a launch failure
    or, worse on some drivers, a silently truncated allocation)."""
    if not _CUDA_AVAIL:
        return 0
    needed = (max_joint * n_classes_y + max_joint) * 4  # int32
    try:
        budget = _nb_cuda.get_current_device().MAX_SHARED_MEMORY_PER_BLOCK
    except Exception:
        budget = 49152  # cc 6.x+ safe default without opt-in
    # Leave headroom for the driver's own per-block reserved shared memory (a few hundred bytes,
    # observed ~1KB on cc 8.9 via ncu's "Driver Shared Memory Per Block").
    return needed if needed <= budget - 2048 else 0


@numba.njit(cache=True)
def _mi_from_joint_counts(joint_counts: np.ndarray, fx_counts: np.ndarray, freqs_y: np.ndarray, n_samples: int, joint_cards: np.ndarray) -> np.ndarray:
    """Host-side MI finalize from accumulated joint/marginal counts -- the EXACT reduction formula and
    iteration order :func:`_cuda_kernel_factory`'s single-launch kernel uses (``sum jf * log(jf/(px*py))``
    over ``i in range(joint_card)`` then ``j in range(n_classes_y)``), so the result is bit-identical to
    the non-chunked path. njit'd: at production pair counts (tens of thousands) x max_joint (up to 128) x
    n_classes_y (up to 16) this loop can reach ~1e8 iterations, which would be a real wall-clock cost in
    plain Python -- the accumulated counts array is small enough to upload/download cheaply, but reducing
    it still needs compiled code."""
    n_pairs = joint_counts.shape[0]
    n_classes_y = freqs_y.shape[0]
    inv_n = 1.0 / n_samples
    mi_out = np.zeros(n_pairs, dtype=np.float64)
    for p in range(n_pairs):
        joint_card = int(joint_cards[p])
        total = 0.0
        for i in range(joint_card):
            fx = fx_counts[p, i]
            if fx == 0:
                continue
            prob_x = fx * inv_n
            for j in range(n_classes_y):
                jc = joint_counts[p, i, j]
                if jc == 0:
                    continue
                jf = jc * inv_n
                prob_y = freqs_y[j]
                if prob_y > 0.0:
                    total += jf * math.log(jf / (prob_x * prob_y))
        mi_out[p] = total
    return mi_out


_ZERO_FILL_KERNEL: Any = None  # lazy-bound numba.cuda fallback for hosts without cupy


def _zero_fill_kernel_factory():
    if not _CUDA_AVAIL:
        return None

    @_nb_cuda.jit
    def _kernel(flat_arr, n):
        idx = _nb_cuda.grid(1)
        stride = _nb_cuda.gridsize(1)
        for i in range(idx, n, stride):
            flat_arr[i] = 0

    return _kernel


def _new_zeroed_device_array(shape: tuple, dtype) -> Any:
    """Allocate a zero-initialised device array WITHOUT staging a same-size host ``np.zeros`` buffer
    through PCIe first.

    Found live (2026-07-10 wellbore 100k-row nsys profiling): the row-chunked kernel's pair-subchunk
    accumulator was zeroed via ``to_device(np.zeros(shape), to=d_arr)`` -- allocating a FULL host-side
    zeros array the same size as the device accumulator (up to ~400MB at production shape) and shipping
    it over PCIe just to zero a buffer whose content doesn't matter. nsys showed ``cuMemcpyDtoH_v2``
    (the accumulator's later readback) averaging 266ms/call with a 4.16s max against an isolated ~1-30ms
    baseline -- consistent with the same WDDM VRAM-oversubscription paging hazard already worked around
    elsewhere in this module; the redundant zero-upload doubles transient memory pressure at exactly the
    moment (pair-subchunk allocation) that pressure is most likely to trigger it. cupy's ``cp.zeros``
    zero-fills device-side (``cudaMemsetAsync``, no host transfer); on hosts without cupy, fall back to a
    trivial numba.cuda zero-fill kernel -- still no host round-trip, just one extra (cheap) launch.
    """
    if _CUPY_AVAIL:
        return _nb_cuda.as_cuda_array(_cp.zeros(shape, dtype=dtype))
    global _ZERO_FILL_KERNEL
    if _ZERO_FILL_KERNEL is None:
        _ZERO_FILL_KERNEL = _zero_fill_kernel_factory()
    arr = _nb_cuda.device_array(shape, dtype=dtype)
    n = int(np.prod(shape))
    flat = arr.reshape(n)
    threads = 256
    blocks = min(1024, -(-n // threads))
    _ZERO_FILL_KERNEL[blocks, threads](flat, n)
    return arr


# Sane ceiling on TOTAL row-chunk x pair-subchunk kernel launches for one batch_pair_mi_cuda_row_chunked
# call. Both chunk-size choosers independently clamp to a safe-but-degenerate floor when free VRAM is
# itself near-zero (row_chunk_rows>=1000, pair_subchunk_rows>=1) -- individually correct (never over-
# allocates), but their PRODUCT can reach tens of millions of launches, each paying real upload+dispatch
# overhead. Past this ceiling, the row-chunked path is no longer worth attempting; see its use below.
_MAX_REASONABLE_ROW_CHUNK_LAUNCHES = 2000


def _choose_row_chunk_rows(n_cols: int, free_bytes: int, budget_frac: float = 0.3) -> int:
    """How many rows of ``factors_data`` (``n_cols`` columns, int32) fit per GPU launch, within a
    DEDICATED VRAM budget fraction independent of pair count -- the pair dimension is bounded
    separately by :func:`_choose_pair_subchunk_rows` (see its docstring for why the two must NOT
    share one combined budget calculation). Clamped to >= 1000 rows (a degenerate 1-row chunk would
    need thousands of tiny launches) and to the RAM-based fallback ceiling used elsewhere in this
    module for consistency.
    """
    budget = max(0, int(free_bytes * budget_frac))
    per_row_bytes = max(1, n_cols * 4 + 4)  # factors_data row (int32) + classes_y entry (int32)
    rows = budget // per_row_bytes
    return int(np.clip(rows, 1000, 5_000_000))


def _choose_pair_subchunk_rows(max_joint: int, n_classes_y: int, free_bytes: int, budget_frac: float = 0.2) -> int:
    """How many pairs' worth of ``(max_joint, n_classes_y)`` histogram accumulator fit in a dedicated
    VRAM budget fraction.

    Found live (2026-07-10 wellbore 100k-row profiling): the ORIGINAL row-chunked design sized the
    accumulator by the FULL ``n_pairs`` passed to one call (``n_pairs * max_joint * n_classes_y * 8``
    bytes) and only chunked ROWS -- but a wide candidate pool (86,736 pairs, max_joint=441,
    n_classes_y=20) needs a ~6 GB accumulator, bigger than the ENTIRE 4 GB card, regardless of how
    small the row-chunk is. The allocation likely silently succeeded via WDDM over-subscription (the
    exact hazard this whole guard chain exists to avoid) and then thrashed: cProfile showed 165
    ``to_device`` calls averaging 4.6s each (an isolated microbench on the same host, same shape,
    shows ~1-30ms) -- 772s of a 1633s wall, the single largest hotspot in the run. The fix chunks
    PAIRS too: :func:`batch_pair_mi_cuda_row_chunked` now loops pair-subchunks (each with its own
    correctly-bounded accumulator) as the OUTER loop, row-chunks as the inner. Clamped to >= 1 pair
    (a single pair's accumulator must always fit; if genuinely one pair's ``max_joint*n_classes_y``
    exceeds the budget the caller's own guard already routed away from GPU) and to 2M as a sane
    ceiling.
    """
    budget = max(0, int(free_bytes * budget_frac))
    per_pair_bytes = max(1, max_joint * n_classes_y * 8 + max_joint * 8)
    n = budget // per_pair_bytes
    return int(np.clip(n, 1, 2_000_000))


def batch_pair_mi_cuda_row_chunked(
    factors_data: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    threads_per_block: int = 128,
) -> np.ndarray:
    """Row-chunked variant of :func:`batch_pair_mi_cuda` for when the FULL ``factors_data`` upload would not
    safely fit in free VRAM. Uploads ``factors_data`` in row-chunks small enough to fit, accumulating joint/
    marginal histogram counts into a persistent device buffer across chunks (counts are additive over rows),
    then finalizes MI on the small accumulated histogram. Bit-identical to :func:`batch_pair_mi_cuda` (same
    reduction formula/order; integer-count accumulation is order-independent) -- see
    ``tests/feature_selection/gpu/test_batch_pair_mi_row_chunked.py``.

    Preserves the GPU speed win at production row counts instead of falling all the way back to the CPU
    njit kernel, which was the prior behaviour whenever the full-array VRAM guard rejected a launch.

    Raises ``RuntimeError``/``ValueError`` under the same preconditions as :func:`batch_pair_mi_cuda`
    (shape guards, shared-mem-derived joint-cardinality cap) -- callers should catch broadly, same as the
    non-chunked kernel.
    """
    global _CUDA_HIST_KERNEL, _CUDA_HIST_KERNEL_SHARED
    if not _CUDA_AVAIL:
        raise RuntimeError("numba.cuda is not available on this host")
    if _CUDA_HIST_KERNEL is None:
        _CUDA_HIST_KERNEL = _cuda_hist_kernel_factory()
        if _CUDA_HIST_KERNEL is None:
            raise RuntimeError("numba.cuda hist-kernel factory failed to build")

    n_pairs = int(pair_a.shape[0])
    if n_pairs == 0:
        return np.empty(0, dtype=np.float64)

    # NOTE: unlike :func:`batch_pair_mi_cuda`, this kernel accumulates into GLOBAL device memory
    # (``d_joint``/``d_fx`` below), not per-block SHARED memory -- so it is NOT bound by
    # ``MAX_Y_BINS_CUDA``/``MAX_JOINT_BINS_CUDA`` (those caps exist solely because the non-chunked
    # kernel's shared-memory histogram is sized to them at compile time; a 48KB-per-block shared-mem
    # budget does not apply to a dynamically-sized global-memory array). The accumulator's actual size
    # (``n_pairs * max_joint * n_classes_y * 8`` bytes) is instead bounded by VRAM via
    # :func:`_choose_row_chunk_rows`, which subtracts it from the row-chunk budget -- a genuine
    # hardware constraint, not an arbitrary compile-time one. Found live (2026-07-10 wellbore 3M-row
    # run): a copy-pasted ``n_classes_y > MAX_Y_BINS_CUDA=16`` check rejected a real n_classes_y=20
    # discretized-target pair-MI call, forcing an unnecessary fall-through to slow CPU njit even though
    # the row-chunked kernel could handle it fine.
    n_classes_y = int(freqs_y.shape[0])

    joint_cards = np.empty(n_pairs, dtype=np.int64)
    max_joint = 0
    for idx in range(n_pairs):
        a, b = int(pair_a[idx]), int(pair_b[idx])
        nb_a, nb_b = int(nbins[a]), int(nbins[b])
        if nb_a < 1 or nb_b < 1:
            raise ValueError(f"degenerate pair ({a}, {b}): nbins=({nb_a}, {nb_b}); at least one column has zero cardinality (skip the pair host-side)")
        jc = nb_a * nb_b
        joint_cards[idx] = jc
        if jc > max_joint:
            max_joint = jc

    if classes_y.size > 0:
        cy_max = int(classes_y.max())
        cy_min = int(classes_y.min())
        if cy_max >= n_classes_y or cy_min < 0:
            raise ValueError(f"classes_y values must be in [0, n_classes_y={n_classes_y}); got [min={cy_min}, max={cy_max}]")
    if factors_data.size > 0:
        fd_min = int(factors_data.min())
        if fd_min < 0:
            raise ValueError(f"factors_data must be non-negative; got min={fd_min} (merge_vars output should be >= 0 by contract)")

    n_samples = int(factors_data.shape[0])
    n_cols = int(factors_data.shape[1])

    try:
        import cupy as cp

        free_b, _total_b = cp.cuda.runtime.memGetInfo()
    except Exception:
        free_b = 512 * 1024 * 1024  # conservative fallback if the probe is unavailable

    row_chunk_rows = _choose_row_chunk_rows(n_cols, free_b)
    pair_subchunk_rows = _choose_pair_subchunk_rows(max_joint, n_classes_y, free_b)
    n_row_chunks = -(-n_samples // row_chunk_rows)
    n_pair_subchunks = -(-n_pairs // pair_subchunk_rows)
    total_launches = n_row_chunks * n_pair_subchunks
    logger.info(
        "batch_pair_mi_cuda_row_chunked: n_samples=%d n_cols=%d n_pairs=%d n_classes_y=%d max_joint=%d "
        "-> row_chunk_rows=%d (%d row-chunk(s)) x pair_subchunk_rows=%d (%d pair-subchunk(s)) = %d launch(es), "
        "free_vram=%.2fGB",
        n_samples, n_cols, n_pairs, n_classes_y, max_joint, row_chunk_rows, n_row_chunks,
        pair_subchunk_rows, n_pair_subchunks, total_launches, free_b / 1024**3,
    )
    if total_launches > _MAX_REASONABLE_ROW_CHUNK_LAUNCHES:
        # Found live (2026-07-10 wellbore 1M-row profiling): when free VRAM is itself near-zero (a
        # cupy pool cap + other resident allocations can leave <5MB), both chunk-size choosers clamp to
        # their floor (row_chunk_rows=1000, pair_subchunk_rows=1) -- individually "safe" (never over-
        # allocates), but together that's `ceil(n_samples/1000) * n_pairs` kernel launches, e.g.
        # 796 x 89,676 = ~71M at a real production shape. Per-launch overhead (upload + dispatch) alone
        # would take many HOURS for work a CPU kernel finishes in seconds. There is no useful amount of
        # GPU chunking left to try once granularity degrades this far -- raise so the caller's existing
        # fallback (``dispatch_batch_pair_mi``'s ``_try_cuda_row_chunked`` -> CPU njit) takes over
        # immediately instead of grinding through a pathological launch count.
        raise RuntimeError(
            f"batch_pair_mi_cuda_row_chunked: would need {total_launches} kernel launches "
            f"(row_chunks={n_row_chunks} x pair_subchunks={n_pair_subchunks}) at free_vram={free_b / 1024**3:.3f}GB "
            f"-- too fragmented to be worthwhile; use the CPU kernel instead",
        )

    factors_data_c = np.ascontiguousarray(factors_data, dtype=np.int32)
    classes_y_c = np.ascontiguousarray(classes_y, dtype=np.int32)
    freqs_y_c = np.ascontiguousarray(freqs_y, dtype=np.float64)
    # nbins is indexed by raw column id (not pair position) and is small -- upload once, reuse across
    # every pair-subchunk, unlike pair_a/pair_b/the accumulator which are genuinely per-subchunk.
    d_nb = _nb_cuda.to_device(np.ascontiguousarray(nbins, dtype=np.int32))

    mi_out = np.empty(n_pairs, dtype=np.float64)
    total_row_chunk_launches = 0

    # Prefer the shared-memory-staged kernel (cuts global-atomic traffic ~O(max_joint*n_classes_y) vs
    # O(chunk_rows) per block -- see _cuda_hist_kernel_shared_factory's docstring for the ncu evidence)
    # whenever the per-block histogram fits the device's shared-memory budget; fall back to the
    # global-atomics-only kernel otherwise (still correct, just slower -- e.g. a very high max_joint x
    # n_classes_y combination).
    shared_mem_bytes = _hist_kernel_shared_fits_budget(max_joint, n_classes_y)
    if shared_mem_bytes > 0:
        if _CUDA_HIST_KERNEL_SHARED is None:
            _CUDA_HIST_KERNEL_SHARED = _cuda_hist_kernel_shared_factory()

    # PAIR-subchunk is the OUTER loop, ROW-chunk the inner: the histogram accumulator
    # (pair_subchunk_rows * max_joint * n_classes_y * 8 bytes) must be sized to whichever pair-subchunk is
    # ACTIVE, bounded independently of the row dimension (see _choose_pair_subchunk_rows -- the bug this
    # fixes let the accumulator alone reach ~6GB on a 4GB card when pair-chunking wasn't done at all).
    # This does mean factors_data row-chunks get re-uploaded once per pair-subchunk; that's a bounded,
    # cheap H2D transfer (~1-30ms per chunk, measured) traded for never over-allocating the accumulator.
    for pair_start in range(0, n_pairs, pair_subchunk_rows):
        pair_end = min(pair_start + pair_subchunk_rows, n_pairs)
        sub_n_pairs = pair_end - pair_start
        d_pa = _nb_cuda.to_device(np.ascontiguousarray(pair_a[pair_start:pair_end], dtype=np.int64))
        d_pb = _nb_cuda.to_device(np.ascontiguousarray(pair_b[pair_start:pair_end], dtype=np.int64))
        d_joint = _new_zeroed_device_array((sub_n_pairs, max_joint, n_classes_y), np.int64)
        d_fx = _new_zeroed_device_array((sub_n_pairs, max_joint), np.int64)

        for row_start in range(0, n_samples, row_chunk_rows):
            row_end = min(row_start + row_chunk_rows, n_samples)
            chunk_rows = row_end - row_start
            d_data_chunk = _nb_cuda.to_device(factors_data_c[row_start:row_end])
            d_cy_chunk = _nb_cuda.to_device(classes_y_c[row_start:row_end])
            if shared_mem_bytes > 0 and _CUDA_HIST_KERNEL_SHARED is not None:
                _CUDA_HIST_KERNEL_SHARED[sub_n_pairs, threads_per_block, 0, shared_mem_bytes](
                    d_data_chunk, d_pa, d_pb, d_nb, d_cy_chunk, d_joint, d_fx, chunk_rows, max_joint, n_classes_y,
                )
            else:
                _CUDA_HIST_KERNEL[sub_n_pairs, threads_per_block](
                    d_data_chunk, d_pa, d_pb, d_nb, d_cy_chunk, d_joint, d_fx, chunk_rows,
                )
            total_row_chunk_launches += 1

        joint_host = d_joint.copy_to_host()
        fx_host = d_fx.copy_to_host()
        mi_out[pair_start:pair_end] = _mi_from_joint_counts(joint_host, fx_host, freqs_y_c, n_samples, joint_cards[pair_start:pair_end])

    logger.debug(
        "batch_pair_mi_cuda_row_chunked: n_pairs=%d n_samples=%d total_row_chunk_launches=%d (pair_subchunks=%d x row_chunks=%d) max_joint=%d",
        n_pairs, n_samples, total_row_chunk_launches, n_pair_subchunks, n_row_chunks, max_joint,
    )
    return mi_out


_CUDA_KERNEL: Any = None  # lazy-bound on first call


def batch_pair_mi_cuda(
    factors_data: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    threads_per_block: int = 128,
) -> np.ndarray:
    """numba.cuda batch pair-MI kernel. Mirrors :func:`batch_pair_mi_njit_prange`.

    Raises ``RuntimeError`` if CUDA is not available. The dispatcher routes
    around this; direct callers should gate on :data:`_CUDA_AVAIL`.

    Shared-memory budget caps (sized for cc 6.x 48 KB per-block budget):
      * Max joint cardinality (``nbins[a] * nbins[b]``): ``MAX_JOINT_BINS_CUDA = 128``
      * Max target cardinality (``n_classes_y``): ``MAX_Y_BINS_CUDA = 16``
    Callers exceeding either limit must fall back to the CPU kernel.

    Preconditions enforced host-side (raise ``ValueError`` on violation):
      * ``factors_data >= 0`` everywhere -- negative codes would underflow
        the joint-index arithmetic ``va * nb_b + vb`` and write to undefined
        shared-mem cells (numba.cuda has no array-bounds checks in release
        mode, so this is silent corruption rather than a crash).
      * ``classes_y[i] < n_classes_y`` for every i -- out-of-range class ids
        would write past the shared histogram into ``sm_fx``.
      * ``nbins[a] >= 1`` for every column referenced by ``pair_a`` /
        ``pair_b`` -- a zero-cardinality column collapses ``joint_card`` to
        zero, returning a meaningless MI=0.
    """
    global _CUDA_KERNEL
    if not _CUDA_AVAIL:
        raise RuntimeError("numba.cuda is not available on this host")
    if _CUDA_KERNEL is None:
        _CUDA_KERNEL = _cuda_kernel_factory()
        if _CUDA_KERNEL is None:
            raise RuntimeError("numba.cuda kernel factory failed to build")

    n_pairs = int(pair_a.shape[0])
    if n_pairs == 0:
        # Critic-flagged P0: ``max(...)`` over the empty pair zip raises
        # ``ValueError`` before any device work; short-circuit cleanly.
        return np.empty(0, dtype=np.float64)

    # Shape guard
    n_classes_y = int(freqs_y.shape[0])
    if n_classes_y > MAX_Y_BINS_CUDA:
        raise ValueError(
            f"n_classes_y={n_classes_y} exceeds CUDA shared-memory budget " f"MAX_Y_BINS_CUDA={MAX_Y_BINS_CUDA}; use the CPU kernel instead",
        )
    # Joint-card + min-cardinality guard: check the largest and smallest pair.
    max_joint = 0
    min_nb = None
    for a, b in zip(pair_a, pair_b):
        nb_a = int(nbins[a])
        nb_b = int(nbins[b])
        if nb_a < 1 or nb_b < 1:
            raise ValueError(
                f"degenerate pair ({int(a)}, {int(b)}): nbins=({nb_a}, {nb_b}); " f"at least one column has zero cardinality (skip the pair host-side)",
            )
        if min_nb is None or min(nb_a, nb_b) < min_nb:
            min_nb = min(nb_a, nb_b)
        if nb_a * nb_b > max_joint:
            max_joint = nb_a * nb_b
    if max_joint > MAX_JOINT_BINS_CUDA:
        raise ValueError(
            f"max joint cardinality nbins[a]*nbins[b]={max_joint} exceeds "
            f"CUDA shared-memory budget MAX_JOINT_BINS_CUDA={MAX_JOINT_BINS_CUDA}; "
            f"use the CPU kernel instead",
        )

    # Critic-flagged P0: out-of-range classes_y or negative factors_data writes
    # past shared memory. Validate on host (cheap: one min/max sweep) so the
    # device kernel can stay branch-free in the hot loop.
    if classes_y.size > 0:
        cy_max = int(classes_y.max())
        cy_min = int(classes_y.min())
        if cy_max >= n_classes_y or cy_min < 0:
            raise ValueError(
                f"classes_y values must be in [0, n_classes_y={n_classes_y}); " f"got [min={cy_min}, max={cy_max}]",
            )
    if factors_data.size > 0:
        fd_min = int(factors_data.min())
        if fd_min < 0:
            raise ValueError(
                f"factors_data must be non-negative; got min={fd_min} " f"(merge_vars output should be >= 0 by contract)",
            )

    n_samples = int(factors_data.shape[0])

    # Move inputs to device. ``ascontiguousarray`` guards against non-C-layout
    # numpy arrays that the harness occasionally produces (e.g. .T views).
    d_data = _nb_cuda.to_device(np.ascontiguousarray(factors_data, dtype=np.int32))
    d_pa = _nb_cuda.to_device(np.ascontiguousarray(pair_a, dtype=np.int64))
    d_pb = _nb_cuda.to_device(np.ascontiguousarray(pair_b, dtype=np.int64))
    d_nb = _nb_cuda.to_device(np.ascontiguousarray(nbins, dtype=np.int32))
    d_cy = _nb_cuda.to_device(np.ascontiguousarray(classes_y, dtype=np.int32))
    d_fy = _nb_cuda.to_device(np.ascontiguousarray(freqs_y, dtype=np.float64))
    d_out = _nb_cuda.device_array(n_pairs, dtype=np.float64)

    _CUDA_KERNEL[n_pairs, threads_per_block](
        d_data, d_pa, d_pb, d_nb, d_cy, d_fy, d_out, n_samples, n_classes_y,
    )
    return np.asarray(d_out.copy_to_host())


# ---------------------------------------------------------------------------
# cupy variant
# ---------------------------------------------------------------------------


def batch_pair_mi_cupy(
    factors_data: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
) -> np.ndarray:
    """CuPy batch pair-MI kernel. Mirrors :func:`batch_pair_mi_njit_prange`.

    Implementation note: CuPy doesn't expose a 2D-histogram primitive, so the
    pair joint code (``cls_x * n_classes_y + cls_y``) is collapsed to a 1D
    bincount per pair. The pair loop runs in Python; each iteration is one
    CuPy kernel launch + one ``bincount`` + a small MI reduction. For low
    pair counts (n_pairs <~ 50) this is dispatch-dominated; the CUDA kernel
    is generally faster. For large pair counts the per-pair work amortises.
    """
    if not _CUPY_AVAIL:
        raise RuntimeError("cupy is not available on this host")
    cp = _cp  # local alias

    if pair_a.shape[0] == 0:
        return np.empty(0, dtype=np.float64)

    d_data = cp.asarray(factors_data, dtype=cp.int32)
    d_classes_y = cp.asarray(classes_y, dtype=cp.int32)
    d_freqs_y = cp.asarray(freqs_y, dtype=cp.float64)
    nb_arr = np.asarray(nbins, dtype=np.int32)
    pa_arr = np.asarray(pair_a, dtype=np.int64)
    pb_arr = np.asarray(pair_b, dtype=np.int64)

    n_samples = int(factors_data.shape[0])
    n_pairs = int(pa_arr.shape[0])
    n_classes_y = int(d_freqs_y.shape[0])
    # Wave 47 (2026-05-20): empty factors_data divides by zero in inv_n; return zeros.
    if n_samples == 0:
        return np.zeros(n_pairs, dtype=np.float64)
    inv_n = 1.0 / n_samples
    # Stage each pair's scalar MI into a RESIDENT (n_pairs,) device buffer and D2H it ONCE at the end, instead
    # of a blocking ``float(mi.get())`` per pair (n_pairs serialising syncs that drain the queue between pairs).
    # Bit-identical: same per-pair scalar written to out_dev[p]; only the transfer is batched into one .get().
    out_dev = cp.empty(n_pairs, dtype=cp.float64)

    for p in range(n_pairs):
        a = int(pa_arr[p])
        b = int(pb_arr[p])
        nb_a = int(nb_arr[a])
        nb_b = int(nb_arr[b])
        joint_card = nb_a * nb_b

        va = d_data[:, a]
        vb = d_data[:, b]
        cls_x = va * nb_b + vb  # 1-D joint code
        joint_idx = cls_x * n_classes_y + d_classes_y  # 1-D flat index
        joint_counts_flat = cp.bincount(
            joint_idx, minlength=joint_card * n_classes_y,
        )[: joint_card * n_classes_y]
        joint_counts = joint_counts_flat.reshape(joint_card, n_classes_y).astype(cp.float64)
        joint_freqs = joint_counts * inv_n
        fx = joint_freqs.sum(axis=1)
        # MI reduction: sum where joint_freqs > 0
        # prob_x[i] = fx[i], prob_y[j] = freqs_y[j]; jf = joint_freqs[i, j]
        # Vectorised: log(jf / (fx[:, None] * freqs_y[None, :])) where jf > 0
        denom = fx[:, None] * d_freqs_y[None, :]
        # Guard zeros to avoid log(0). cupy's where + log are safe under masked.
        mask = (joint_freqs > 0) & (denom > 0)
        ratio = cp.where(mask, joint_freqs / cp.where(denom > 0, denom, 1.0), 1.0)
        log_term = cp.where(mask, cp.log(ratio), 0.0)
        out_dev[p] = (joint_freqs * log_term).sum()

    return np.asarray(cp.asnumpy(out_dev))


# ---------------------------------------------------------------------------
# Size-aware dispatcher
# ---------------------------------------------------------------------------


# Crossover thresholds derived from bench_batch_pair_mi_prange.py on a GTX 1050 Ti
# (cc 6.1, 768 CUDA cores, 4 GB VRAM) vs an i7 4-physical-core CPU. Measured points:
#
#   | n_rows  x  n_pairs | layer2_prange | cuda    | cuda/cpu_njit |
#   |--------------------|---------------|---------|----------------|
#   |  200 000  x   28   |   8.56 x      |  8.00x  |  0.93x (CPU)   |
#   |  500 000  x  120   |  13.94 x      | 29.88x  |  2.14x (CUDA)  |
#   | 1 000 000  x   66   |  11.47 x      | 20.19x  |  1.76x (CUDA)  |
#
# CUDA pulls ahead of the CPU njit kernel around n_rows ~= 400k (with enough
# pairs to amortise the per-block fixed cost). Below that, the CPU prange
# kernel keeps the GTX-1050-Ti grid under-occupied (28 pairs => 28 blocks of 128
# threads = 3584 active threads on a card that wants 6-10x more for full
# occupancy), and the H2D / D2H transfer overhead dominates.
#
# CuPy never beat numba.cuda in any of the measured points (always 2-5x SLOWER)
# because each pair dispatches one Python-side bincount kernel; the per-launch
# overhead defeats the per-pair work. It only becomes competitive on very
# large pair counts (>200) where the launch cost amortises -- those thresholds
# stay defensive.
#
# Callers can override the heuristic via ``dispatch_batch_pair_mi(..., force_backend=)``.
# Wave 23 P1 (2026-05-20): the 4 hardcoded thresholds below are
# "measured on GTX 1050 Ti cc 6.1" defaults; per
# `feedback_use_kernel_tuning_cache_for_gpu` they should be lookup-
# driven via ``pyutilz.performance.kernel_tuning.cache`` so that consumer
# Ampere GPUs (~5-10x lower cuda crossover) and high-VRAM cards don't
# leave 2-4x on the table.
#
# These remain as the source-code fallback; the live dispatch path
# below now consults the cache first and uses these only when the
# cache has no entry for the live HW yet (first-run / no-sweep state).
CUDA_MIN_ROWS = 400_000
CUDA_MIN_PAIRS = 16
CUPY_MIN_ROWS = 5_000_000
CUPY_MIN_PAIRS = 200

# Kernel-tuning-cache integration (get_or_tune + @kernel_tuner) ---------------
#
# Wave 23 P1 (2026-05-20) flagged the four hardcoded crossover thresholds above
# as HW-overfit (GTX 1050 Ti cc 6.1). Per ``feedback_use_kernel_tuning_cache_for_gpu``
# the live dispatch now consults the per-host cache via the shared ``get_or_tune``
# orchestrator and a measured backend-crossover sweep. The thresholds above remain
# the source-code fallback, applied verbatim when the cache has no entry for the
# live HW yet (mirrors ``signal/dtw.py``).
#
# 2-D axes: ``n_samples`` (dominant; primary sweep axis) and ``n_pairs`` (held at a
# representative value for the sweep -- 64, above CUDA_MIN_PAIRS=16 and below
# CUPY_MIN_PAIRS=200, matching the measured 28-120-pair bench band -- and threaded
# through as an extra region key so the cached regions stay keyed on both axes).
_BPMI_SWEEP_N_PAIRS_GRID = [16, 64, 256]  # full n_pairs axis (was a single fixed 64)
# Grid floor reaches below the GPU crossover (measured ~85-100k rows on a laptop RTX 500 Ada) so the cache learns the CPU-favorable
# low-n region instead of extrapolating the lowest measured cell down to n=0 (which mis-routed 50-75k-row calls to a slower GPU launch).
_BPMI_SWEEP_N_SAMPLES = [50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
_BPMI_SWEEP_N_CLASSES_Y = 4
_BPMI_SWEEP_N_BINS = 8  # joint card = 8*8 = 64 <= MAX_JOINT_BINS_CUDA fallback (128)
_BPMI_SALT = 3  # serial-njit variant + full 2-D (n_samples x n_pairs) grid + grid floor lowered below the GPU crossover

# Serial CPU variant: recompile the SAME prange body WITHOUT ``parallel`` -> numba
# treats prange as range. The tuner now picks njit_serial (small n: no thread-spawn
# overhead) vs njit_parallel per region, not just CPU-vs-GPU.
from numba import njit as _njit

# getattr fallback: under NUMBA_DISABLE_JIT=1 the kernel is a plain function (no
# .py_func) and njit is a pass-through, so serial == the same callable.
batch_pair_mi_njit_serial = _njit(nogil=True, cache=True)(getattr(batch_pair_mi_njit_prange, "py_func", batch_pair_mi_njit_prange))


def _make_batch_pair_mi_inputs(dims: dict):
    """Synthetic (factors_data, pair_a, pair_b, nbins, classes_y, freqs_y) at
    ``dims['n_samples']`` rows x ``dims['n_pairs']`` pairs. Bins are capped so the
    per-pair joint cardinality stays inside the CUDA shared-mem budget (so the cuda
    variant is exercised, not guard-rejected)."""
    rng = np.random.default_rng(0)
    n_samples = int(dims["n_samples"])
    n_pairs = int(dims["n_pairs"])
    nbins_val = _BPMI_SWEEP_N_BINS
    n_features = 32  # enough columns for up to 256 distinct (a, b) pairs
    factors_data = rng.integers(0, nbins_val, size=(n_samples, n_features)).astype(np.int32)
    nbins = np.full(n_features, nbins_val, dtype=np.int32)
    pair_a = rng.integers(0, n_features, size=n_pairs).astype(np.int64)
    pair_b = (pair_a + 1 + rng.integers(0, n_features - 1, size=n_pairs)) % n_features
    pair_b = pair_b.astype(np.int64)
    classes_y = rng.integers(0, _BPMI_SWEEP_N_CLASSES_Y, size=n_samples).astype(np.int32)
    freqs_y = np.bincount(classes_y, minlength=_BPMI_SWEEP_N_CLASSES_Y).astype(np.float64) / max(1, n_samples)
    return (factors_data, pair_a, pair_b, nbins, classes_y, freqs_y)


def _run_batch_pair_mi_sweep() -> list:
    """Full (n_samples x n_pairs) grid sweep -> backend_choice regions: njit_serial /
    njit_parallel / cuda / cupy, fastest EQUIVALENT per cell. Both n_samples and
    n_pairs are swept (not a fixed-n_pairs 1-D crossover). Inputs are host-resident
    (the FS pipeline feeds the host dataframe) so there is no residency axis. GPU
    variants only when available; cupy/cuda fp reductions agree with njit to a
    loosened rtol (last-bit log() reassociation)."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {
        "njit_serial": lambda *a: batch_pair_mi_njit_serial(*a),
        "njit_parallel": lambda *a: batch_pair_mi_njit_prange(*a),
    }
    if _CUDA_AVAIL:
        variants["cuda"] = lambda *a: batch_pair_mi_cuda(*a)
    if _CUPY_AVAIL:
        variants["cupy"] = lambda *a: batch_pair_mi_cupy(*a)
    return sweep_backend_grid(  # type: ignore[no-any-return]  # pyutilz helper returns the declared list of results
        variants,
        {"n_samples": _BPMI_SWEEP_N_SAMPLES, "n_pairs": _BPMI_SWEEP_N_PAIRS_GRID},
        _make_batch_pair_mi_inputs,
        reference="njit_serial",
        repeats=5, equiv_rtol=1e-3, equiv_atol=1e-3,
    )


def _batch_pair_mi_fallback_choice(n_samples: int, n_pairs: int) -> str:
    """Pre-sweep heuristic (the spec's dynamic fallback callable): the old
    CUDA_/CUPY_MIN_* GPU thresholds + availability; on CPU, parallel njit above a
    modest row count (below it the thread-spawn overhead loses to serial)."""
    if _CUPY_AVAIL and n_samples >= CUPY_MIN_ROWS and n_pairs >= CUPY_MIN_PAIRS:
        return "cupy"
    if _CUDA_AVAIL and n_samples >= CUDA_MIN_ROWS and n_pairs >= CUDA_MIN_PAIRS:
        return "cuda"
    return "njit_parallel" if n_samples >= 100_000 else "njit_serial"


def _batch_pair_mi_backend_choice(n_samples: int, n_pairs: int) -> str:
    """Per-host backend (njit_serial/njit_parallel/cuda/cupy) for this (n_samples,
    n_pairs) via the spec's choose(); maps a legacy 'njit' region (pre serial/parallel
    split) to njit_parallel."""
    bc = _BPMI_SPEC.choose(n_samples=int(n_samples), n_pairs=int(n_pairs))
    return "njit_parallel" if bc == "njit" else bc


def _required_gpu_bytes(factors_data: np.ndarray, pair_a: np.ndarray, nbins: np.ndarray, classes_y: np.ndarray, freqs_y: np.ndarray) -> int:
    """Estimated device-resident bytes for one ``batch_pair_mi_cuda``/``batch_pair_mi_cupy`` call.

    ``factors_data`` is uploaded WHOLESALE (every column, not just the ones referenced by this pair
    chunk) and always as int32 regardless of its host dtype -- that upload dominates the footprint at
    production scale (n=2.4M rows already needs ~4GB as int32, i.e. the entire VRAM budget of a 4GB
    card) and is invariant across chunks, so this must be checked BEFORE every cuda/cupy attempt, not
    just once.
    """
    n_pairs = int(pair_a.shape[0])
    return int(factors_data.size * 4 + n_pairs * (8 + 8 + 8) + nbins.size * 4 + classes_y.size * 4 + freqs_y.size * 8)


def _gpu_upload_fits(required_bytes: int, *, n_samples: int = 0, n_cols: int = 0, n_pairs: int = 0, context: str = "batch_pair_mi") -> bool:
    """Pre-flight VRAM check before launching a cuda/cupy pair-MI kernel -- mirrors the ``_should_use_cuda``
    guard pattern already used by ``_cmi_cuda.py`` / ``gpu.py`` / ``hermite_fe`` / ``friend_graph_gpu.py`` /
    ``batch_mi_noise_gate_gpu.py`` / ``_permutation_null_pair_resident.py`` (this module was the one
    remaining GPU dispatch site without the guard). Two layers: a relative cap (<=50% of currently free
    VRAM, capped at 1.5 GB) as a cheap first pass, then the shared ABSOLUTE cushion floor from
    ``_fe_gpu_vram.fe_gpu_has_vram_cushion`` (>=1 GB free after the allocation, env-overridable via
    ``MLFRAME_FE_GPU_MIN_FREE_MB``) so every GPU dispatch site in this package agrees on one definition of
    "safe to launch".

    Why this matters (2026-07-10 wellbore 3M-row production crash): on a small-VRAM WDDM host, uploading an
    oversized array does NOT reliably raise a catchable CUDA OOM -- WDDM can transparently over-subscribe
    device memory via host-paging, so the upload "succeeds" and the kernel launch then grinds through
    PCIe-paged memory for minutes before the OS kills the process with NO Python exception, no traceback,
    and no Windows Event Log trace (silent ``EXIT_CODE=1``, confirmed via a real 2.4M-row/423-column
    production run). A pre-flight check is the only way to avoid entering that state at all; catching an
    exception afterward is too late because there may be none to catch.

    A REJECTION is always logged at WARNING with the full sizing context (rows/cols/pairs/dtype, requested
    GB, free/total VRAM) so a production run is diagnosable from the log alone -- never a silent fallback.

    Permissive (``True``) whenever cupy/memGetInfo is unavailable, matching every sibling guard's fail-open
    contract for non-GPU / probe-failure hosts.
    """
    cap = 1536 * 1024 * 1024  # 1.5 GB conservative cap (shared small card), same default as _cmi_cuda._should_use_cuda
    try:
        import cupy as cp

        free_b, total_b = cp.cuda.runtime.memGetInfo()
        cap = min(cap, int(free_b * 0.5))
    except Exception as e:
        logger.debug("%s._gpu_upload_fits: memGetInfo failed (%s); permissive", context, e)
        return True
    if required_bytes > cap:
        logger.warning(
            "%s: GPU upload REJECTED -- requested %.2fGB (n_samples=%d, n_cols=%d, n_pairs=%d, dtype=int32) "
            "exceeds the safe relative cap %.2fGB (50%% of %.2fGB free / %.2fGB total VRAM) -- routing to a "
            "row-chunked GPU path or CPU njit instead of risking a silent VRAM-oversubscription crash",
            context, required_bytes / 1024**3, n_samples, n_cols, n_pairs, cap / 1024**3, free_b / 1024**3, total_b / 1024**3,
        )
        return False
    try:
        from mlframe.feature_selection.filters._fe_gpu_vram import fe_gpu_has_vram_cushion

        if not fe_gpu_has_vram_cushion(required_bytes):
            logger.warning(
                "%s: GPU upload REJECTED -- requested %.2fGB (n_samples=%d, n_cols=%d, n_pairs=%d, dtype=int32) "
                "would breach the absolute VRAM cushion floor (free=%.2fGB, total=%.2fGB) -- routing to a "
                "row-chunked GPU path or CPU njit instead of risking a silent VRAM-oversubscription crash",
                context, required_bytes / 1024**3, n_samples, n_cols, n_pairs, free_b / 1024**3, total_b / 1024**3,
            )
            return False
    except Exception as e:
        logger.debug("%s._gpu_upload_fits: cushion probe failed (%s); permissive", context, e)
    return True


def dispatch_batch_pair_mi(
    factors_data: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    force_backend: str | None = None,
) -> tuple[np.ndarray, str]:
    """Pick the fastest backend by (n_samples, n_pairs) heuristic and run it.

    Returns ``(mi_array, backend_name)`` so callers can log which path fired.
    ``force_backend in {"njit", "cuda", "cupy"}`` overrides the heuristic.

    When the FULL upload would not safely fit in free VRAM (see :func:`_gpu_upload_fits`) but CUDA IS
    available, routes to :func:`batch_pair_mi_cuda_row_chunked` -- a row-chunked GPU path that still gets
    the GPU speed win by uploading ``factors_data`` in VRAM-sized row-blocks and accumulating the joint
    histogram across them (bit-identical result; see that function's docstring). Only fully drops to the
    CPU njit kernel when even that fails (no CUDA, or a genuine runtime/driver fault) -- a slower CORRECT
    result is still preferred over ever risking the silent-crash upload, but "slower" no longer means
    "no GPU at all" whenever CUDA is present.
    """
    n_samples = int(factors_data.shape[0])
    n_cols = int(factors_data.shape[1]) if factors_data.ndim == 2 else 0
    n_pairs = int(pair_a.shape[0])
    _req_bytes = _required_gpu_bytes(factors_data, pair_a, nbins, classes_y, freqs_y)
    _vram_ok = _gpu_upload_fits(_req_bytes, n_samples=n_samples, n_cols=n_cols, n_pairs=n_pairs)

    def _try_cuda_row_chunked(reason: str) -> tuple[np.ndarray, str] | None:
        if not _CUDA_AVAIL:
            return None
        try:
            mi = batch_pair_mi_cuda_row_chunked(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y)
            logger.info("batch_pair_mi: %s -- completed via row-chunked CUDA (GPU speed preserved, VRAM-safe)", reason)
            return mi, "cuda_row_chunked"
        except Exception as e:
            logger.warning("batch_pair_mi: row-chunked CUDA also failed (%s: %s) -- falling back to CPU njit", type(e).__name__, e)
            return None

    # Explicit override
    if force_backend is not None:
        force_backend = force_backend.lower()
        if force_backend == "cuda" and _CUDA_AVAIL:
            if _vram_ok:
                try:
                    return batch_pair_mi_cuda(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "cuda"
                except Exception as e:
                    logger.warning("batch_pair_mi: forced CUDA backend failed (%s: %s) -- trying row-chunked CUDA", type(e).__name__, e)
                    _result = _try_cuda_row_chunked("forced CUDA backend failed on the full-upload path")
                    if _result is not None:
                        return _result
            else:
                _result = _try_cuda_row_chunked("forced CUDA backend requested but full upload does not fit VRAM")
                if _result is not None:
                    return _result
        elif force_backend == "cupy" and _CUPY_AVAIL and _vram_ok:
            try:
                return batch_pair_mi_cupy(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "cupy"
            except Exception as e:
                logger.warning("batch_pair_mi: forced cupy backend failed (%s: %s) -- falling back to CPU njit", type(e).__name__, e)
        return batch_pair_mi_njit_prange(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "njit"

    # Per-host backend (njit/cuda/cupy) from the kernel_tuning_cache via the shared
    # get_or_tune orchestrator; measurement-backed fallback = the old CUDA_/CUPY_MIN_*
    # thresholds. Guarded by live availability (the tuning host had the backend; a
    # reader may not) -- preserves the original cupy-then-cuda-then-njit preference order.
    choice = _batch_pair_mi_backend_choice(n_samples, n_pairs)

    if choice == "cupy" and _CUPY_AVAIL and _vram_ok:
        try:
            return batch_pair_mi_cupy(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "cupy"
        except Exception:  # nosec B110 - optional/best-effort path, rationale documented
            pass  # fall through

    if choice == "cuda" and _CUDA_AVAIL:
        if _vram_ok:
            try:
                return batch_pair_mi_cuda(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "cuda"
            except Exception:
                # Shape guard tripped or a runtime/driver fault -> try row-chunked CUDA, then CPU. Broadened
                # from ``(ValueError, RuntimeError)`` (2026-07-10): numba's ``CudaAPIError``/``CudaDriverError``
                # derive directly from ``Exception``, not ``RuntimeError``, so a genuine CUDA driver fault
                # used to skip this handler and propagate to the caller uncaught.
                _result = _try_cuda_row_chunked("full-upload CUDA kernel raised")
                if _result is not None:
                    return _result
        else:
            _result = _try_cuda_row_chunked("size-heuristic picked CUDA but full upload does not fit VRAM")
            if _result is not None:
                return _result

    # CPU: serial vs parallel njit per the tuned choice (tag stays "njit").
    if choice == "njit_serial":
        return batch_pair_mi_njit_serial(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "njit"
    return batch_pair_mi_njit_prange(factors_data, pair_a, pair_b, nbins, classes_y, freqs_y), "njit"


def _free_ram_bytes_for_chunking() -> int:
    """Best-effort free physical RAM in bytes; conservative fallback if psutil is missing. Mirrors
    ``_mrmr_sis_screen._free_ram_bytes`` -- duplicated (not imported) to avoid a cross-package import
    cycle (``_mrmr_sis_screen`` itself lazily imports sibling FE modules that can reach this file)."""
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return 2 * 1024**3  # 2 GB conservative fallback


def _fallback_pair_chunk_size(free_bytes: int) -> int:
    """Measurement-backed default pairs-per-chunk from free RAM alone.

    Per chunk we hold, at most simultaneously: two int64 id arrays (``pair_a``/``pair_b``) and one
    float64 output array, i.e. ``chunk_pairs * (8 + 8 + 8)`` bytes, plus per-backend transient overhead
    (the CUDA/CuPy paths additionally stage a same-sized device-resident buffer). Budget conservatively
    at ~1/16 of free RAM for the transient so this never competes meaningfully with the caller's own
    ``data``/``cached_MIs`` state. Clamped to a sane [50_000, 20_000_000] pairs.
    """
    budget = max(1, free_bytes // 16)
    chunk = int(budget // 48)  # 3 arrays * 8 bytes * ~2x safety headroom
    return int(np.clip(chunk, 50_000, 20_000_000))


def _choose_pair_chunk_size(free_bytes: int) -> int:
    """Look the pairs-per-chunk up in the kernel_tuning_cache keyed on a free-RAM bucket; fall back to
    the measured analytic default. Mirrors ``_mrmr_sis_screen._choose_chunk_width``'s pattern -- this is
    a MEMORY-SAFETY bound (never hardcoded), not a throughput choice; the throughput-critical decision
    (which backend: njit/cuda/cupy) remains fully delegated to :func:`dispatch_batch_pair_mi` per chunk,
    which is already kernel_tuning_cache-driven via ``_batch_pair_mi_backend_choice``."""
    gb = max(1, int(free_bytes // (1024**3)))
    ram_bucket = int(gb.bit_length())
    fb = _fallback_pair_chunk_size(free_bytes)
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        ktc = KernelTuningCache.load_or_create()
        hit = ktc.lookup("mrmr_batch_pair_mi_chunk_size", ram_bucket=ram_bucket)
        if hit and "chunk_pairs" in hit:
            return int(np.clip(int(hit["chunk_pairs"]), 50_000, 20_000_000))
        try:
            ktc.update(
                "mrmr_batch_pair_mi_chunk_size",
                axes=["ram_bucket"],
                regions=[{"ram_bucket": ram_bucket, "chunk_pairs": fb}],
            )
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            import logging

            logging.getLogger(__name__).debug("suppressed in batch_pair_mi_gpu.py (chunk-size cache update): %s", e)
    except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
        import logging

        logging.getLogger(__name__).debug("suppressed in batch_pair_mi_gpu.py (chunk-size cache lookup): %s", e)
    return fb


def _iter_upper_triangle_pair_chunks(k: int, chunk_pairs: int):
    """Yield ``(a_pos, b_pos)`` int64 POSITION arrays (0-based positions into a length-``k`` id list) for
    successive row-blocks of the upper-triangle pair space (``a_pos < b_pos``), each containing at most
    ``chunk_pairs`` pairs.

    Never materialises the full ``C(k, 2)`` pair list: each row ``i`` contributes ``k - 1 - i`` pairs via
    a plain ``np.arange``, so the per-chunk cost is ``O(chunk_pairs)`` and the total cost across the whole
    generator is ``O(k + total_pairs)`` -- the same asymptotic work an exhaustive pairwise scan requires
    regardless of implementation, with peak memory bounded by ``chunk_pairs`` instead of ``C(k, 2)``.
    """
    if k < 2 or chunk_pairs < 1:
        return
    i = 0
    while i < k - 1:
        rows: list[int] = []
        cum = 0
        while i < k - 1 and cum < chunk_pairs:
            rows.append(i)
            cum += k - 1 - i
            i += 1
        row_ids = np.asarray(rows, dtype=np.int64)
        counts = (k - 1 - row_ids).astype(np.int64)
        a_pos = np.repeat(row_ids, counts)
        b_pos = np.concatenate([np.arange(r + 1, k, dtype=np.int64) for r in rows])
        yield a_pos, b_pos


def dispatch_batch_pair_mi_chunked(
    factors_data: np.ndarray,
    ids: np.ndarray,
    nbins: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    force_backend: str | None = None,
    max_pairs_per_chunk: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Full-upper-triangle batch pair-MI over ``ids`` (all-pairs among ``ids``), processed in RAM-bounded
    row-block chunks so peak memory never scales with ``C(len(ids), 2)``.

    Replaces the previous pattern of building the full ``pair_a``/``pair_b``/output arrays via
    ``np.triu_indices`` up front (``O(k^2)`` memory -- infeasible past a few thousand columns; at
    k=100_000, ``C(k,2)`` ~= 5e9 pairs would need ~120 GB just for the index/output arrays). Each chunk is
    still dispatched through the existing :func:`dispatch_batch_pair_mi` (so backend selection stays fully
    kernel_tuning_cache-driven, per-chunk); only the ENUMERATION of which pairs to compute is chunked.

    Returns ``(pair_a_ids, pair_b_ids, mi_values, backend_counts)`` where ``pair_a_ids``/``pair_b_ids`` are
    the actual column ids (not positions) and ``backend_counts`` maps backend name -> number of chunks
    that ran on it (for logging; a mixed-backend run is possible if a GPU chunk fails mid-sweep and the
    per-chunk call falls through to CPU).
    """
    ids_arr = np.asarray(ids, dtype=np.int64)
    k = int(ids_arr.shape[0])
    if k < 2:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64), {}

    chunk_pairs = int(max_pairs_per_chunk) if max_pairs_per_chunk else _choose_pair_chunk_size(_free_ram_bytes_for_chunking())
    chunk_pairs = max(1, chunk_pairs)

    a_out: list[np.ndarray] = []
    b_out: list[np.ndarray] = []
    mi_out: list[np.ndarray] = []
    backend_counts: dict[str, int] = {}

    for a_pos, b_pos in _iter_upper_triangle_pair_chunks(k, chunk_pairs):
        pair_a = ids_arr[a_pos]
        pair_b = ids_arr[b_pos]
        mi_chunk, backend_used = dispatch_batch_pair_mi(
            factors_data=factors_data,
            pair_a=pair_a,
            pair_b=pair_b,
            nbins=nbins,
            classes_y=classes_y,
            freqs_y=freqs_y,
            force_backend=force_backend,
        )
        a_out.append(pair_a)
        b_out.append(pair_b)
        mi_out.append(mi_chunk)
        backend_counts[backend_used] = backend_counts.get(backend_used, 0) + 1

    if not a_out:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64), {}

    return (
        np.concatenate(a_out),
        np.concatenate(b_out),
        np.concatenate(mi_out),
        backend_counts,
    )


# Register with the @kernel_tuner registry so retune_all / mlframe-tune-kernels
# discover + batch-tune batch_pair_mi. GPU-capable (cuda/cupy backends).
from pyutilz.performance.kernel_tuning.registry import kernel_tuner

_BPMI_SPEC = kernel_tuner(
    kernel_name="batch_pair_mi",
    variant_fns=(batch_pair_mi_njit_serial, batch_pair_mi_njit_prange),  # CPU bodies; GPU covered by salt
    tuner=_run_batch_pair_mi_sweep,
    axes={"n_samples": list(_BPMI_SWEEP_N_SAMPLES), "n_pairs": list(_BPMI_SWEEP_N_PAIRS_GRID)},
    fallback=_batch_pair_mi_fallback_choice,  # callable (n_samples, n_pairs) -> str
    gpu_capable=True,
    salt=_BPMI_SALT,
    cli_label="batch_pair_mi",
)


__all__ = [
    "batch_pair_mi_njit_prange",
    "batch_pair_mi_cuda",
    "batch_pair_mi_cupy",
    "dispatch_batch_pair_mi",
    "dispatch_batch_pair_mi_chunked",
    "_CUDA_AVAIL",
    "_CUPY_AVAIL",
]
