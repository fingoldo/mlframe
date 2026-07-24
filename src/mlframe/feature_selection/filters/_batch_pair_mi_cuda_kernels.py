"""``numba.cuda`` kernel machinery for :mod:`batch_pair_mi_gpu`.

Carved out of ``batch_pair_mi_gpu.py`` to keep that file under the repo's
1000-LOC gate (see ``tests/test_meta/test_no_file_over_1k_loc.py``). This
module owns the shared-memory-budget derivation, the two CUDA kernel
factories (single-shot + row-chunked, plus the row-chunked kernel's
shared-memory-staged variant), the host-side MI finalize for the row-chunked
accumulator, and the two public entry points (:func:`batch_pair_mi_cuda`,
:func:`batch_pair_mi_cuda_row_chunked`).

``_CUDA_AVAIL``/``_nb_cuda`` are re-derived here (not imported from the
parent) to avoid a circular import: the parent imports several names back
from this module.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numba
import numpy as np

logger = logging.getLogger(__name__)

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

# bench-attempt-rejected (2026-07-14): raising this 16 -> 20 to admit the wellbore-100k production
# target (n_classes_y=20, whose rejection sent the whole batched pair-MI into the row-chunked path,
# 118 launches / 227.7s) DOES fit shared memory (256*(20+1)*8 = 43,008 B <= 48 KB, joint stays 256)
# -- but it cannot restore the full-resident kernel for that workload anyway: the same production
# shape has max_joint = 441 (nbins=21 -> 21*21), which exceeds MAX_JOINT_BINS_CUDA=256, so the full
# kernel is rejected on the JOINT cap regardless of the y cap. Meanwhile the y cap leaks into the
# row-chunked path's own shared-vs-global kernel-variant selection, changing its chunk geometry and
# breaking its fragmentation-bail contract at y=20 fixtures. The real fix for the 227s row-chunked
# cost is a global-memory-histogram kernel variant for large joint*y shapes (no smem caps at all),
# not a cap bump -- tracked as the follow-up.
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
        """One CUDA block per pair: accumulate this row-chunk's joint/marginal histogram via global atomics."""
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
        """One CUDA block per pair: stage this row-chunk's histogram in shared memory, then flush once to global."""
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
    or, worse on some drivers, a silently truncated allocation).

    Pure arithmetic against a device shared-memory budget -- doesn't need a live CUDA device to
    evaluate (the ``try/except`` below already falls back to the cc 6.x+ safe constant when no
    device is queryable), so this must NOT early-return 0 on a CUDA-less host; doing so made the
    gate untestable off-GPU and, worse, would silently disable the shared-memory fast path's
    threshold logic wherever it's exercised without a live device."""
    needed = (max_joint * n_classes_y + max_joint) * 4  # int32
    try:
        budget = _nb_cuda.get_current_device().MAX_SHARED_MEMORY_PER_BLOCK
    except Exception:
        budget = 49152  # cc 6.x+ safe default without opt-in
    # Leave headroom for the driver's own per-block reserved shared memory (a few hundred bytes,
    # observed ~1KB on cc 8.9 via ncu's "Driver Shared Memory Per Block").
    return needed if needed <= budget - 2048 else 0


def _mi_from_joint_counts_cupy(joint_counts_dev, fx_counts_dev, freqs_y_dev, n_samples: int) -> Any:
    """Device-side twin of :func:`_mi_from_joint_counts`: reduces the ``(n_pairs, max_joint,
    n_classes_y)`` accumulator to a ``(n_pairs,)`` MI vector ON-DEVICE via cupy broadcasting, so the
    row-chunked path's PCIe D2H shrinks from the full histogram accumulator (``n_pairs*max_joint*
    n_classes_y*8`` bytes -- 698MB at the wellbore-100k production pair-subchunk shape, 9 subchunks
    -> ~6.3GB total transferred) to the final ``(n_pairs,)`` float64 result alone.

    Bit-identical to :func:`_mi_from_joint_counts`: same ``sum jf*log(jf/(px*py))`` reduction over the
    SAME (i, j) grid, only reassociated across a different (but still commutative/associative-safe
    integer-count-derived) summation order -- fp reduction order differs at the ~1e-15 ULP level
    (verified in the accompanying regression test), never a selection-relevant magnitude.

    ``fx==0`` and ``freqs_y[j]<=0`` cells are implicitly excluded via a ``cp.where`` mask (mirrors the
    host loop's ``if fx == 0: continue`` / ``if prob_y > 0.0`` guards) rather than skipped by iteration,
    since cupy broadcasting always touches the full grid -- correctness-equivalent, not a performance
    concern (the grid is already resident and small relative to the accumulator it replaces)."""
    inv_n = 1.0 / n_samples
    prob_x = fx_counts_dev.astype(_cp.float64) * inv_n  # (n_pairs, max_joint)
    prob_y = freqs_y_dev.astype(_cp.float64)  # (n_classes_y,)
    jf = joint_counts_dev.astype(_cp.float64) * inv_n  # (n_pairs, max_joint, n_classes_y)
    denom = prob_x[:, :, None] * prob_y[None, None, :]
    valid = (joint_counts_dev != 0) & (fx_counts_dev[:, :, None] != 0) & (prob_y[None, None, :] > 0.0)
    terms = _cp.where(valid, jf * _cp.log(jf / denom), 0.0)
    return terms.sum(axis=(1, 2))


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
    """Build a CUDA kernel that zero-fills a flat device array in place, or None if CUDA is unavailable."""
    if not _CUDA_AVAIL:
        return None

    @_nb_cuda.jit
    def _kernel(flat_arr, n):
        """Grid-stride zero-fill of ``flat_arr[:n]``."""
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
    if n_samples == 0:
        # GPU_INFRA_A-2 fix (mrmr_audit_2026-07-22): mirrors batch_pair_mi_cupy's explicit
        # `if n_samples == 0: return zeros` guard -- without it, _mi_from_joint_counts_cupy/
        # _mi_from_joint_counts's `inv_n = 1.0 / n_samples` raises ZeroDivisionError for an empty input.
        return np.zeros(n_pairs, dtype=np.float64)

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
    # nbins is indexed by raw column id (not pair position), small, and fit-constant (cardinalities never
    # change) -- resident_operand (content-hash keyed) shares the upload not just across this call's
    # pair-subchunks but across every OUTER call to this function within the same fit too, instead of a
    # fresh to_device on every call (2026-07-12, was re-uploaded every outer call despite never changing).
    from ._fe_resident_operands import resident_operand
    d_nb = resident_operand(nbins, "bpmi_nbins", dtype=np.int32)

    mi_out = np.empty(n_pairs, dtype=np.float64)
    total_row_chunk_launches = 0
    # freqs_y_c is fit-constant (the y-marginal never changes across pair-subchunks OR row-chunks within
    # this call) but the finalize step re-uploaded it via cp.asarray on EVERY pair-subchunk iteration
    # (cProfile, 2026-07-15 wellbore fit: cupy._core.core.array 56.1s / 19620 calls, ~6540 outer iterations
    # x 3 asarray calls -- freqs_y_c was one of the 3, purely wasted since only d_joint/d_fx change per
    # iteration). Upload it ONCE before the loop; bit-identical (same array, same device, same dtype).
    _d_freqs_y_const = _cp.asarray(freqs_y_c) if _CUPY_AVAIL else None

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

        if _CUPY_AVAIL:
            d_joint_cp = _cp.asarray(d_joint)
            d_fx_cp = _cp.asarray(d_fx)
            mi_sub = _mi_from_joint_counts_cupy(d_joint_cp, d_fx_cp, _d_freqs_y_const, n_samples)
            mi_out[pair_start:pair_end] = _cp.asnumpy(mi_sub)
        else:
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

    # RESIDENT UPLOAD (2026-07-12): ``factors_data``/``nbins``/``classes_y``/``freqs_y`` are all
    # fit-constant across the whole greedy FE round (this kernel is reached once per pair-chunk of the
    # SAME candidate pool, per ``dispatch_batch_pair_mi_chunked``), yet were re-uploaded via raw
    # ``_nb_cuda.to_device`` on EVERY call -- ~200MB of ``factors_data`` alone at 100k x 500, the
    # single highest-magnitude finding in the whole-file audit. ``resident_operand`` (this package's
    # proven fit-constant GPU cache, already used by 28+ sibling files) content-hashes each array and
    # returns a cached cupy device array on a repeat upload -- verified (see module docstring / CUDA
    # Array Interface) that a cupy device array is a drop-in argument for a ``numba.cuda.jit`` kernel
    # launch, so no separate numba.cuda-flavored cache is needed. ``pair_a``/``pair_b`` genuinely vary
    # per call (the chunk's own pair ids) and stay a fresh per-call upload; ``d_out`` is a fresh output
    # buffer every call (never cached). Bit-identical: same values, only the transfer is deduplicated.
    from ._fe_resident_operands import resident_operand
    d_data = resident_operand(factors_data, "bpmi_factors_data", dtype=np.int32)
    d_nb = resident_operand(nbins, "bpmi_nbins", dtype=np.int32)
    d_cy = resident_operand(classes_y, "bpmi_classes_y", dtype=np.int32)
    d_fy = resident_operand(freqs_y, "bpmi_freqs_y", dtype=np.float64)
    d_pa = _nb_cuda.to_device(np.ascontiguousarray(pair_a, dtype=np.int64))
    d_pb = _nb_cuda.to_device(np.ascontiguousarray(pair_b, dtype=np.int64))
    d_out = _nb_cuda.device_array(n_pairs, dtype=np.float64)

    _CUDA_KERNEL[n_pairs, threads_per_block](
        d_data, d_pa, d_pb, d_nb, d_cy, d_fy, d_out, n_samples, n_classes_y,
    )
    return np.asarray(d_out.copy_to_host())
