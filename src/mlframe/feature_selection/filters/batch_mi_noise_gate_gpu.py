"""GPU backend for the batched FE-candidate MI + permutation noise-gate kernel.

Twin of the CPU njit ``batch_mi_with_noise_gate`` (``info_theory.py``), modeled
EXACTLY on ``batch_pair_mi_gpu.py``: two GPU joint-histogram backends (cupy +
numba.cuda), a per-host backend chooser via ``get_or_tune``, a measurement-backed
fallback heuristic, a real CPU-vs-GPU sweep tuner, an availability-guarded
dispatcher, and a ``@kernel_tuner`` registration.

BIT-IDENTITY (non-negotiable - FE recovery is pinned)
------------------------------------------------------
The noise-gate rejection is a float comparison (``mi_perm >= original_mi``), so a
GPU MI that drifts by a single ULP would flip borderline rejections and change
which engineered features MRMR keeps. To stay bit-identical we split the work:

  1. The ``npermutations`` y-shuffles run on the CPU with the IDENTICAL LCG /
     Fisher-Yates stream the CPU kernel uses (``base_seed*2654435761 + (i+1)``
     then the PCG step). Only 3 small shuffles by default - cheap.
  2. The GPU computes the INTEGER joint histograms (deterministic counting, hence
     bit-exact) for all ``K`` candidate columns against the original ``y`` AND
     against each shuffled ``y``.
  3. The MI / entropy is computed FROM those integer counts on the CPU via the
     SAME accumulation order as the CPU kernel's ``_relevance_from_dense``
     (``for i in K_x: for j in K_y`` over ascending bin codes, skipping empty
     bins). Empty (pruned) bins contribute exactly 0 to both the MI sum and the
     H(X) sum, so computing from the FULL (non-densified) ``nbins_k`` counts is
     numerically identical to the CPU kernel's dense path.
  4. The identical rejection rule is applied.

The only GPU work is the O(n*K) counting; the entropy + comparison stay on the
bit-exact CPU path. ``_mi_from_counts_cpu`` is a tiny njit reduction that
reproduces ``_relevance_from_dense`` exactly given the integer counts.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import cupy

import numpy as np
from mlframe.utils.log_throttle import log_throttle

logger = logging.getLogger(__name__)

# CPU reference kernel (the required win + always-correct fallback).
from .info_theory import batch_mi_with_noise_gate as _cpu_batch_mi_with_noise_gate

# Compute kernels (GPU dep probes, cupy/cuda kernel factories + their compile helpers,
# and the bit-exact CPU njit reducers) live in the sibling module; carved out to keep
# this module under the 1k-LOC ceiling. Re-exported below so every import path resolves.
from ._batch_mi_noise_gate_fused import fused_gate_mi
from ._batch_mi_noise_gate_gpu_helpers import column_major_codes, launch_hist_and_reduce, resident_codes_on_device, tiled_permutation_counts
from ._batch_mi_noise_gate_kernels import (
    _nb_cuda,
    cuda_available,
    cupy,
    cupy_available,
    _cupy_bincount_known_size,  # noqa: F401 - re-exported; read by tests/feature_selection/gpu/test_batch_mi_noise_gate_gpu.py
    _mi_columns_from_counts_cpu,
    _fisher_yates_shuffle,
    _cuda_mi_from_counts_kernel_factory,
    _cuda_su_from_counts_kernel_factory,
    _gate_from_mi,
    _build_shuffle_matrix,
    _cuda_hist_kernel_factory,
    _cuda_hist_kernel_batched_factory,
    _cuda_hist_kernel_batched_shared_factory,
    _cuda_hist_kernel_batched_shared_cm_factory,
    _cuda_shared_mem_per_block,
)

# Lazy-compiled kernel caches (the factories above build on first use; the public
# functions assign into these via ``global``).
_CUDA_MI_KERNEL: "object | None" = None


# ---------------------------------------------------------------------------
# cupy backend
# ---------------------------------------------------------------------------


from mlframe._numba_parallel_guard import parallel_kernel_entry


from ._batch_mi_noise_gate_screen_v1 import (
    _screen_host_codes,
    _bin_range_for_host_discrete,
    batch_mi_with_noise_gate_cupy_v1,
)


def batch_mi_with_noise_gate_cupy(
    disc_2d: np.ndarray,
    factors_nbins: np.ndarray,
    classes_y: np.ndarray,
    classes_y_safe: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    base_seed: np.uint64,
    min_nonzero_confidence: float,
    use_su: bool,
    dtype: type = np.int32,
) -> np.ndarray:
    """GPU-RESIDENT CuPy twin of ``batch_mi_with_noise_gate``. BIT-IDENTICAL.

    Collapses the v1 kernel's O(npermutations) tiny H2D/D2H round-trips to O(1)
    transfers per batch (the win on small consumer GPUs + modest PCIe, where v1
    was transfer/sync-bound - ~83% wall-time at ~18% GPU-util on a GTX 1050 Ti):

      1. ONE H2D up front: ``disc_2d`` (as ``d_base``), the original target codes,
         and the FULL ``(npermutations, n)`` shuffle matrix built on the host with
         the IDENTICAL LCG (``_build_shuffle_matrix``).
      2. ALL permutations' joint histograms are computed ON THE DEVICE, tiled over
         the permutation axis to fit a queried free-memory budget (so the 1050 Ti's
         4GB never OOMs). Each tile bincounts ``(rows_in_tile)`` flattened indices.
      3. ONE D2H per batch: the stacked integer count matrix
         ``(npermutations+1, total_size)`` int64 - a single length-``(P+1)*total``
         copy, NOT a per-permutation count matrix.

    The MI / SU is then reduced from those INTEGER counts on the bit-exact CPU path
    (``_mi_from_counts_cpu``), so the result is identical to v1 AND to the CPU njit
    kernel. Raises ``RuntimeError`` if cupy is unavailable.
    """
    if not cupy_available():
        raise RuntimeError("cupy is not available on this host")
    cp = cupy()

    n, K = int(disc_2d.shape[0]), int(disc_2d.shape[1])
    fe_mi = np.zeros(K, dtype=np.float64)
    if K == 0 or n == 0:
        return fe_mi
    _screen_host_codes(disc_2d, factors_nbins, classes_y, freqs_y, "batch_mi_with_noise_gate_cupy")

    K_y = int(freqs_y.shape[0])
    nbins_arr = np.asarray(factors_nbins, dtype=np.int64)
    per_col_size = nbins_arr * K_y
    offsets = np.zeros(K + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(per_col_size)
    total_size = int(offsets[K])

    # PER-CELL INDEX dtype: d_base / d_idx values are bounded by ``total_size - 1`` (= the
    # sum of ``nbins_k * K_y``), which for any realistic FE batch is << 2^31, so int32 HALVES these
    # (n, K) / (rows, n*K) device buffers vs the old int64 - the (100k, 4096) base is 1.5 GiB int32 vs
    # the 3 GiB int64 that OOM'd the 4GB GPU. GATED: int64 is retained when ``total_size`` could exceed
    # int32 (pathological nbins*K_y), so this is BIT-IDENTICAL - the bincount indices are the same
    # integer values, only the store width narrows. (Distinct from the rejected COUNTS-D2H narrowing
    # below: that narrows the OUTPUT counts; this narrows the INDEX buffers.)
    _INT32_MAX = 2_147_483_647
    idx_dtype = cp.int32 if total_size <= _INT32_MAX else cp.int64

    # ---- ONE H2D: discretized frame + per-(r,k) base index (offsets[k] + disc*K_y).
    d_disc = cp.asarray(np.ascontiguousarray(disc_2d, dtype=np.int32))  # (n, K)
    d_offsets = cp.asarray(offsets[:K].reshape(1, K)).astype(idx_dtype)  # (1, K)
    d_base = d_offsets + d_disc.astype(idx_dtype) * cp.asarray(K_y, dtype=idx_dtype)  # (n, K) idx_dtype

    nperm = int(npermutations) if npermutations and npermutations > 0 else 0
    P1 = nperm + 1  # number of y-vectors = nperm + 1

    # RESIDENT shuffle matrix: row 0 = original y, rows 1.. = the nperm shuffles - built +
    # uploaded ONCE per (target, seed, nperm) and reused across a fit's cupy dispatches, mirroring the
    # numba.cuda resident gate's ``_resident_y_all_device`` cache (this path previously had NO cache analog
    # and rebuilt + re-H2D'd this (P1, n) matrix on every one of a fit's ~30 dispatches). See
    # ``_resident_y_all_device_for_cupy``.
    d_y_all_resident = _resident_y_all_device_for_cupy(classes_y, classes_y_safe, base_seed, nperm, n, P1)
    d_y_all = d_y_all_resident.astype(idx_dtype)  # (P1, n) - y codes < K_y, fit idx_dtype (always a fresh cast, never aliases the cached int32 buffer)

    counts_all = tiled_permutation_counts(cp, d_base, d_y_all, n, K, total_size, P1, idx_dtype)
    del d_base, d_disc, d_offsets, d_y_all

    # ---- Bit-exact CPU MI reduction from the integer counts (same as v1). Per-column
    # reduction batched into one njit call per row (bit-identical, kills dispatch overhead).
    _col_off = offsets[:K]
    _all_pos = np.ones(K, dtype=np.float64)
    original_mi = _mi_columns_from_counts_cpu(np.ascontiguousarray(counts_all[0]), _col_off, nbins_arr, K_y, freqs_y, n, use_su, _all_pos)

    if nperm <= 0:
        return _gate_from_mi(original_mi, [], 0, min_nonzero_confidence)

    # bench-attempt-rejected (2026-06-07): a fused ``if nfailed[k] >= max_failed: continue``
    # early-exit over this host perm reduction (skip doomed columns' remaining
    # _mi_from_counts_cpu) is BYTE-IDENTICAL but gave NO scene wall win (595.75s without
    # -> 641/650s with, 2 runs, idle box): small default npermutations rarely triggers the
    # cutoff while the per-(perm,col) branch costs more than it saves. Keep the simple
    # full-reduction + _gate_from_mi. Re-evaluate only under high-npermutations workloads.
    perm_mis = []
    for i in range(nperm):
        mp = _mi_columns_from_counts_cpu(np.ascontiguousarray(counts_all[i + 1]), _col_off, nbins_arr, K_y, freqs_y, n, use_su, original_mi)
        perm_mis.append(mp)

    return _gate_from_mi(original_mi, perm_mis, nperm, min_nonzero_confidence)


# ---------------------------------------------------------------------------
# numba.cuda backend
# ---------------------------------------------------------------------------

# Lazy-compiled numba.cuda kernel caches (the factories imported from the kernels
# sibling build on first use; the public functions assign into these via ``global``).
_CUDA_HIST_KERNEL: "object | None" = None
_CUDA_SU_KERNEL: "object | None" = None
_CUDA_HIST_KERNEL_BATCHED_SHARED: "object | None" = None
_CUDA_HIST_KERNEL_BATCHED_SHARED_CM: "object | None" = None
_CUDA_HIST_KERNEL_BATCHED: "object | None" = None

# Device shuffled-y matrix cache: y_all = [classes_y; nperm Fisher-Yates shuffles] is
# DETERMINISTIC from (classes_y, classes_y_safe, base_seed, nperm) - and the target is identical across
# ALL ~30 noise-gate dispatches of a fit, so this (P, n) int32 matrix was rebuilt (njit) AND re-uploaded
# (H2D, ~14MB total) every dispatch. Cache the device matrix by WEAKREF IDENTITY of classes_y + the
# scalar key: built+uploaded once per fit, reused thereafter. Bit-identical (same shuffle stream).
# Per-key device cache (was a single slot, which two concurrent noise-gate dispatches with different targets
# clobbered: dispatch B's d_y overwrote A's, so A read B's shuffled-y matrix). Keyed on
# (id(classes_y), base_seed, nperm, n, P) with a co-validating weakref so an id-recycle onto a different
# target array can never false-hit. Bounded FIFO so distinct targets across a long fit don't grow it.
_DY_DEVICE_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()  # key -> (weakref(classes_y), d_y)
_DY_DEVICE_CACHE_MAX = 8
# The get -> move_to_end -> popitem LRU sequence below is
# NOT atomic under the GIL's per-bytecode-op granularity; two MRMR.fit() calls racing in different
# threads of the same process (not itself forbidden anywhere in this module) could violate the
# _DY_DEVICE_CACHE_MAX eviction discipline. The returned device array itself was always race-safe
# (every hit re-validates ref() is classes_y before use), so this is a memory-growth guard, not a
# correctness fix.
_DY_DEVICE_CACHE_LOCK = threading.Lock()


def _histgate_upload(host_arr: np.ndarray, role: str, dtype: Any) -> Any:
    """Upload a host operand for the numba.cuda histgate kernels, content-cached via
    ``resident_operand`` when cupy is installed alongside numba.cuda - a numba.cuda kernel accepts a
    cupy device array directly (CUDA Array Interface; confirmed no host round-trip), so this is a
    drop-in for ``_nb_cuda.to_device`` that additionally dedups fit-constant operands (``freqs_y`` /
    the per-column ``offsets`` / ``nbins``) re-uploaded under a different role elsewhere (the cupy
    backend, other resident-operand FE consumers). Falls back to a plain ``_nb_cuda.to_device`` when
    cupy is absent (a numba.cuda-only host) so that combination is never regressed."""
    if cupy_available():
        from ._fe_resident_operands import resident_operand

        return resident_operand(host_arr, role, dtype=dtype)
    return _nb_cuda.to_device(np.ascontiguousarray(host_arr, dtype=dtype))


def _resident_y_all_device(classes_y, classes_y_safe, base_seed, nperm, n, P):
    """Device (P, n) int32 [orig y; nperm shuffles], cached by (id+weakref(classes_y), base_seed, nperm, n, P).
    Built + uploaded once per (target, key); reused across a fit's noise-gate dispatches for the same target."""
    import weakref

    c = _DY_DEVICE_CACHE
    key = (id(classes_y), int(base_seed), int(nperm), int(n), int(P))
    with _DY_DEVICE_CACHE_LOCK:
        hit = c.get(key)
        if hit is not None:
            ref, d_y = hit
            if ref() is classes_y and d_y is not None:
                c.move_to_end(key)
                return d_y
            c.pop(key, None)  # weakref dead (id recycled onto a different target) -> drop the stale entry
    y_all = np.empty((P, n), dtype=np.int32)
    y_all[0, :] = np.asarray(classes_y, dtype=np.int32)
    if nperm > 0:
        # `_build_shuffle_matrix` is `@njit(parallel=True)` and both residency helpers are reachable
        # from `check_prospective_fe_pairs`, which runs a thread pool. Two threads inside one prange
        # region aborts the process on macOS. Held here rather than at the dispatcher so the
        # serialisation covers the kernel call and not the surrounding GPU work.
        with parallel_kernel_entry():
            y_all[1:, :] = _build_shuffle_matrix(np.asarray(classes_y_safe), np.uint64(base_seed), nperm).astype(np.int32)
    d_y = _nb_cuda.to_device(np.ascontiguousarray(y_all))
    with _DY_DEVICE_CACHE_LOCK:
        try:
            c[key] = (weakref.ref(classes_y), d_y)
            c.move_to_end(key)
            while len(c) > _DY_DEVICE_CACHE_MAX:
                # evict-ok: a miss re-uploads y
                c.popitem(last=False)
        except TypeError:
            c.pop(key, None)
    return d_y


# CUPY-NATIVE twin of ``_DY_DEVICE_CACHE``, used only when numba.cuda is unavailable on this
# host (cupy-only): mirrors the SAME (id+weakref(classes_y), base_seed, nperm, n, P) key + LRU discipline,
# built via ``cp.asarray`` instead of ``_nb_cuda.to_device``.
_DY_DEVICE_CACHE_CUPY: "OrderedDict[tuple, tuple]" = OrderedDict()  # key -> (weakref(classes_y), d_y)
_DY_DEVICE_CACHE_CUPY_MAX = 8
_DY_DEVICE_CACHE_CUPY_LOCK = threading.Lock()  # - see _DY_DEVICE_CACHE_LOCK's note


def _resident_y_all_device_cupy(classes_y, classes_y_safe, base_seed, nperm, n, P):
    """CUPY-native fallback for ``_resident_y_all_device_for_cupy`` on a cupy-only host (no numba.cuda):
    same (id+weakref(classes_y), base_seed, nperm, n, P) keying and LRU eviction, built via ``cp.asarray``."""
    import weakref

    cp = cupy()
    c = _DY_DEVICE_CACHE_CUPY
    key = (id(classes_y), int(base_seed), int(nperm), int(n), int(P))
    with _DY_DEVICE_CACHE_CUPY_LOCK:
        hit = c.get(key)
        if hit is not None:
            ref, d_y = hit
            if ref() is classes_y and d_y is not None:
                c.move_to_end(key)
                return d_y
            c.pop(key, None)  # weakref dead (id recycled onto a different target) -> drop the stale entry
    y_all = np.empty((P, n), dtype=np.int32)
    y_all[0, :] = np.asarray(classes_y, dtype=np.int32)
    if nperm > 0:
        # `_build_shuffle_matrix` is `@njit(parallel=True)` and both residency helpers are reachable
        # from `check_prospective_fe_pairs`, which runs a thread pool. Two threads inside one prange
        # region aborts the process on macOS. Held here rather than at the dispatcher so the
        # serialisation covers the kernel call and not the surrounding GPU work.
        with parallel_kernel_entry():
            y_all[1:, :] = _build_shuffle_matrix(np.asarray(classes_y_safe), np.uint64(base_seed), nperm).astype(np.int32)
    d_y = cp.asarray(np.ascontiguousarray(y_all))
    with _DY_DEVICE_CACHE_CUPY_LOCK:
        try:
            c[key] = (weakref.ref(classes_y), d_y)
            c.move_to_end(key)
            while len(c) > _DY_DEVICE_CACHE_CUPY_MAX:
                # evict-ok: a miss re-uploads y
                c.popitem(last=False)
        except TypeError:
            c.pop(key, None)
    return d_y


def _resident_y_all_device_for_cupy(classes_y, classes_y_safe, base_seed, nperm, n, P):
    """Return a (P, n) int32 CUPY device array of [orig y; nperm shuffles] for ``batch_mi_with_noise_gate_cupy``.

    When numba.cuda is ALSO available, this reuses ``_resident_y_all_device``'s cache via a ZERO-COPY
    ``cp.asarray`` view (confirmed: cupy wraps a foreign CUDA-Array-Interface buffer by device pointer, no
    copy - ``cp_view.data.ptr == numba_device_array.__cuda_array_interface__['data'][0]``) instead of a
    second independent upload, so a cuda dispatch followed by a cupy dispatch on the SAME target (or
    vice versa) shares the identical device buffer - true cross-backend dedup, not just a same-backend
    cache. Falls back to the cupy-native ``_resident_y_all_device_cupy`` when numba.cuda is unavailable
    (a cupy-only host), so that combination still gets the per-fit reuse win."""
    if cuda_available():
        return cupy().asarray(_resident_y_all_device(classes_y, classes_y_safe, base_seed, nperm, n, P))
    return _resident_y_all_device_cupy(classes_y, classes_y_safe, base_seed, nperm, n, P)


def _batch_mi_noise_gate_su_delegate(
    disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y, npermutations, base_seed, min_nonzero_confidence, use_su, dtype, threads_per_block
) -> np.ndarray:
    """SU-entropy noise gate: delegates to the CPU-entropy cuda twin, handing it the cached device shuffle matrix."""
    # SU (Symmetric Uncertainty) has no on-device entropy form, so this delegates to the CPU-entropy
    # ``batch_mi_with_noise_gate_cuda`` twin - but that twin's per-permutation loop used to rebuild
    # (CPU Fisher-Yates) AND re-upload (H2D) the shuffled-y codes from scratch on every one of the
    # ~30 noise-gate dispatches of a fit, even though the target/seed/nperm are fit-constant and the
    # SAME (P, n) shuffle matrix is exactly what ``_resident_y_all_device`` already builds+caches for
    # the non-SU resident path. Build/fetch that cached device matrix here and hand it through
    # (``d_y_all_resident``) so the SU delegate slices its rows directly instead of re-shuffling +
    # re-uploading - BIT-IDENTICAL (same LCG stream; a device-array row slice returns the exact bytes
    # ``_fisher_yates_shuffle`` + ``to_device`` produced, confirmed no host round-trip).
    _n_su = int(disc_2d.shape[0])
    _nperm_su = int(npermutations) if npermutations and npermutations > 0 else 0
    _d_y_all_su = None
    if cuda_available() and _n_su > 0:
        try:
            _d_y_all_su = _resident_y_all_device(classes_y, classes_y_safe, base_seed, _nperm_su, _n_su, _nperm_su + 1)
        except Exception as e:
            log_throttle(
                logger,
                "noise_gate.resident_y",
                logging.WARNING,
                "noise gate: _resident_y_all_device failed (%s), using the per-call shuffle and upload path",
                e,
            )
            _d_y_all_su = None
    return batch_mi_with_noise_gate_cuda(
        disc_2d,
        factors_nbins,
        classes_y,
        classes_y_safe,
        freqs_y,
        npermutations,
        base_seed,
        min_nonzero_confidence,
        use_su,
        dtype,
        128 if threads_per_block is None else threads_per_block,
        d_y_all_resident=_d_y_all_su,
    )


_NOISE_GATE_BACKEND_OVERRIDE: "str | None" = None  # "bits" / "hist" while the kernel-tuning sweep times one backend; None -> the tuning-cache choice


def force_noise_gate_backend(value: "str | None") -> "str | None":
    """Force the resident gate's backend (``"bits"``, ``"hist"``) or release the force (``None``); returns the previous setting so a caller can restore it."""
    global _NOISE_GATE_BACKEND_OVERRIDE
    previous = _NOISE_GATE_BACKEND_OVERRIDE
    _NOISE_GATE_BACKEND_OVERRIDE = value
    return previous


def _noise_gate_bits_matrix(d_disc, nbins_arr, d_y, K_y, n, K):
    """``(P, K)`` MI matrix from device bit sets when the backend choice is the bit-set gate and the kernel applies, else ``None`` (the histogram pair then runs)."""
    forced = _NOISE_GATE_BACKEND_OVERRIDE
    if forced == "hist":
        return None
    from ._batch_mi_noise_gate_bits import bits_gate_mi, noise_gate_backend_choice

    if forced is None and noise_gate_backend_choice(n, K, K_y, int(np.max(nbins_arr))) != "bits":
        return None
    mi = bits_gate_mi(d_disc, nbins_arr, d_y, K_y)
    if mi is None and forced == "bits":
        raise RuntimeError("the bit-set noise gate declined this shape")
    return mi


def batch_mi_with_noise_gate_cuda_resident(
    disc_2d: np.ndarray,
    factors_nbins: np.ndarray,
    classes_y: np.ndarray,
    classes_y_safe: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    base_seed: np.uint64,
    min_nonzero_confidence: float,
    use_su: bool,
    dtype: type = np.int32,
    threads_per_block: "int | None" = None,
    d_disc_resident: cupy.ndarray | None = None,
) -> np.ndarray:
    """FULL-GPU-RESIDENT noise gate: batched histogram (all perms, one launch) -> GPU MI kernel -> only
    the (P, K) MI matrix D2H -> host gate decision. The entire permutation noise gate runs on the device
    from resident counts (no per-perm counts D2H, no CPU entropy). Selection-equivalent to
    ``batch_mi_with_noise_gate_cuda`` (same perm gate; GPU MI reproduces the CPU reduction order to fp
    round-off). ``use_su`` has no GPU entropy form here -> delegates to the CPU-entropy cuda path.

    RESIDENT-CODES HANDOFF (2026-06-21, gated): ``d_disc_resident`` - when the FE chunk binned the codes
    ON the GPU and kept them resident - is the (n, K) DEVICE codes array (cupy or numba-cuda; consumed via
    the CUDA Array Interface). Passing it lets the histogram kernel read the codes IN PLACE, skipping the
    H2D re-upload of ``disc_2d`` (the codes were produced on the GPU, so re-uploading them is a pointless
    round-trip). It MUST hold the SAME integer codes as ``disc_2d`` (the producer keeps the exact bytes it
    D2H'd) -> counts, MI and the gate decision are BIT-IDENTICAL to the host-codes path. ``None`` (the
    default, and any narrow-dtype mismatch) keeps the H2D-from-host path unchanged."""
    global _CUDA_SU_KERNEL
    if use_su:
        # Symmetric uncertainty runs from the same resident counts as MI, through its own reduction kernel; the host-entropy delegate stays the fallback
        # when that kernel cannot be built.
        if _CUDA_SU_KERNEL is None and cuda_available():
            _CUDA_SU_KERNEL = _cuda_su_from_counts_kernel_factory()
        if _CUDA_SU_KERNEL is None:
            return _batch_mi_noise_gate_su_delegate(
                disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y, npermutations, base_seed, min_nonzero_confidence, use_su, dtype, threads_per_block
            )
    global _CUDA_HIST_KERNEL_BATCHED, _CUDA_HIST_KERNEL_BATCHED_SHARED, _CUDA_HIST_KERNEL_BATCHED_SHARED_CM, _CUDA_MI_KERNEL
    if not cuda_available():
        raise RuntimeError("numba.cuda is not available on this host")
    if _CUDA_HIST_KERNEL_BATCHED is None:
        _CUDA_HIST_KERNEL_BATCHED = _cuda_hist_kernel_batched_factory()
    if _CUDA_HIST_KERNEL_BATCHED_SHARED is None:
        _CUDA_HIST_KERNEL_BATCHED_SHARED = _cuda_hist_kernel_batched_shared_factory()
    if _CUDA_HIST_KERNEL_BATCHED_SHARED_CM is None:
        _CUDA_HIST_KERNEL_BATCHED_SHARED_CM = _cuda_hist_kernel_batched_shared_cm_factory()
    if _CUDA_MI_KERNEL is None:
        _CUDA_MI_KERNEL = _cuda_mi_from_counts_kernel_factory()
    if _CUDA_HIST_KERNEL_BATCHED is None or _CUDA_MI_KERNEL is None:
        raise RuntimeError("numba.cuda kernel factory failed to build")

    n = int(disc_2d.shape[0])
    K = int(disc_2d.shape[1])
    fe_mi = np.zeros(K, dtype=np.float64)
    if K == 0 or n == 0:
        return fe_mi

    # OOB SCREEN (FIX2): the batched-hist kernels use raw codes DIRECTLY as a flat shared-mem
    # / counts offset (``cx * K_y + cy`` - _batch_mi_noise_gate_kernels.py:428/482) sized from
    # nbins_col[k] and K_y. A -1 sentinel or a code >= its cardinality indexes outside the histogram ->
    # cudaErrorIllegalAddress (a hard GPU crash). Screen the HOST codes here (cheap min/max) so an
    # upstream OOB surfaces as a clear ValueError. Skipped when the codes are resident (binner-produced,
    # already on-device): re-syncing them would defeat the resident-handoff and they are dense by contract.
    _bin_range_for_host_discrete(d_disc_resident, disc_2d, factors_nbins, classes_y, freqs_y)
    # Lever C: HW-aware + KTC-tuned threads/block for the batched-hist launch (default 128 left
    # the SM ~1/16 occupied; the row-loop parallelises across threads). When the caller did not pin a count
    # (threads_per_block is None) look the per-host tuned count up from the kernel_tuning_cache (candidate set
    # derived from device occupancy); falls back to 128 on any failure. Block size NEVER changes the integer
    # counts -> MI + gate decision bit-identical. The KTC probe passes an explicit count (skips the lookup).
    if threads_per_block is None:
        try:
            from ._gpu_resident_histgate_ktc import histgate_threads

            threads_per_block = int(histgate_threads(n))
        # best-effort: the block size never changes the integer counts, so 128 gives the same MI and gate decision
        except Exception as e:  # best-effort: the block size never changes the integer counts
            logger.debug("histgate_threads() lookup failed, using the default 128: %s", e)
            threads_per_block = 128
    K_y = int(freqs_y.shape[0])
    nbins_arr = np.asarray(factors_nbins, dtype=np.int64)
    per_col_size = nbins_arr * K_y
    offsets = np.zeros(K + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(per_col_size)
    total_size = int(offsets[K])
    nperm = int(npermutations) if npermutations and npermutations > 0 else 0
    P = nperm + 1

    # d_y (P, n) device shuffle matrix - cached once per fit (target constant across dispatches).
    d_y = _resident_y_all_device(classes_y, classes_y_safe, base_seed, nperm, n, P)

    d_disc = resident_codes_on_device(d_disc_resident, disc_2d, n, K)
    # offsets/nbins/freqs_y ARE fit-constant (the target's class table + the per-column bin-count layout
    # recur across a fit's dispatches) - route through the content-keyed resident cache instead of a
    # fresh to_device every call (see _histgate_upload).
    d_off = _histgate_upload(offsets[:K], "histgate_off", np.int64)
    d_nb = _histgate_upload(nbins_arr, "histgate_nb", np.int32)
    d_freq = _histgate_upload(freqs_y, "histgate_freq", np.float64)
    # Bit-set gate (``_batch_mi_noise_gate_bits``): the (P, K) MI matrix from popcounts of device bit sets, 7x faster than the histogram pair at 10 bins x 10 classes; the tuning cache decides.
    if not use_su:
        _bits_mi = _noise_gate_bits_matrix(d_disc, nbins_arr, d_y, K_y, n, K)
        if _bits_mi is not None:
            return _gate_from_mi(_bits_mi[0], _bits_mi[1:], nperm, min_nonzero_confidence) if nperm > 0 else _gate_from_mi(_bits_mi[0], [], 0, min_nonzero_confidence)
    # Fused histogram + reduction: no (P, total_size) counts matrix (444 MB at 27744 candidates x 100 shuffles), one launch; falls through when it does not apply.
    _fused_mi = fused_gate_mi(d_disc, nbins_arr, d_nb, d_y, d_freq, freqs_y, n, K, K_y, P, threads_per_block, use_su)
    if _fused_mi is not None:
        return _gate_from_mi(_fused_mi[0], _fused_mi[1:], nperm, min_nonzero_confidence) if nperm > 0 else _gate_from_mi(_fused_mi[0], [], 0, min_nonzero_confidence)
    d_counts = _nb_cuda.device_array(P * total_size, dtype=np.int64)
    d_ref = _nb_cuda.to_device(np.ones(K, dtype=np.float64))  # compute every (k,p); host applies the gate
    d_out = _nb_cuda.device_array((P, K), dtype=np.float64)

    # Shared-mem privatized hist when the per-column histogram (max nb_k * K_y int32) fits the device
    # shared budget (kills the global-atomic contention the metrics exposed); else the global-atomic
    # kernel (any-cardinality fallback). Bit-identical either way.
    _max_hist = int(nbins_arr.max()) * K_y if K > 0 else 0
    _sh_bytes = _max_hist * 4
    _sh_budget = _cuda_shared_mem_per_block()
    _use_shared = _CUDA_HIST_KERNEL_BATCHED_SHARED is not None and _sh_budget > 0 and 0 < _sh_bytes <= int(_sh_budget * 0.75)

    _d_disc_cm = column_major_codes(d_disc, _use_shared, _CUDA_HIST_KERNEL_BATCHED_SHARED_CM)
    # The shared kernel OVERWRITES every counts slot (flushes its full [off:off+nb_k*K_y] slice), so the
    # host zeroing is redundant there; the global-atomic kernel needs the zeroed buffer.
    if not _use_shared:
        d_counts[:] = 0

    launch_hist_and_reduce(
        _use_shared, _d_disc_cm, d_disc, d_off, d_nb, d_y, d_counts, n, K, K_y, total_size, P, threads_per_block, _sh_bytes,
        _CUDA_HIST_KERNEL_BATCHED_SHARED_CM, _CUDA_HIST_KERNEL_BATCHED_SHARED, _CUDA_HIST_KERNEL_BATCHED, _CUDA_MI_KERNEL, _CUDA_SU_KERNEL,
        use_su, d_freq, d_ref, d_out, freqs_y,
    )
    out_mi = d_out.copy_to_host()  # (P, K) - the only sizeable D2H
    original_mi = out_mi[0]
    if nperm <= 0:
        return _gate_from_mi(original_mi, [], 0, min_nonzero_confidence)
    return _gate_from_mi(original_mi, out_mi[1:], nperm, min_nonzero_confidence)


def batch_mi_with_noise_gate_cuda(
    disc_2d: np.ndarray,
    factors_nbins: np.ndarray,
    classes_y: np.ndarray,
    classes_y_safe: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    base_seed: np.uint64,
    min_nonzero_confidence: float,
    use_su: bool,
    dtype: type = np.int32,
    threads_per_block: int = 128,
    d_y_all_resident: "Any | None" = None,
) -> np.ndarray:
    """numba.cuda GPU twin of ``batch_mi_with_noise_gate``. BIT-IDENTICAL.

    GPU work: a global-atomic joint-histogram kernel (one block per column) over
    the original y and each CPU-shuffled y. MI reduced from the integer counts on
    the CPU via ``_mi_from_counts_cpu`` (bit-exact). Raises ``RuntimeError`` if
    numba.cuda is unavailable.

    ``d_y_all_resident`` - an optional cached (P, n) int32 device matrix (row 0 = original ``classes_y``,
    rows 1.. = the ``nperm`` Fisher-Yates shuffles), typically ``_resident_y_all_device``'s return value.
    When given, each per-target pass slices the matching row directly (a device view, no host round-trip)
    INSTEAD OF rebuilding the shuffle on the CPU and re-uploading it - the SU delegate in
    ``batch_mi_with_noise_gate_cuda_resident`` passes this so the fit-constant shuffle stream is uploaded
    ONCE per fit instead of once per dispatch. Row slicing returns the exact bytes the CPU
    shuffle+``to_device`` path would have produced (same LCG stream), so this is bit-identical; ``None``
    (the default) keeps the original per-call CPU-shuffle + H2D path unchanged.
    """
    global _CUDA_HIST_KERNEL
    if not cuda_available():
        raise RuntimeError("numba.cuda is not available on this host")
    if _CUDA_HIST_KERNEL is None:
        _CUDA_HIST_KERNEL = _cuda_hist_kernel_factory()
        if _CUDA_HIST_KERNEL is None:
            raise RuntimeError("numba.cuda kernel factory failed to build")

    n = int(disc_2d.shape[0])
    K = int(disc_2d.shape[1])
    fe_mi = np.zeros(K, dtype=np.float64)
    if K == 0 or n == 0:
        return fe_mi
    _screen_host_codes(disc_2d, factors_nbins, classes_y, freqs_y, "batch_mi_with_noise_gate_cuda")

    K_y = int(freqs_y.shape[0])
    nbins_arr = np.asarray(factors_nbins, dtype=np.int64)
    per_col_size = nbins_arr * K_y
    offsets = np.zeros(K + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(per_col_size)
    total_size = int(offsets[K])

    # disc_2d is the FE-candidate matrix - varies per dispatch (a different candidate-column chunk each
    # time), so it is NOT fit-constant and is left as a plain per-call upload (see the twin resident
    # function's identical reasoning for its own d_disc fallback).
    d_disc = _nb_cuda.to_device(np.ascontiguousarray(disc_2d, dtype=np.int32))
    # offsets/nbins ARE fit-constant across a fit's dispatches -> content-keyed resident cache.
    d_off = _histgate_upload(offsets[:K], "histgate_off", np.int64)
    d_nb = _histgate_upload(nbins_arr, "histgate_nb", np.int32)

    # One block per column; for tiny K the grid is small (low-occupancy) - the
    # cupy backend is preferred at the realistic large-batch sizes where GPU wins,
    # so silence the cosmetic NumbaPerformanceWarning here.
    from mlframe.utils.warning_filters import silenced_once as _silenced_once

    try:
        from numba.core.errors import NumbaPerformanceWarning as _NbPerfWarn
    except ImportError:
        _NbPerfWarn = Warning

    def _counts_from_device_y(d_y) -> np.ndarray:
        """Launch the joint-histogram CUDA kernel against an ALREADY-RESIDENT target-codes device array, returning the flat per-column counts array copied back to host."""
        d_counts = _nb_cuda.device_array(total_size, dtype=np.int64)
        d_counts[:] = 0
        with _silenced_once(_NbPerfWarn, r"numba\.cuda"):
            _CUDA_HIST_KERNEL[K, threads_per_block](d_disc, d_off, d_nb, d_y, d_counts, n, K_y)  # type: ignore[index]  # numba cuda kernel[grid, block] launch syntax, not real indexing
        return np.asarray(d_counts.copy_to_host())

    def _counts_for(y_codes_host: np.ndarray) -> np.ndarray:
        """Upload one target's discretized codes to device and launch the joint-histogram CUDA kernel across all K candidate columns, returning the flat per-column counts array copied back to host."""
        d_y = _nb_cuda.to_device(np.ascontiguousarray(y_codes_host, dtype=np.int32))
        return _counts_from_device_y(d_y)

    _col_off = offsets[:K]
    _all_pos = np.ones(K, dtype=np.float64)  # original-y pass: compute every column
    if d_y_all_resident is not None:
        counts_orig = _counts_from_device_y(d_y_all_resident[0])
    else:
        counts_orig = _counts_for(np.asarray(classes_y, dtype=np.int32))
    original_mi = _mi_columns_from_counts_cpu(counts_orig, _col_off, nbins_arr, K_y, freqs_y, n, use_su, _all_pos)

    if npermutations <= 0:
        return _gate_from_mi(original_mi, [], 0, min_nonzero_confidence)

    cy_safe = np.asarray(classes_y_safe)
    # bench-attempt-rejected (2026-06-07): perm-reduction early-exit (see cupy-v2 path) -
    # BYTE-IDENTICAL but no scene wall win; keep full reduction + _gate_from_mi.
    # The per-column MI reduction is batched into one njit call per permutation
    # (_mi_columns_from_counts_cpu) - skips columns where original_mi<=0 exactly as the
    # old inner loop did; bit-identical, kills the 224k per-call dispatch overhead.
    perm_mis = []
    for i in range(npermutations):
        if d_y_all_resident is not None:
            counts_p = _counts_from_device_y(d_y_all_resident[i + 1])
        else:
            shuffled = _fisher_yates_shuffle(cy_safe, np.uint64(base_seed), i)
            counts_p = _counts_for(np.asarray(shuffled, dtype=np.int32))
        mp = _mi_columns_from_counts_cpu(counts_p, _col_off, nbins_arr, K_y, freqs_y, n, use_su, original_mi)
        perm_mis.append(mp)

    return _gate_from_mi(original_mi, perm_mis, npermutations, min_nonzero_confidence)


# Backend chooser + tuner (mirrors batch_pair_mi_gpu.py). Sweep grid, code-version, and the
# get_or_tune dispatch decision live in the sibling module; carved out to keep this file under
# the 1k-LOC ceiling. Re-exported below so every import path resolves.
from ._batch_mi_noise_gate_tuning import (
    _BMING_SWEEP_N_ROWS,
    _BMING_SWEEP_N_COLS,
    _BMING_SALT,
    _batch_mi_noise_gate_code_version,
    _batch_mi_noise_gate_backend_choice,
    _batch_mi_noise_gate_fallback_choice,  # noqa: F401 - re-exported; consumed by ..filters._feature_engineering_pairs._pairs_dispatch
    _run_batch_mi_noise_gate_sweep,
    ensure_batch_mi_noise_gate_tuning,
)


def dispatch_batch_mi_with_noise_gate_gpu(
    disc_2d: np.ndarray,
    factors_nbins: np.ndarray,
    classes_y: np.ndarray,
    classes_y_safe: np.ndarray,
    freqs_y: np.ndarray,
    npermutations: int,
    base_seed: np.uint64,
    min_nonzero_confidence: float,
    use_su: bool,
    dtype: type = np.int32,
    force_backend: str | None = None,
) -> tuple[np.ndarray, str] | None:
    """Run the chosen GPU backend, returning ``(fe_mi, backend_name)`` or ``None``
    when GPU is unavailable / not chosen (so the caller falls back to the CPU
    kernel). Mirrors ``dispatch_batch_pair_mi``'s force_backend + availability
    guards. The CPU kernel is NOT run here - the caller owns the CPU path.
    """
    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        # This dispatcher never had its own
        # MLFRAME_DISABLE_GPU/CUDA_VISIBLE_DEVICES self-check, unlike its dispatch_batch_pair_mi and
        # dispatch_friend_graph_stats siblings - it relied entirely on its caller checking first.
        return None

    n = int(disc_2d.shape[0])
    K = int(disc_2d.shape[1])

    # ABSOLUTE cushion guard: on a near-full / SHARED card return None so the caller runs the CPU
    # kernel, BEFORE the per-tile relative ``free_b * 0.35`` budget below (computed only after the pool may have
    # already eaten the device). The dominant device buffer is the (rows, n*K) int index array; estimate one
    # y-row's worth as the cushion's bytes_needed. Pure ADD - tightens, never loosens; permissive without cupy.
    try:
        from ._fe_gpu_vram import fe_gpu_has_vram_cushion

        if not fe_gpu_has_vram_cushion(n * max(K, 1) * 8):
            return None
    except Exception as e:  # nosec B110 - best-effort/optional path, no module logger
        logger.debug("fe_gpu_has_vram_cushion() check failed, proceeding without the VRAM cushion gate: %s", e)

    if force_backend is not None:
        fb = force_backend.lower()
        if fb == "cupy" and cupy_available():
            return (
                batch_mi_with_noise_gate_cupy(
                    disc_2d,
                    factors_nbins,
                    classes_y,
                    classes_y_safe,
                    freqs_y,
                    npermutations,
                    base_seed,
                    min_nonzero_confidence,
                    use_su,
                    dtype,
                ),
                "cupy",
            )
        if fb == "cuda" and cuda_available():
            return (
                batch_mi_with_noise_gate_cuda(
                    disc_2d,
                    factors_nbins,
                    classes_y,
                    classes_y_safe,
                    freqs_y,
                    npermutations,
                    base_seed,
                    min_nonzero_confidence,
                    use_su,
                    dtype,
                ),
                "cuda",
            )
        return None  # forced CPU (or unavailable forced backend) -> caller uses CPU

    choice = _batch_mi_noise_gate_backend_choice(n, K)

    if choice == "cupy" and cupy_available():
        try:
            return (
                batch_mi_with_noise_gate_cupy(
                    disc_2d,
                    factors_nbins,
                    classes_y,
                    classes_y_safe,
                    freqs_y,
                    npermutations,
                    base_seed,
                    min_nonzero_confidence,
                    use_su,
                    dtype,
                ),
                "cupy",
            )
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed: %s", e)
            pass
    if choice == "cuda" and cuda_available():
        try:
            return (
                batch_mi_with_noise_gate_cuda(
                    disc_2d,
                    factors_nbins,
                    classes_y,
                    classes_y_safe,
                    freqs_y,
                    npermutations,
                    base_seed,
                    min_nonzero_confidence,
                    use_su,
                    dtype,
                ),
                "cuda",
            )
        except Exception as e:
            # numba's CudaAPIError/CudaDriverError derive
            # directly from Exception, not RuntimeError - broaden to match the already-fixed
            # dispatch_batch_pair_mi sibling so a genuine CUDA driver fault can't escape uncaught.
            logging.getLogger(__name__).debug("batch_mi_noise_gate cuda backend failed (%s); falling back to CPU", e)
            pass

    return None  # CPU region (or GPU failed) -> caller runs the CPU kernel


# Register with the @kernel_tuner registry so retune_all / mlframe-tune-kernels
# discover + batch-tune batch_mi_noise_gate. GPU-capable (cuda/cupy backends).
try:
    from pyutilz.performance.kernel_tuning.registry import kernel_tuner

    kernel_tuner(
        kernel_name="batch_mi_noise_gate",
        variant_fns=(_cpu_batch_mi_with_noise_gate,),  # CPU body; GPU covered by salt
        tuner=_run_batch_mi_noise_gate_sweep,
        axes={"n_rows": list(_BMING_SWEEP_N_ROWS), "n_cols": list(_BMING_SWEEP_N_COLS)},
        fallback={"backend_choice": "cpu"},
        gpu_capable=True,
        salt=_BMING_SALT,
        cli_label="batch_mi_noise_gate",
    )
except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
    logging.getLogger(__name__).debug("suppressed: %s", e)
    pass


def __getattr__(name: str):
    """Keep the former ``_CUDA_AVAIL`` / ``_CUPY_AVAIL`` re-exports working, resolved through the lazy accessors."""
    if name == "_CUDA_AVAIL":
        return cuda_available()
    if name == "_CUPY_AVAIL":
        return cupy_available()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "batch_mi_with_noise_gate_cupy",
    "batch_mi_with_noise_gate_cupy_v1",
    "batch_mi_with_noise_gate_cuda",
    "dispatch_batch_mi_with_noise_gate_gpu",
    "_batch_mi_noise_gate_code_version",
    "_batch_mi_noise_gate_backend_choice",
    "_run_batch_mi_noise_gate_sweep",
    "ensure_batch_mi_noise_gate_tuning",
    # Served by the module ``__getattr__`` above, not by a module-level binding, which is why ruff cannot see them.
    "_CUDA_AVAIL",  # noqa: F822
    "_CUPY_AVAIL",  # noqa: F822
]
