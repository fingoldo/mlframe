"""GPU backend for the batched FE-candidate MI + permutation noise-gate kernel.

Twin of the CPU njit ``batch_mi_with_noise_gate`` (``info_theory.py``), modeled
EXACTLY on ``batch_pair_mi_gpu.py``: two GPU joint-histogram backends (cupy +
numba.cuda), a per-host backend chooser via ``get_or_tune``, a measurement-backed
fallback heuristic, a real CPU-vs-GPU sweep tuner, an availability-guarded
dispatcher, and a ``@kernel_tuner`` registration.

BIT-IDENTITY (non-negotiable -- FE recovery is pinned)
------------------------------------------------------
The noise-gate rejection is a float comparison (``mi_perm >= original_mi``), so a
GPU MI that drifts by a single ULP would flip borderline rejections and change
which engineered features MRMR keeps. To stay bit-identical we split the work:

  1. The ``npermutations`` y-shuffles run on the CPU with the IDENTICAL LCG /
     Fisher-Yates stream the CPU kernel uses (``base_seed*2654435761 + (i+1)``
     then the PCG step). Only 3 small shuffles by default -- cheap.
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

import os
import time
import logging
from collections import OrderedDict

import numpy as np

# CPU reference kernel (the required win + always-correct fallback).
from .info_theory import batch_mi_with_noise_gate as _cpu_batch_mi_with_noise_gate

# Compute kernels (GPU dep probes, cupy/cuda kernel factories + their compile helpers,
# and the bit-exact CPU njit reducers) live in the sibling module; carved out to keep
# this module under the 1k-LOC ceiling. Re-exported below so every import path resolves.
from ._batch_mi_noise_gate_kernels import (
    _nb_cuda,
    _CUDA_AVAIL,
    _cp,
    _CUPY_AVAIL,
    _cupy_bincount_known_size,
    _mi_from_counts_cpu,
    _mi_columns_from_counts_cpu,
    _fisher_yates_shuffle,
    _cuda_mi_from_counts_kernel_factory,
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


def batch_mi_with_noise_gate_cupy_v1(
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
    """CuPy GPU twin of ``batch_mi_with_noise_gate``. BIT-IDENTICAL.

    PRIOR (v1) kernel -- kept per the keep-all-kernel-versions rule so it can be
    re-benched on bigger GPUs. SUPERSEDED on small/consumer cards by
    ``batch_mi_with_noise_gate_cupy`` (GPU-resident, O(1) transfers per batch).

    bench-note (GTX 1050 Ti, 4GB Pascal cc6.1, n=2407 K=2000 nperm=3): this v1
    does 1 H2D + 1 D2H PER permutation (cp.asarray(shuffled) + cp.asnumpy(flat)),
    so a self-profile of MRMR.fit spent 82.7% wall-time here at ~18% GPU-util --
    transfer/sync-bound. The resident rewrite collapses that to O(1) transfers.

    GPU work: for the original y and each CPU-shuffled y, compute the joint
    histogram ``(nbins_k, K_y)`` for every candidate column via a single batched
    ``cupy.bincount`` over the flattened ``(col_code * K_y + y_code)`` index. The
    MI is reduced from those INTEGER counts on the CPU via ``_mi_from_counts_cpu``
    (bit-exact). Raises ``RuntimeError`` if cupy is unavailable.
    """
    if not _CUPY_AVAIL:
        raise RuntimeError("cupy is not available on this host")
    cp = _cp

    n = int(disc_2d.shape[0])
    K = int(disc_2d.shape[1])
    fe_mi = np.zeros(K, dtype=np.float64)
    if K == 0 or n == 0:
        return fe_mi

    K_y = int(freqs_y.shape[0])
    nbins_arr = np.asarray(factors_nbins, dtype=np.int64)

    # Move the discretized frame to device ONCE (int32). The per-column joint
    # code is ``col_code * K_y + y_code``; flattening the whole frame lets a
    # single bincount per shuffled-y collect all columns at once.
    d_disc = cp.asarray(np.ascontiguousarray(disc_2d, dtype=np.int32))  # (n, K)
    # Column-block offsets so column k occupies counts[k] = bins [off[k], off[k]+nb_k*K_y).
    per_col_size = nbins_arr * K_y  # (K,) joint size per column
    offsets = np.zeros(K + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(per_col_size)
    total_size = int(offsets[K])
    # PER-CELL INDEX dtype (2026-06-20, parallel to the resident kernel): v1 scores ONE y at a time, so
    # d_base / d_idx are bounded by ``total_size - 1`` -- int32 HALVES these (n, K) buffers vs int64 when
    # that fits (gated -> bit-identical, the bincount indices are the same values). Kept for re-bench
    # parity with the resident kernel.
    idx_dtype = cp.int32 if total_size <= 2_147_483_647 else cp.int64
    d_offsets = cp.asarray(offsets[:K].reshape(1, K)).astype(idx_dtype)  # (1, K) broadcastable

    # Column index of each cell, broadcast (n, K).
    # global_index[r, k] = offsets[k] + disc[r,k]*K_y + y_code[r]
    # We build the per-(r,k) base ``offsets[k] + disc[r,k]*K_y`` once, then add
    # the y-code (which changes per shuffle) before each bincount.
    d_base = d_offsets + d_disc.astype(idx_dtype) * cp.asarray(K_y, dtype=idx_dtype)  # (n, K) idx_dtype

    def _joint_counts_for(y_codes_host: np.ndarray) -> np.ndarray:
        """Return a host list of (nbins_k, K_y) int64 count matrices for all K cols
        against ``y_codes_host`` (length n)."""
        d_y = cp.asarray(np.ascontiguousarray(y_codes_host, dtype=np.int64)).reshape(n, 1).astype(idx_dtype)
        d_idx = (d_base + d_y).reshape(-1)  # (n*K,) flat global indices (idx_dtype; <= total_size-1)
        # OPT-D: known-size bincount (skip cupy.bincount's host-sync validations);
        # byte-identical counts. ``total_size`` is exact, ``d_idx`` non-negative by construction.
        flat = _cupy_bincount_known_size(d_idx, total_size)
        return np.asarray(cp.asnumpy(flat))  # (total_size,) int64 on host

    # Original MI. Per-column MI reduction batched into one njit call (bit-identical).
    _col_off = offsets[:K]
    _all_pos = np.ones(K, dtype=np.float64)
    counts_orig = _joint_counts_for(np.asarray(classes_y, dtype=np.int64))
    original_mi = _mi_columns_from_counts_cpu(counts_orig, _col_off, nbins_arr, K_y, freqs_y, n, use_su, _all_pos)

    if npermutations <= 0:
        return _gate_from_mi(original_mi, [], 0, min_nonzero_confidence)

    cy_safe = np.asarray(classes_y_safe)
    # bench-attempt-rejected (2026-06-07): perm-reduction early-exit (see cupy-v2 path) --
    # BYTE-IDENTICAL but no scene wall win; keep full reduction + _gate_from_mi.
    perm_mis = []
    for i in range(npermutations):
        shuffled = _fisher_yates_shuffle(cy_safe, np.uint64(base_seed), i)
        counts_p = _joint_counts_for(np.asarray(shuffled, dtype=np.int64))
        mp = _mi_columns_from_counts_cpu(counts_p, _col_off, nbins_arr, K_y, freqs_y, n, use_su, original_mi)
        perm_mis.append(mp)

    return _gate_from_mi(original_mi, perm_mis, npermutations, min_nonzero_confidence)


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
    was transfer/sync-bound -- ~83% wall-time at ~18% GPU-util on a GTX 1050 Ti):

      1. ONE H2D up front: ``disc_2d`` (as ``d_base``), the original target codes,
         and the FULL ``(npermutations, n)`` shuffle matrix built on the host with
         the IDENTICAL LCG (``_build_shuffle_matrix``).
      2. ALL permutations' joint histograms are computed ON THE DEVICE, tiled over
         the permutation axis to fit a queried free-memory budget (so the 1050 Ti's
         4GB never OOMs). Each tile bincounts ``(rows_in_tile)`` flattened indices.
      3. ONE D2H per batch: the stacked integer count matrix
         ``(npermutations+1, total_size)`` int64 -- a single length-``(P+1)*total``
         copy, NOT a per-permutation count matrix.

    The MI / SU is then reduced from those INTEGER counts on the bit-exact CPU path
    (``_mi_from_counts_cpu``), so the result is identical to v1 AND to the CPU njit
    kernel. Raises ``RuntimeError`` if cupy is unavailable.
    """
    if not _CUPY_AVAIL:
        raise RuntimeError("cupy is not available on this host")
    cp = _cp

    n = int(disc_2d.shape[0])
    K = int(disc_2d.shape[1])
    fe_mi = np.zeros(K, dtype=np.float64)
    if K == 0 or n == 0:
        return fe_mi

    K_y = int(freqs_y.shape[0])
    nbins_arr = np.asarray(factors_nbins, dtype=np.int64)
    per_col_size = nbins_arr * K_y
    offsets = np.zeros(K + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(per_col_size)
    total_size = int(offsets[K])

    # PER-CELL INDEX dtype (2026-06-20): d_base / d_idx values are bounded by ``total_size - 1`` (= the
    # sum of ``nbins_k * K_y``), which for any realistic FE batch is << 2^31, so int32 HALVES these
    # (n, K) / (rows, n*K) device buffers vs the old int64 -- the (100k, 4096) base is 1.5 GiB int32 vs
    # the 3 GiB int64 that OOM'd the 4GB GPU. GATED: int64 is retained when ``total_size`` could exceed
    # int32 (pathological nbins*K_y), so this is BIT-IDENTICAL -- the bincount indices are the same
    # integer values, only the store width narrows. (Distinct from the rejected COUNTS-D2H narrowing
    # below: that narrows the OUTPUT counts; this narrows the INDEX buffers.)
    _INT32_MAX = 2_147_483_647
    idx_dtype = cp.int32 if total_size <= _INT32_MAX else cp.int64

    # ---- ONE H2D: discretized frame + per-(r,k) base index (offsets[k] + disc*K_y).
    d_disc = cp.asarray(np.ascontiguousarray(disc_2d, dtype=np.int32))  # (n, K)
    d_offsets = cp.asarray(offsets[:K].reshape(1, K)).astype(idx_dtype)  # (1, K)
    d_base = d_offsets + d_disc.astype(idx_dtype) * cp.asarray(K_y, dtype=idx_dtype)  # (n, K) idx_dtype

    nperm = int(npermutations) if npermutations and npermutations > 0 else 0

    # ---- ONE H2D: the original codes + the full shuffle matrix, stacked as the
    # set of y-vectors to score: row 0 = original y, rows 1.. = the nperm shuffles.
    y_orig = np.ascontiguousarray(classes_y, dtype=np.int64).reshape(1, n)
    if nperm > 0:
        shuf = _build_shuffle_matrix(np.asarray(classes_y_safe), np.uint64(base_seed), nperm)
        y_all_host = np.empty((nperm + 1, n), dtype=np.int64)
        y_all_host[0, :] = y_orig[0, :]
        y_all_host[1:, :] = shuf.astype(np.int64)
    else:
        y_all_host = y_orig
    P1 = y_all_host.shape[0]  # number of y-vectors = nperm + 1
    d_y_all = cp.asarray(y_all_host).astype(idx_dtype)  # (P1, n) -- y codes < K_y, fit idx_dtype

    # ---- 4GB tiling over the permutation axis. Each tile of ``rows`` y-vectors
    # materialises a (rows, n) int64 index array + a (rows, total_size) int64 count
    # array on device. Budget the bigger of the two (~ rows * max(n, total_size) * 8B)
    # against queried free memory (use a fraction to leave headroom for d_base etc.).
    try:
        free_b, _tot_b = cp.cuda.runtime.memGetInfo()
    except Exception:
        free_b = 512 * 1024 * 1024  # conservative default
    budget = int(free_b * 0.35)
    # Per y-row device cost of a tile: the flat index array (n*K int64) + the
    # bincount output slot (total_size int64) + the y-codes (n int64). The n*K
    # index array dominates; keep the divisor honest so the 1050 Ti never OOMs.
    bytes_per_row = 8 * (n * K + total_size + n)
    rows_per_tile = max(1, budget // max(1, bytes_per_row))
    if rows_per_tile > P1:
        rows_per_tile = P1
    if rows_per_tile < P1:
        import logging
        logging.getLogger(__name__).info(
            "batch_mi_noise_gate cupy: tiling %d y-vectors into tiles of %d (free=%dMB, n=%d, total_size=%d)",
            P1, rows_per_tile, free_b // (1024 * 1024), n, total_size,
        )

    # Output integer counts for every y-vector, kept compact (P1, total_size) and
    # downloaded with ONE D2H PER TILE (O(num_tiles) D2H total, == 1 when it all fits).
    counts_all = np.empty((P1, total_size), dtype=np.int64)
    d_base_3d = d_base.reshape(1, n, K)  # broadcast base over the y-rows of a tile

    start = 0
    while start < P1:
        stop = min(start + rows_per_tile, P1)
        rows = stop - start
        # (rows, n) y-codes for this tile.
        d_y_tile = d_y_all[start:stop].reshape(rows, n, 1)  # (rows, n, 1) int64
        # per-(row, r, k) global index, then offset each row into its own
        # ``total_size`` slot so a SINGLE bincount fills all rows of the tile at once:
        #   slot = row*total_size + (base[r,k] + y[row,r])
        d_idx = (d_base_3d + d_y_tile).reshape(rows, n * K)  # (rows, n*K) idx_dtype
        # The row-spanning flat index reaches ``rows*total_size``; widen to int64 ONLY when that exceeds
        # int32 (else stay int32, halving the largest buffer too). idx_dtype already covers d_idx itself.
        flat_dtype = cp.int32 if (int(rows) * total_size) <= _INT32_MAX else cp.int64
        d_row_off = (cp.arange(rows, dtype=flat_dtype) * total_size).reshape(rows, 1)
        d_flat = (d_idx.astype(flat_dtype, copy=False) + d_row_off).reshape(-1)  # (rows*n*K,)
        # OPT-D: known-size bincount (skip cupy.bincount's two host-sync validations);
        # byte-identical counts (same kernel). ``rows*total_size`` is the exact size and
        # ``d_flat`` is non-negative by construction (offsets + non-negative codes).
        tile_counts = _cupy_bincount_known_size(d_flat, rows * total_size)
        # ONE D2H for the whole tile.
        # bench-attempt-rejected (2026-06-07): NARROW this D2H int64 -> int32 (the counts are
        # bounded by n << 2^31 so a cast is byte-identical) to halve the transferred bytes (Q2b).
        # Net LOSS: the on-device ``astype(int32)`` kernel launch + sync costs MORE than the saved
        # bytes on the scene tile shapes (0.42-0.99x; only the largest 1200col x 4perm tile wins
        # 1.43x). Crucially the ~16% "asnumpy" the scene sampler attributes here is NOT the
        # transfer -- it is the main thread BLOCKED in asnumpy's implicit sync waiting on the
        # preceding async index-build + bincount GPU kernels (the real, irreducible cost on the
        # PCIe-bound 1050 Ti). Narrowing the transfer cannot shrink GPU compute, and the cast
        # adds more of it. (proto profiling/bench_cupy_d2h_narrow.py)
        counts_all[start:stop, :] = cp.asnumpy(tile_counts).reshape(rows, total_size)
        del d_y_tile, d_idx, d_row_off, d_flat, tile_counts
        start = stop
    del d_base, d_base_3d, d_disc, d_offsets, d_y_all

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
_CUDA_HIST_KERNEL_BATCHED_SHARED: "object | None" = None
_CUDA_HIST_KERNEL_BATCHED_SHARED_CM: "object | None" = None
_CUDA_HIST_KERNEL_BATCHED: "object | None" = None

# Device shuffled-y matrix cache (2026-06-21): y_all = [classes_y; nperm Fisher-Yates shuffles] is
# DETERMINISTIC from (classes_y, classes_y_safe, base_seed, nperm) -- and the target is identical across
# ALL ~30 noise-gate dispatches of a fit, so this (P, n) int32 matrix was rebuilt (njit) AND re-uploaded
# (H2D, ~14MB total) every dispatch. Cache the device matrix by WEAKREF IDENTITY of classes_y + the
# scalar key: built+uploaded once per fit, reused thereafter. Bit-identical (same shuffle stream).
# Per-key device cache (was a single slot, which two concurrent noise-gate dispatches with different targets
# clobbered: dispatch B's d_y overwrote A's, so A read B's shuffled-y matrix). Keyed on
# (id(classes_y), base_seed, nperm, n, P) with a co-validating weakref so an id-recycle onto a different
# target array can never false-hit. Bounded FIFO so distinct targets across a long fit don't grow it.
_DY_DEVICE_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()  # key -> (weakref(classes_y), d_y)
_DY_DEVICE_CACHE_MAX = 8


def _resident_y_all_device(classes_y, classes_y_safe, base_seed, nperm, n, P):
    """Device (P, n) int32 [orig y; nperm shuffles], cached by (id+weakref(classes_y), base_seed, nperm, n, P).
    Built + uploaded once per (target, key); reused across a fit's noise-gate dispatches for the same target."""
    import weakref
    c = _DY_DEVICE_CACHE
    key = (id(classes_y), int(base_seed), int(nperm), int(n), int(P))
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
        y_all[1:, :] = _build_shuffle_matrix(np.asarray(classes_y_safe), np.uint64(base_seed), nperm).astype(np.int32)
    d_y = _nb_cuda.to_device(np.ascontiguousarray(y_all))
    try:
        c[key] = (weakref.ref(classes_y), d_y)
        c.move_to_end(key)
        while len(c) > _DY_DEVICE_CACHE_MAX:
            c.popitem(last=False)
    except TypeError:
        c.pop(key, None)
    return d_y


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
    d_disc_resident=None,
) -> np.ndarray:
    """FULL-GPU-RESIDENT noise gate: batched histogram (all perms, one launch) -> GPU MI kernel -> only
    the (P, K) MI matrix D2H -> host gate decision. The entire permutation noise gate runs on the device
    from resident counts (no per-perm counts D2H, no CPU entropy). Selection-equivalent to
    ``batch_mi_with_noise_gate_cuda`` (same perm gate; GPU MI reproduces the CPU reduction order to fp
    round-off). ``use_su`` has no GPU entropy form here -> delegates to the CPU-entropy cuda path.

    RESIDENT-CODES HANDOFF (2026-06-21, gated): ``d_disc_resident`` -- when the FE chunk binned the codes
    ON the GPU and kept them resident -- is the (n, K) DEVICE codes array (cupy or numba-cuda; consumed via
    the CUDA Array Interface). Passing it lets the histogram kernel read the codes IN PLACE, skipping the
    H2D re-upload of ``disc_2d`` (the codes were produced on the GPU, so re-uploading them is a pointless
    round-trip). It MUST hold the SAME integer codes as ``disc_2d`` (the producer keeps the exact bytes it
    D2H'd) -> counts, MI and the gate decision are BIT-IDENTICAL to the host-codes path. ``None`` (the
    default, and any narrow-dtype mismatch) keeps the H2D-from-host path unchanged."""
    if use_su:
        return batch_mi_with_noise_gate_cuda(
            disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y, npermutations,
            base_seed, min_nonzero_confidence, use_su, dtype,
            128 if threads_per_block is None else threads_per_block,
        )
    global _CUDA_HIST_KERNEL_BATCHED, _CUDA_HIST_KERNEL_BATCHED_SHARED, _CUDA_HIST_KERNEL_BATCHED_SHARED_CM, _CUDA_MI_KERNEL
    if not _CUDA_AVAIL:
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

    # OOB SCREEN (FIX2, 2026-06-28): the batched-hist kernels use raw codes DIRECTLY as a flat shared-mem
    # / counts offset (``cx * K_y + cy`` -- _batch_mi_noise_gate_kernels.py:428/482) sized from
    # nbins_col[k] and K_y. A -1 sentinel or a code >= its cardinality indexes outside the histogram ->
    # cudaErrorIllegalAddress (a hard GPU crash). Screen the HOST codes here (cheap min/max) so an
    # upstream OOB surfaces as a clear ValueError. Skipped when the codes are resident (binner-produced,
    # already on-device): re-syncing them would defeat the resident-handoff and they are dense by contract.
    if d_disc_resident is None:
        _dmin = int(disc_2d.min()); _dmax = int(disc_2d.max())
        _kx_max = int(np.asarray(factors_nbins, dtype=np.int64).max())
        if _dmin < 0 or _dmax >= _kx_max:
            raise ValueError(
                "batch_mi_with_noise_gate_cuda_resident disc_2d codes out of range "
                "(min=%d, max=%d) for nbins max=%d; a -1 sentinel or over-range code would index "
                "outside the device histogram (illegal address)." % (_dmin, _dmax, _kx_max)
            )
        if classes_y.size:
            _cy_min = int(classes_y.min()); _cy_max = int(classes_y.max())
            _ky = int(freqs_y.shape[0])
            if _cy_min < 0 or _cy_max >= _ky:
                raise ValueError(
                    "batch_mi_with_noise_gate_cuda_resident classes_y out of range " "(min=%d, max=%d) for K_y=%d (illegal address)." % (_cy_min, _cy_max, _ky)
                )
    # Lever C (2026-06-23): HW-aware + KTC-tuned threads/block for the batched-hist launch (default 128 left
    # the SM ~1/16 occupied; the row-loop parallelises across threads). When the caller did not pin a count
    # (threads_per_block is None) look the per-host tuned count up from the kernel_tuning_cache (candidate set
    # derived from device occupancy); falls back to 128 on any failure. Block size NEVER changes the integer
    # counts -> MI + gate decision bit-identical. The KTC probe passes an explicit count (skips the lookup).
    if threads_per_block is None:
        try:
            from ._gpu_resident_histgate_ktc import histgate_threads
            threads_per_block = int(histgate_threads(n))
        except Exception:
            threads_per_block = 128
    K_y = int(freqs_y.shape[0])
    nbins_arr = np.asarray(factors_nbins, dtype=np.int64)
    per_col_size = nbins_arr * K_y
    offsets = np.zeros(K + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(per_col_size)
    total_size = int(offsets[K])
    nperm = int(npermutations) if npermutations and npermutations > 0 else 0
    P = nperm + 1

    # d_y (P, n) device shuffle matrix -- cached once per fit (target constant across dispatches).
    d_y = _resident_y_all_device(classes_y, classes_y_safe, base_seed, nperm, n, P)

    # H2D the codes in their NATIVE narrow dtype (int8/int16 -- the bins are < 256/65536) instead of
    # up-casting to int32: nvprof showed the resident gate's wall is dominated by this (n, K) codes H2D,
    # not the (microsecond) kernels, so a 4x-smaller int8 transfer is the real lever. The hist kernel
    # reads disc_2d[r, k] as an index either way (numba compiles a per-dtype variant); counts unchanged.
    # RESIDENT-CODES HANDOFF (gated): when the producer kept the codes on-device, consume that array IN
    # PLACE (CUDA Array Interface) instead of re-uploading -- skips the (n, K) codes H2D entirely. Requires
    # a matching shape + a narrow (itemsize<=2) dtype so the hist kernel reads it as an index exactly like
    # the host path; any mismatch (or None) falls back to the H2D so it can never change the counts.
    _disc_dt = disc_2d.dtype if disc_2d.dtype.itemsize <= 2 else np.int32
    d_disc = None
    if d_disc_resident is not None:
        try:
            _rshape = tuple(int(s) for s in d_disc_resident.shape)
            _rdt = np.dtype(d_disc_resident.dtype)
            if _rshape == (n, K) and _rdt.itemsize <= 2 and _rdt == np.dtype(_disc_dt):
                d_disc = d_disc_resident  # resident device codes -> NO H2D (the round-trip we eliminate)
        except Exception:
            d_disc = None
    if d_disc is None:
        d_disc = _nb_cuda.to_device(np.ascontiguousarray(disc_2d, dtype=_disc_dt))
    d_off = _nb_cuda.to_device(np.ascontiguousarray(offsets[:K], dtype=np.int64))
    d_nb = _nb_cuda.to_device(np.ascontiguousarray(nbins_arr, dtype=np.int32))
    d_freq = _nb_cuda.to_device(np.ascontiguousarray(freqs_y, dtype=np.float64))
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

    # COLUMN-MAJOR coalesced disc (2026-06-23): the shared-mem hist kernel is LOAD-bound (CUDA-event
    # decomposition: per-row shared atomics ~3% of the kernel, ~12 GB/s << ~96 GB/s read peak), throttled
    # by the stride-K ``disc_2d[r, k]`` read (~1/8 coalesced). Reading the SAME values from a (K, n) C-order
    # buffer is fully coalesced -> 12.55x kernel-only (147 GB/s), 5.59x NET with a fresh transpose folded in
    # at n=100k/K=583/P=26 (this corrects the 2026-06-21 "shared-atomic-bound" rejection). The transpose is
    # paid ONCE (all P y-vectors share the one (K, n) disc). Counts are layout-invariant -> BIT-IDENTICAL
    # (verified maxdiff 0). Gated ON; ``MLFRAME_FE_GPU_HISTGATE_CM=0`` forces the row-major kernel; any
    # transpose/compile failure falls back to it too, so CPU / no-CUDA is byte-unchanged.
    _cm_on = os.environ.get("MLFRAME_FE_GPU_HISTGATE_CM", "1").strip().lower() in ("1", "true", "on", "yes")
    _d_disc_cm = None
    if _use_shared and _cm_on and _CUDA_HIST_KERNEL_BATCHED_SHARED_CM is not None:
        try:
            import cupy as _cp_cm
            from ._gpu_resident_select import transpose_codes_to_cm as _transpose_codes_to_cm
            # COALESCED tiled int-codes transpose (2026-06-24): cp.ascontiguousarray(disc.T) was an
            # uncoalesced strided copy -- nsys charged it cupy_copy__int8_int8 46 calls / 1776ms / 29.5% of
            # GPU-kernel time @F2 100k. The tiled kernel reads (n,K) + writes (K,n) coalesced, BIT-IDENTICAL
            # (same values, same (K,n) C-order layout -> counts unchanged); falls back to ascontiguousarray.
            _d_disc_cm = _transpose_codes_to_cm(_cp_cm.asarray(d_disc))  # (K, n) C-order, coalesced disc load
        except Exception:
            logging.getLogger(__name__).debug("histgate column-major transpose failed; row-major fallback", exc_info=True)
            _d_disc_cm = None
    # The shared kernel OVERWRITES every counts slot (flushes its full [off:off+nb_k*K_y] slice), so the
    # host zeroing is redundant there; the global-atomic kernel needs the zeroed buffer.
    if not _use_shared:
        d_counts[:] = 0

    import warnings as _warnings
    try:
        from numba.core.errors import NumbaPerformanceWarning as _NbPerfWarn
    except Exception:
        _NbPerfWarn = None
    with _warnings.catch_warnings():
        if _NbPerfWarn is not None:
            _warnings.simplefilter("ignore", _NbPerfWarn)
        if _use_shared and _d_disc_cm is not None:
            _CUDA_HIST_KERNEL_BATCHED_SHARED_CM[(K, P), threads_per_block, 0, _sh_bytes](  # type: ignore[index]  # numba cuda kernel[grid, block] launch syntax, not real indexing
                _d_disc_cm, d_off, d_nb, d_y, d_counts, n, K_y, total_size,
            )
        elif _use_shared:
            _CUDA_HIST_KERNEL_BATCHED_SHARED[(K, P), threads_per_block, 0, _sh_bytes](  # type: ignore[index]  # numba cuda kernel[grid, block] launch syntax, not real indexing
                d_disc, d_off, d_nb, d_y, d_counts, n, K_y, total_size,
            )
        else:
            _CUDA_HIST_KERNEL_BATCHED[(K, P), threads_per_block](d_disc, d_off, d_y, d_counts, n, K_y, total_size)  # type: ignore[index]  # numba cuda kernel[grid, block] launch syntax, not real indexing
        _tot = K * P
        _blocks = (_tot + threads_per_block - 1) // threads_per_block
        _CUDA_MI_KERNEL[_blocks, threads_per_block](  # type: ignore[index]  # numba cuda kernel[grid, block] launch syntax, not real indexing
            d_counts, d_off, d_nb, d_freq, n, K_y, d_ref, total_size, P, d_out,
        )
    out_mi = d_out.copy_to_host()  # (P, K) -- the only sizeable D2H
    original_mi = out_mi[0]
    if nperm <= 0:
        return _gate_from_mi(original_mi, [], 0, min_nonzero_confidence)
    perm_mis = [out_mi[i] for i in range(1, P)]
    return _gate_from_mi(original_mi, perm_mis, nperm, min_nonzero_confidence)


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
) -> np.ndarray:
    """numba.cuda GPU twin of ``batch_mi_with_noise_gate``. BIT-IDENTICAL.

    GPU work: a global-atomic joint-histogram kernel (one block per column) over
    the original y and each CPU-shuffled y. MI reduced from the integer counts on
    the CPU via ``_mi_from_counts_cpu`` (bit-exact). Raises ``RuntimeError`` if
    numba.cuda is unavailable.
    """
    global _CUDA_HIST_KERNEL
    if not _CUDA_AVAIL:
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

    K_y = int(freqs_y.shape[0])
    nbins_arr = np.asarray(factors_nbins, dtype=np.int64)
    per_col_size = nbins_arr * K_y
    offsets = np.zeros(K + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(per_col_size)
    total_size = int(offsets[K])

    d_disc = _nb_cuda.to_device(np.ascontiguousarray(disc_2d, dtype=np.int32))
    d_off = _nb_cuda.to_device(np.ascontiguousarray(offsets[:K], dtype=np.int64))
    d_nb = _nb_cuda.to_device(np.ascontiguousarray(nbins_arr, dtype=np.int32))

    # One block per column; for tiny K the grid is small (low-occupancy) -- the
    # cupy backend is preferred at the realistic large-batch sizes where GPU wins,
    # so silence the cosmetic NumbaPerformanceWarning here.
    import warnings as _warnings
    try:
        from numba.core.errors import NumbaPerformanceWarning as _NbPerfWarn
    except Exception:
        _NbPerfWarn = None

    def _counts_for(y_codes_host: np.ndarray) -> np.ndarray:
        d_y = _nb_cuda.to_device(np.ascontiguousarray(y_codes_host, dtype=np.int32))
        d_counts = _nb_cuda.device_array(total_size, dtype=np.int64)
        d_counts[:] = 0
        with _warnings.catch_warnings():
            if _NbPerfWarn is not None:
                _warnings.simplefilter("ignore", _NbPerfWarn)
            _CUDA_HIST_KERNEL[K, threads_per_block](d_disc, d_off, d_nb, d_y, d_counts, n, K_y)  # type: ignore[index]  # numba cuda kernel[grid, block] launch syntax, not real indexing
        return np.asarray(d_counts.copy_to_host())

    _col_off = offsets[:K]
    _all_pos = np.ones(K, dtype=np.float64)  # original-y pass: compute every column
    counts_orig = _counts_for(np.asarray(classes_y, dtype=np.int32))
    original_mi = _mi_columns_from_counts_cpu(counts_orig, _col_off, nbins_arr, K_y, freqs_y, n, use_su, _all_pos)

    if npermutations <= 0:
        return _gate_from_mi(original_mi, [], 0, min_nonzero_confidence)

    cy_safe = np.asarray(classes_y_safe)
    # bench-attempt-rejected (2026-06-07): perm-reduction early-exit (see cupy-v2 path) --
    # BYTE-IDENTICAL but no scene wall win; keep full reduction + _gate_from_mi.
    # The per-column MI reduction is batched into one njit call per permutation
    # (_mi_columns_from_counts_cpu) -- skips columns where original_mi<=0 exactly as the
    # old inner loop did; bit-identical, kills the 224k per-call dispatch overhead.
    perm_mis = []
    for i in range(npermutations):
        shuffled = _fisher_yates_shuffle(cy_safe, np.uint64(base_seed), i)
        counts_p = _counts_for(np.asarray(shuffled, dtype=np.int32))
        mp = _mi_columns_from_counts_cpu(counts_p, _col_off, nbins_arr, K_y, freqs_y, n, use_su, original_mi)
        perm_mis.append(mp)

    return _gate_from_mi(original_mi, perm_mis, npermutations, min_nonzero_confidence)


# ---------------------------------------------------------------------------
# Backend chooser + tuner (mirrors batch_pair_mi_gpu.py)
# ---------------------------------------------------------------------------

# Measurement-backed fallback thresholds for the CPU-vs-GPU crossover, keyed on
# n_rows and n_cols (the FE batch shape). The CPU njit-prange kernel wins for
# small/medium batches (tiny joint-hist pass; the H2D copy of the discretized
# frame + per-shuffle bincount launch overhead dominate on GPU). GPU pays off for
# large K at moderate-to-large n. These are the source-code fallback; the live
# dispatch consults the per-host kernel_tuning_cache first (per
# feedback_use_kernel_tuning_cache_for_gpu) so consumer Ampere/Hopper cards learn
# their own crossover instead of inheriting the dev box's threshold.
#
# Dev-box measurement (cupy 13.6.0, cuda_available=True; nbins=10, nperm=3,
# n_classes_y=4), CPU = njit-prange, speedup = CPU/GPU:
#   n=700:  K=64 .16x  K=256 .16x  K=1024 .33-.45x  K=4096 .65-.82x  -> CPU all
#   n=2407: K=64 .26-.43x  K=256 cupy 1.09x  K=1024 ~9-11x  K=4096 ~8-11x -> GPU K>=256
#   n=100000 (LARGE-N, GTX 1050 Ti 4GB): K=256 26.3x  K=1024 32.9x  (bit-identical to
#     the CPU kernel); K=4096 GPU OOMs on the 4GB card (3.3GB alloc) -> CPU fallback.
# So the GPU win starts at n>=~2000 AND K>=256 on this host AND GROWS with n (the O(n*K)
# counting amortises the fixed H2D + launch overhead); below that the launch + H2D
# overhead loses to the tiny CPU joint-hist pass. (Note the CPU njit-prange has a sharp
# slowdown at n=2407,K>=1024 -- per-thread (nbins x K_y) joint buffers x 1024 columns
# thrash cache -- which widens the GPU win there; the cache captures it.)
GPU_MIN_ROWS = 2_000
GPU_MIN_COLS = 256

# LARGE-N ROUTING FIX (2026-06-08). The sweep grid previously topped out at
# n_rows=10000, so EVERY query beyond it (the n=50000/100000/1M FE batches the
# large-n MRMR path actually produces) fell through to the multi-dim grid's
# "catch-all" region -- whose winner is the decision at the LARGEST swept cell
# (n_rows=10000, n_cols=4096). On a card where that corner does not win for GPU
# (e.g. the 4GB 1050 Ti, where K=4096 OOMs -> CPU), the catch-all is CPU and the
# GPU path is DEAD for ALL large n at ALL K -- even though the GPU is measured
# 26-33x FASTER and bit-identical at n=100000, K=256/1024. Extending the grid to
# n_rows=50000/100000 makes the catch-all corner a genuine large-n cell, so on
# capable HW (>=8GB Ampere/Hopper -- e.g. the user's RTX 2070, where the
# (100000, n_cols[-1]) corner fits in memory and GPU wins) the cache routes large
# batches to GPU. On this 4GB box the top-K corner still OOMs -> the corner picks
# CPU (correct locally), but the per-cell large-n bands at GPU-friendly K
# (256/1024) now exist and the fallback heuristic (n>=2000 & K>=256 -> cupy)
# already routes large n to GPU before any sweep lands. The GPU variants gracefully
# skip OOMing sweep cells (sweep_backend_grid try/excepts each cell), so adding
# large-n cells never breaks the sweep on small cards.
_BMING_SWEEP_N_ROWS = [700, 2_407, 10_000, 50_000, 100_000]
_BMING_SWEEP_N_COLS = [64, 256, 1_024, 4_096]
_BMING_SWEEP_NBINS = 10
_BMING_SWEEP_N_CLASSES_Y = 4
_BMING_SWEEP_NPERM = 3
_BMING_SWEEP_MNC = 0.99
_BMING_SALT = 2  # bump on any numerics change / grid change to invalidate stale per-host cache


def _make_batch_mi_noise_gate_inputs(dims: dict):
    """Synthetic (disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y,
    npermutations, base_seed, min_nonzero_confidence, use_su, dtype) tuple at
    ``dims['n_rows']`` rows x ``dims['n_cols']`` candidate columns, matching the
    CPU/GPU kernel call signature."""
    rng = np.random.default_rng(0)
    n = int(dims["n_rows"])
    K = int(dims["n_cols"])
    nbins = _BMING_SWEEP_NBINS
    disc_2d = rng.integers(0, nbins, size=(n, K)).astype(np.int32)
    factors_nbins = np.full(K, nbins, dtype=np.int64)
    classes_y = rng.integers(0, _BMING_SWEEP_N_CLASSES_Y, size=n).astype(np.int32)
    classes_y_safe = classes_y.copy()
    freqs_y = np.bincount(classes_y, minlength=_BMING_SWEEP_N_CLASSES_Y).astype(np.float64) / max(1, n)
    return (
        disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y,
        _BMING_SWEEP_NPERM, np.uint64(0), _BMING_SWEEP_MNC, False, np.int32,
    )


def _run_batch_mi_noise_gate_sweep() -> list:
    """Full (n_rows x n_cols) grid sweep -> backend_choice regions: cpu / cuda /
    cupy, fastest BIT-IDENTICAL variant per cell. The GPU variants are exact (they
    only move the integer counting to the GPU; entropy stays on the bit-exact CPU
    path), so equivalence holds at array_equal -- a tight rtol/atol is used for the
    sweep's equivalence harness."""
    from pyutilz.dev.benchmarking import sweep_backend_grid

    variants = {
        "cpu": lambda *a: _cpu_batch_mi_with_noise_gate(*a),
    }
    if _CUDA_AVAIL:
        variants["cuda"] = lambda *a: batch_mi_with_noise_gate_cuda(*a)
    if _CUPY_AVAIL:
        variants["cupy"] = lambda *a: batch_mi_with_noise_gate_cupy(*a)
    # MEMORY-AWARE grid filter. The CPU reference kernel allocates ~int64 (n, K) intermediates, so the
    # top cell (100k x 4096 ~ 3 GiB) OOMs the HOST on RAM-tight boxes and -- because the per-cell guard
    # does not cover the shared input-gen + reference allocation -- kills the WHOLE sweep, leaving 0
    # regions (so the dispatch is stuck on the fallback heuristic forever). Drop the n_cols that would
    # exceed a fraction of free host RAM at the largest n_row, so the sweep COMPLETES with the runnable
    # cells (partial per-host tuning beats no tuning). The cartesian grid only lets us filter whole
    # columns; that is acceptable -- the dropped large-K-at-large-n cells are exactly the OOM ones, and
    # the fallback (cupy at K>=256) already routes them correctly.
    n_rows = list(_BMING_SWEEP_N_ROWS)
    n_cols = list(_BMING_SWEEP_N_COLS)
    try:
        import psutil
        free = int(psutil.virtual_memory().available)
    except Exception:
        free = 4 * 1024**3
    budget = int(free * 0.4)
    max_n = max(n_rows) if n_rows else 1
    fitting = [k for k in n_cols if max_n * int(k) * 8 * 3 <= budget]
    n_cols = fitting or [min(n_cols)]  # always keep at least the smallest column
    return sweep_backend_grid(  # type: ignore[no-any-return]  # pyutilz helper returns the declared list of results
        variants,
        {"n_rows": n_rows, "n_cols": n_cols},
        _make_batch_mi_noise_gate_inputs,
        reference="cpu",
        repeats=3, equiv_rtol=1e-9, equiv_atol=1e-12,
    )


def _batch_mi_noise_gate_code_version():
    """code_version over the CPU body + GPU bodies + the shared bit-exact reducer;
    re-tunes on any kernel edit."""
    try:
        from pyutilz.performance.kernel_tuning.code_versioning import compute_code_version

        fns = [_cpu_batch_mi_with_noise_gate, _mi_from_counts_cpu, _gate_from_mi]
        if _CUDA_AVAIL:
            fns.append(batch_mi_with_noise_gate_cuda)
        if _CUPY_AVAIL:
            fns.append(batch_mi_with_noise_gate_cupy)
            fns.append(_build_shuffle_matrix)
        return compute_code_version(*fns, salt=_BMING_SALT)
    except Exception:
        return None


def ensure_batch_mi_noise_gate_tuning(force: bool = False):
    """Force-run the ``batch_mi_noise_gate`` CPU-vs-GPU sweep and persist it per-host.

    This is the CLI / ``refresh-all`` entry point that was MISSING -- without it the noise-gate sweep
    only ever fired ASYNC during the first real fit (a multi-minute grid sweep that thrashes the GPU
    mid-MRMR). Pre-running it via the CLI avoids that first-fit cost. Persists with the SAME
    ``code_version`` + ``axes`` the live dispatch's ``get_or_tune`` uses (see
    ``_batch_mi_noise_gate_backend_choice``), so the tuned regions are a HIT for production. Returns the
    region list (``[]``/``None`` if cupy/CUDA absent or the sweep fails -> caller reports a skip)."""
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache
    except Exception:
        return None
    cache = KernelTuningCache.load_or_create()
    if cache is None:
        return None
    if not force:
        regions = cache.get_regions("batch_mi_noise_gate")
        if regions:
            return regions
    import logging
    log = logging.getLogger(__name__)
    log.info("kernel_tuning_cache: batch_mi_noise_gate sweep starting")
    t0 = time.perf_counter()
    try:
        regions = _run_batch_mi_noise_gate_sweep()
    except Exception as e:
        log.warning("kernel_tuning_cache: batch_mi_noise_gate sweep failed: %s", e)
        return None
    log.info("kernel_tuning_cache: batch_mi_noise_gate sweep done in %.2fs", time.perf_counter() - t0)
    if regions:
        try:
            cache.update(
                "batch_mi_noise_gate", axes=["n_rows", "n_cols"], regions=regions,
                code_version=_batch_mi_noise_gate_code_version(),
            )
        except OSError as e:
            log.warning("kernel_tuning_cache: batch_mi_noise_gate save failed: %s", e)
    return regions


def _batch_mi_noise_gate_fallback_choice(n_rows: int, n_cols: int) -> str:
    """Pre-sweep heuristic: GPU only for large n AND large K (where the O(n*K)
    counting amortises the H2D copy + per-shuffle launch overhead); CPU otherwise.
    Prefers cupy over cuda (single batched bincount per shuffle vs per-column
    block launch)."""
    if n_rows >= GPU_MIN_ROWS and n_cols >= GPU_MIN_COLS:
        if _CUPY_AVAIL:
            return "cupy"
        if _CUDA_AVAIL:
            return "cuda"
    return "cpu"


def _batch_mi_noise_gate_backend_choice(n_rows: int, n_cols: int) -> str:
    """Per-host backend (cpu/cuda/cupy) for this (n_rows, n_cols) via the shared
    get_or_tune orchestrator; measurement-backed threshold fallback."""
    try:
        from pyutilz.performance.kernel_tuning.cache import KernelTuningCache

        result = KernelTuningCache.load_or_create().get_or_tune(
            "batch_mi_noise_gate",
            dims={"n_rows": int(n_rows), "n_cols": int(n_cols)},
            tuner=_run_batch_mi_noise_gate_sweep,
            axes=["n_rows", "n_cols"],
            fallback={"backend_choice": _batch_mi_noise_gate_fallback_choice(n_rows, n_cols)},
            code_version=_batch_mi_noise_gate_code_version(),
        )
        bc = result if isinstance(result, str) else str((result or {}).get("backend_choice", ""))
        # Legacy "gpu" region (pre-cupy/cuda split) -> resolve to the available GPU backend.
        if bc == "gpu":
            bc = "cupy" if _CUPY_AVAIL else ("cuda" if _CUDA_AVAIL else "cpu")
        if bc in ("cpu", "cuda", "cupy"):
            return bc
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("batch_mi_noise_gate get_or_tune failed: %s", e)
    return _batch_mi_noise_gate_fallback_choice(n_rows, n_cols)


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
):
    """Run the chosen GPU backend, returning ``(fe_mi, backend_name)`` or ``None``
    when GPU is unavailable / not chosen (so the caller falls back to the CPU
    kernel). Mirrors ``dispatch_batch_pair_mi``'s force_backend + availability
    guards. The CPU kernel is NOT run here -- the caller owns the CPU path.
    """
    n = int(disc_2d.shape[0])
    K = int(disc_2d.shape[1])

    # ABSOLUTE cushion guard (2026-07-05): on a near-full / SHARED card return None so the caller runs the CPU
    # kernel, BEFORE the per-tile relative ``free_b * 0.35`` budget below (computed only after the pool may have
    # already eaten the device). The dominant device buffer is the (rows, n*K) int index array; estimate one
    # y-row's worth as the cushion's bytes_needed. Pure ADD -- tightens, never loosens; permissive without cupy.
    try:
        from ._fe_gpu_vram import fe_gpu_has_vram_cushion
        if not fe_gpu_has_vram_cushion(n * max(K, 1) * 8):
            return None
    except Exception:  # nosec B110 - best-effort/optional path, no module logger
        pass

    if force_backend is not None:
        fb = force_backend.lower()
        if fb == "cupy" and _CUPY_AVAIL:
            return batch_mi_with_noise_gate_cupy(
                disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y,
                npermutations, base_seed, min_nonzero_confidence, use_su, dtype,
            ), "cupy"
        if fb == "cuda" and _CUDA_AVAIL:
            return batch_mi_with_noise_gate_cuda(
                disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y,
                npermutations, base_seed, min_nonzero_confidence, use_su, dtype,
            ), "cuda"
        return None  # forced CPU (or unavailable forced backend) -> caller uses CPU

    choice = _batch_mi_noise_gate_backend_choice(n, K)

    if choice == "cupy" and _CUPY_AVAIL:
        try:
            return batch_mi_with_noise_gate_cupy(
                disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y,
                npermutations, base_seed, min_nonzero_confidence, use_su, dtype,
            ), "cupy"
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logging.getLogger(__name__).debug("suppressed in batch_mi_noise_gate_gpu.py:935: %s", e)
            pass
    if choice == "cuda" and _CUDA_AVAIL:
        try:
            return batch_mi_with_noise_gate_cuda(
                disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y,
                npermutations, base_seed, min_nonzero_confidence, use_su, dtype,
            ), "cuda"
        except (ValueError, RuntimeError):
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
    logging.getLogger(__name__).debug("suppressed in batch_mi_noise_gate_gpu.py:964: %s", e)
    pass


__all__ = [
    "batch_mi_with_noise_gate_cupy",
    "batch_mi_with_noise_gate_cupy_v1",
    "batch_mi_with_noise_gate_cuda",
    "dispatch_batch_mi_with_noise_gate_gpu",
    "_batch_mi_noise_gate_code_version",
    "_batch_mi_noise_gate_backend_choice",
    "_run_batch_mi_noise_gate_sweep",
    "ensure_batch_mi_noise_gate_tuning",
    "_CUDA_AVAIL",
    "_CUPY_AVAIL",
]
