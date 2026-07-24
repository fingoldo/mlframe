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
import logging
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

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
    P1 = nperm + 1  # number of y-vectors = nperm + 1

    # RESIDENT shuffle matrix (2026-07-13): row 0 = original y, rows 1.. = the nperm shuffles -- built +
    # uploaded ONCE per (target, seed, nperm) and reused across a fit's cupy dispatches, mirroring the
    # numba.cuda resident gate's ``_resident_y_all_device`` cache (this path previously had NO cache analog
    # and rebuilt + re-H2D'd this (P1, n) matrix on every one of a fit's ~30 dispatches). See
    # ``_resident_y_all_device_for_cupy``.
    d_y_all_resident = _resident_y_all_device_for_cupy(classes_y, classes_y_safe, base_seed, nperm, n, P1)
    d_y_all = d_y_all_resident.astype(idx_dtype)  # (P1, n) -- y codes < K_y, fit idx_dtype (always a fresh cast, never aliases the cached int32 buffer)

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
        logger.info(
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
# GPU_INFRA_A-12 fix (mrmr_audit_2026-07-22): the get -> move_to_end -> popitem LRU sequence below is
# NOT atomic under the GIL's per-bytecode-op granularity; two MRMR.fit() calls racing in different
# threads of the same process (not itself forbidden anywhere in this module) could violate the
# _DY_DEVICE_CACHE_MAX eviction discipline. The returned device array itself was always race-safe
# (every hit re-validates ref() is classes_y before use), so this is a memory-growth guard, not a
# correctness fix.
_DY_DEVICE_CACHE_LOCK = threading.Lock()


def _histgate_upload(host_arr: np.ndarray, role: str, dtype: Any) -> Any:
    """Upload a host operand for the numba.cuda histgate kernels, content-cached via
    ``resident_operand`` when cupy is installed alongside numba.cuda -- a numba.cuda kernel accepts a
    cupy device array directly (CUDA Array Interface; confirmed no host round-trip), so this is a
    drop-in for ``_nb_cuda.to_device`` that additionally dedups fit-constant operands (``freqs_y`` /
    the per-column ``offsets`` / ``nbins``) re-uploaded under a different role elsewhere (the cupy
    backend, other resident-operand FE consumers). Falls back to a plain ``_nb_cuda.to_device`` when
    cupy is absent (a numba.cuda-only host) so that combination is never regressed."""
    if _CUPY_AVAIL:
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
        y_all[1:, :] = _build_shuffle_matrix(np.asarray(classes_y_safe), np.uint64(base_seed), nperm).astype(np.int32)
    d_y = _nb_cuda.to_device(np.ascontiguousarray(y_all))
    with _DY_DEVICE_CACHE_LOCK:
        try:
            c[key] = (weakref.ref(classes_y), d_y)
            c.move_to_end(key)
            while len(c) > _DY_DEVICE_CACHE_MAX:
                c.popitem(last=False)
        except TypeError:
            c.pop(key, None)
    return d_y


# CUPY-NATIVE twin of ``_DY_DEVICE_CACHE`` (2026-07-13), used only when numba.cuda is unavailable on this
# host (cupy-only): mirrors the SAME (id+weakref(classes_y), base_seed, nperm, n, P) key + LRU discipline,
# built via ``cp.asarray`` instead of ``_nb_cuda.to_device``.
_DY_DEVICE_CACHE_CUPY: "OrderedDict[tuple, tuple]" = OrderedDict()  # key -> (weakref(classes_y), d_y)
_DY_DEVICE_CACHE_CUPY_MAX = 8
_DY_DEVICE_CACHE_CUPY_LOCK = threading.Lock()  # GPU_INFRA_A-12 fix -- see _DY_DEVICE_CACHE_LOCK's note


def _resident_y_all_device_cupy(classes_y, classes_y_safe, base_seed, nperm, n, P):
    """CUPY-native fallback for ``_resident_y_all_device_for_cupy`` on a cupy-only host (no numba.cuda):
    same (id+weakref(classes_y), base_seed, nperm, n, P) keying and LRU eviction, built via ``cp.asarray``."""
    import weakref
    cp = _cp
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
        y_all[1:, :] = _build_shuffle_matrix(np.asarray(classes_y_safe), np.uint64(base_seed), nperm).astype(np.int32)
    d_y = cp.asarray(np.ascontiguousarray(y_all))
    with _DY_DEVICE_CACHE_CUPY_LOCK:
        try:
            c[key] = (weakref.ref(classes_y), d_y)
            c.move_to_end(key)
            while len(c) > _DY_DEVICE_CACHE_CUPY_MAX:
                c.popitem(last=False)
        except TypeError:
            c.pop(key, None)
    return d_y


def _resident_y_all_device_for_cupy(classes_y, classes_y_safe, base_seed, nperm, n, P):
    """Return a (P, n) int32 CUPY device array of [orig y; nperm shuffles] for ``batch_mi_with_noise_gate_cupy``.

    When numba.cuda is ALSO available, this reuses ``_resident_y_all_device``'s cache via a ZERO-COPY
    ``cp.asarray`` view (confirmed: cupy wraps a foreign CUDA-Array-Interface buffer by device pointer, no
    copy -- ``cp_view.data.ptr == numba_device_array.__cuda_array_interface__['data'][0]``) instead of a
    second independent upload, so a cuda dispatch followed by a cupy dispatch on the SAME target (or
    vice versa) shares the identical device buffer -- true cross-backend dedup, not just a same-backend
    cache. Falls back to the cupy-native ``_resident_y_all_device_cupy`` when numba.cuda is unavailable
    (a cupy-only host), so that combination still gets the per-fit reuse win."""
    if _CUDA_AVAIL:
        return _cp.asarray(_resident_y_all_device(classes_y, classes_y_safe, base_seed, nperm, n, P))
    return _resident_y_all_device_cupy(classes_y, classes_y_safe, base_seed, nperm, n, P)


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
        # SU (Symmetric Uncertainty) has no on-device entropy form, so this delegates to the CPU-entropy
        # ``batch_mi_with_noise_gate_cuda`` twin -- but that twin's per-permutation loop used to rebuild
        # (CPU Fisher-Yates) AND re-upload (H2D) the shuffled-y codes from scratch on every one of the
        # ~30 noise-gate dispatches of a fit, even though the target/seed/nperm are fit-constant and the
        # SAME (P, n) shuffle matrix is exactly what ``_resident_y_all_device`` already builds+caches for
        # the non-SU resident path. Build/fetch that cached device matrix here and hand it through
        # (``d_y_all_resident``) so the SU delegate slices its rows directly instead of re-shuffling +
        # re-uploading -- BIT-IDENTICAL (same LCG stream; a device-array row slice returns the exact bytes
        # ``_fisher_yates_shuffle`` + ``to_device`` produced, confirmed no host round-trip).
        _n_su = int(disc_2d.shape[0])
        _nperm_su = int(npermutations) if npermutations and npermutations > 0 else 0
        _d_y_all_su = None
        if _CUDA_AVAIL and _n_su > 0:
            try:
                _d_y_all_su = _resident_y_all_device(classes_y, classes_y_safe, base_seed, _nperm_su, _n_su, _nperm_su + 1)
            except Exception:
                _d_y_all_su = None  # best-effort: any failure falls back to the per-call shuffle+upload path
        return batch_mi_with_noise_gate_cuda(
            disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y, npermutations,
            base_seed, min_nonzero_confidence, use_su, dtype,
            128 if threads_per_block is None else threads_per_block,
            d_y_all_resident=_d_y_all_su,
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
        # disc_2d is the discretized FE-CANDIDATE matrix -- a different chunk of engineered columns on
        # (almost) every dispatch, so it is NOT fit-constant like y/freqs_y; content-hashing it here would
        # pay an O(n*K) hash on every call for a near-guaranteed miss. Left as a plain per-call upload
        # (the resident-codes handoff above is the real lever for this operand).
        d_disc = _nb_cuda.to_device(np.ascontiguousarray(disc_2d, dtype=_disc_dt))
    # offsets/nbins/freqs_y ARE fit-constant (the target's class table + the per-column bin-count layout
    # recur across a fit's dispatches) -- route through the content-keyed resident cache instead of a
    # fresh to_device every call (see _histgate_upload).
    d_off = _histgate_upload(offsets[:K], "histgate_off", np.int64)
    d_nb = _histgate_upload(nbins_arr, "histgate_nb", np.int32)
    d_freq = _histgate_upload(freqs_y, "histgate_freq", np.float64)
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
    d_y_all_resident: "Any | None" = None,
) -> np.ndarray:
    """numba.cuda GPU twin of ``batch_mi_with_noise_gate``. BIT-IDENTICAL.

    GPU work: a global-atomic joint-histogram kernel (one block per column) over
    the original y and each CPU-shuffled y. MI reduced from the integer counts on
    the CPU via ``_mi_from_counts_cpu`` (bit-exact). Raises ``RuntimeError`` if
    numba.cuda is unavailable.

    ``d_y_all_resident`` -- an optional cached (P, n) int32 device matrix (row 0 = original ``classes_y``,
    rows 1.. = the ``nperm`` Fisher-Yates shuffles), typically ``_resident_y_all_device``'s return value.
    When given, each per-target pass slices the matching row directly (a device view, no host round-trip)
    INSTEAD OF rebuilding the shuffle on the CPU and re-uploading it -- the SU delegate in
    ``batch_mi_with_noise_gate_cuda_resident`` passes this so the fit-constant shuffle stream is uploaded
    ONCE per fit instead of once per dispatch. Row slicing returns the exact bytes the CPU
    shuffle+``to_device`` path would have produced (same LCG stream), so this is bit-identical; ``None``
    (the default) keeps the original per-call CPU-shuffle + H2D path unchanged.
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

    # disc_2d is the FE-candidate matrix -- varies per dispatch (a different candidate-column chunk each
    # time), so it is NOT fit-constant and is left as a plain per-call upload (see the twin resident
    # function's identical reasoning for its own d_disc fallback).
    d_disc = _nb_cuda.to_device(np.ascontiguousarray(disc_2d, dtype=np.int32))
    # offsets/nbins ARE fit-constant across a fit's dispatches -> content-keyed resident cache.
    d_off = _histgate_upload(offsets[:K], "histgate_off", np.int64)
    d_nb = _histgate_upload(nbins_arr, "histgate_nb", np.int32)

    # One block per column; for tiny K the grid is small (low-occupancy) -- the
    # cupy backend is preferred at the realistic large-batch sizes where GPU wins,
    # so silence the cosmetic NumbaPerformanceWarning here.
    import warnings as _warnings
    try:
        from numba.core.errors import NumbaPerformanceWarning as _NbPerfWarn
    except Exception:
        _NbPerfWarn = None

    def _counts_from_device_y(d_y) -> np.ndarray:
        """Launch the joint-histogram CUDA kernel against an ALREADY-RESIDENT target-codes device array, returning the flat per-column counts array copied back to host."""
        d_counts = _nb_cuda.device_array(total_size, dtype=np.int64)
        d_counts[:] = 0
        with _warnings.catch_warnings():
            if _NbPerfWarn is not None:
                _warnings.simplefilter("ignore", _NbPerfWarn)
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
    # bench-attempt-rejected (2026-06-07): perm-reduction early-exit (see cupy-v2 path) --
    # BYTE-IDENTICAL but no scene wall win; keep full reduction + _gate_from_mi.
    # The per-column MI reduction is batched into one njit call per permutation
    # (_mi_columns_from_counts_cpu) -- skips columns where original_mi<=0 exactly as the
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
):
    """Run the chosen GPU backend, returning ``(fe_mi, backend_name)`` or ``None``
    when GPU is unavailable / not chosen (so the caller falls back to the CPU
    kernel). Mirrors ``dispatch_batch_pair_mi``'s force_backend + availability
    guards. The CPU kernel is NOT run here -- the caller owns the CPU path.
    """
    from ._gpu_policy import gpu_globally_disabled

    if gpu_globally_disabled():
        # GPU_INFRA_A-3 fix (mrmr_audit_2026-07-22): this dispatcher never had its own
        # MLFRAME_DISABLE_GPU/CUDA_VISIBLE_DEVICES self-check, unlike its dispatch_batch_pair_mi and
        # dispatch_friend_graph_stats siblings -- it relied entirely on its caller checking first.
        return None

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
            logger.debug("suppressed in batch_mi_noise_gate_gpu.py:935: %s", e)
            pass
    if choice == "cuda" and _CUDA_AVAIL:
        try:
            return batch_mi_with_noise_gate_cuda(
                disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y,
                npermutations, base_seed, min_nonzero_confidence, use_su, dtype,
            ), "cuda"
        except Exception as e:
            # GPU_INFRA_A-1 fix (mrmr_audit_2026-07-22): numba's CudaAPIError/CudaDriverError derive
            # directly from Exception, not RuntimeError -- broaden to match the already-fixed
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
