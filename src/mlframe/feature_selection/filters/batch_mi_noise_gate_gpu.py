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

import math
from typing import Any

import numpy as np
from numba import njit, prange

# CPU reference kernel (the required win + always-correct fallback).
from .info_theory import batch_mi_with_noise_gate as _cpu_batch_mi_with_noise_gate

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
    d_offsets = cp.asarray(offsets[:K].reshape(1, K))  # (1, K) broadcastable
    d_Ky = np.int64(K_y)

    # Column index of each cell, broadcast (n, K).
    # global_index[r, k] = offsets[k] + disc[r,k]*K_y + y_code[r]
    # We build the per-(r,k) base ``offsets[k] + disc[r,k]*K_y`` once, then add
    # the y-code (which changes per shuffle) before each bincount.
    d_base = d_offsets + d_disc.astype(cp.int64) * d_Ky  # (n, K) int64

    def _joint_counts_for(y_codes_host: np.ndarray) -> np.ndarray:
        """Return a host list of (nbins_k, K_y) int64 count matrices for all K cols
        against ``y_codes_host`` (length n)."""
        d_y = cp.asarray(np.ascontiguousarray(y_codes_host, dtype=np.int64)).reshape(n, 1)  # (n,1)
        d_idx = (d_base + d_y).reshape(-1)  # (n*K,) flat global indices
        # OPT-D: known-size bincount (skip cupy.bincount's host-sync validations);
        # byte-identical counts. ``total_size`` is exact, ``d_idx`` non-negative by construction.
        flat = _cupy_bincount_known_size(d_idx, total_size)
        return cp.asnumpy(flat)  # (total_size,) int64 on host

    # Original MI.
    counts_orig = _joint_counts_for(np.asarray(classes_y, dtype=np.int64))
    original_mi = np.zeros(K, dtype=np.float64)
    for k in range(K):
        nb_k = int(nbins_arr[k])
        block = counts_orig[offsets[k]: offsets[k] + nb_k * K_y].reshape(nb_k, K_y)
        original_mi[k] = _mi_from_counts_cpu(block, nb_k, freqs_y, n, use_su)

    if npermutations <= 0:
        return _gate_from_mi(original_mi, [], 0, min_nonzero_confidence)

    cy_safe = np.asarray(classes_y_safe)
    # bench-attempt-rejected (2026-06-07): perm-reduction early-exit (see cupy-v2 path) --
    # BYTE-IDENTICAL but no scene wall win; keep full reduction + _gate_from_mi.
    perm_mis = []
    for i in range(npermutations):
        shuffled = _fisher_yates_shuffle(cy_safe, np.uint64(base_seed), i)
        counts_p = _joint_counts_for(np.asarray(shuffled, dtype=np.int64))
        mp = np.zeros(K, dtype=np.float64)
        for k in range(K):
            if original_mi[k] <= 0.0:
                continue
            nb_k = int(nbins_arr[k])
            block = counts_p[offsets[k]: offsets[k] + nb_k * K_y].reshape(nb_k, K_y)
            mp[k] = _mi_from_counts_cpu(block, nb_k, freqs_y, n, use_su)
        perm_mis.append(mp)

    return _gate_from_mi(original_mi, perm_mis, npermutations, min_nonzero_confidence)


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

    # ---- ONE H2D: discretized frame + per-(r,k) base index (offsets[k] + disc*K_y).
    d_disc = cp.asarray(np.ascontiguousarray(disc_2d, dtype=np.int32))  # (n, K)
    d_offsets = cp.asarray(offsets[:K].reshape(1, K))                   # (1, K)
    d_base = d_offsets + d_disc.astype(cp.int64) * np.int64(K_y)        # (n, K) int64

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
    d_y_all = cp.asarray(y_all_host)  # (P1, n) int64

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
        d_idx = (d_base_3d + d_y_tile).reshape(rows, n * K)              # (rows, n*K)
        d_row_off = (cp.arange(rows, dtype=cp.int64) * np.int64(total_size)).reshape(rows, 1)
        d_flat = (d_idx + d_row_off).reshape(-1)                         # (rows*n*K,)
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

    # ---- Bit-exact CPU MI reduction from the integer counts (same as v1).
    original_mi = np.zeros(K, dtype=np.float64)
    for k in range(K):
        nb_k = int(nbins_arr[k])
        block = counts_all[0, offsets[k]: offsets[k] + nb_k * K_y].reshape(nb_k, K_y)
        original_mi[k] = _mi_from_counts_cpu(block, nb_k, freqs_y, n, use_su)

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
        cp_row = counts_all[i + 1]
        mp = np.zeros(K, dtype=np.float64)
        for k in range(K):
            if original_mi[k] <= 0.0:
                continue
            nb_k = int(nbins_arr[k])
            block = cp_row[offsets[k]: offsets[k] + nb_k * K_y].reshape(nb_k, K_y)
            mp[k] = _mi_from_counts_cpu(block, nb_k, freqs_y, n, use_su)
        perm_mis.append(mp)

    return _gate_from_mi(original_mi, perm_mis, nperm, min_nonzero_confidence)


# ---------------------------------------------------------------------------
# numba.cuda backend
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
        nb_k = nbins_col[k]
        off = col_offsets[k]
        tid = _nb_cuda.threadIdx.x
        nthreads = _nb_cuda.blockDim.x
        for r in range(tid, n, nthreads):
            cx = disc_2d[r, k]
            cy = y_codes[r]
            # global index: off + cx*K_y + cy
            _nb_cuda.atomic.add(counts_flat, off + cx * K_y + cy, 1)

    return _kernel


_CUDA_HIST_KERNEL: Any = None


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
            _CUDA_HIST_KERNEL[K, threads_per_block](d_disc, d_off, d_nb, d_y, d_counts, n, K_y)
        return d_counts.copy_to_host()

    counts_orig = _counts_for(np.asarray(classes_y, dtype=np.int32))
    original_mi = np.zeros(K, dtype=np.float64)
    for k in range(K):
        nb_k = int(nbins_arr[k])
        block = counts_orig[offsets[k]: offsets[k] + nb_k * K_y].reshape(nb_k, K_y)
        original_mi[k] = _mi_from_counts_cpu(block, nb_k, freqs_y, n, use_su)

    if npermutations <= 0:
        return _gate_from_mi(original_mi, [], 0, min_nonzero_confidence)

    cy_safe = np.asarray(classes_y_safe)
    # bench-attempt-rejected (2026-06-07): perm-reduction early-exit (see cupy-v2 path) --
    # BYTE-IDENTICAL but no scene wall win; keep full reduction + _gate_from_mi.
    perm_mis = []
    for i in range(npermutations):
        shuffled = _fisher_yates_shuffle(cy_safe, np.uint64(base_seed), i)
        counts_p = _counts_for(np.asarray(shuffled, dtype=np.int32))
        mp = np.zeros(K, dtype=np.float64)
        for k in range(K):
            if original_mi[k] <= 0.0:
                continue
            nb_k = int(nbins_arr[k])
            block = counts_p[offsets[k]: offsets[k] + nb_k * K_y].reshape(nb_k, K_y)
            mp[k] = _mi_from_counts_cpu(block, nb_k, freqs_y, n, use_su)
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
    return sweep_backend_grid(
        variants,
        {"n_rows": _BMING_SWEEP_N_ROWS, "n_cols": _BMING_SWEEP_N_COLS},
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
        except Exception:
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
except Exception:
    pass


__all__ = [
    "batch_mi_with_noise_gate_cupy",
    "batch_mi_with_noise_gate_cupy_v1",
    "batch_mi_with_noise_gate_cuda",
    "dispatch_batch_mi_with_noise_gate_gpu",
    "_batch_mi_noise_gate_code_version",
    "_batch_mi_noise_gate_backend_choice",
    "_run_batch_mi_noise_gate_sweep",
    "_CUDA_AVAIL",
    "_CUPY_AVAIL",
]
