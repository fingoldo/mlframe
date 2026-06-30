"""GPU-RESIDENT conditional/marginal permutation null for the CMI redundancy gate (carved 2026-06-28).

The within-stratum permutation null in ``_fe_cmi_redundancy_gate._conditional_perm_null`` is the dominant
un-ported residency gap on the GPU-strict F2 path (measured ~10.1s / 44 calls on F2 STRICT 300k): it runs the
25-shuffle argsort+gather and the per-perm plug-in CMI on CPU numpy whenever the analytic chi-square null does
NOT engage (sparse contingency cells / n below the analytic floor). The analytic fast-path already covers the
dense-cell large-n case; this module ports the EXACT PERMUTATION fallback to cupy so the codes stay resident on
device and only the bounded scalar ``(floor, null_mean)`` returns to the host.

Design (mirrors the CPU path's vectorised within-stratum shuffle exactly):
  * Candidate / target / support CODE arrays are moved to the device ONCE and held resident.
  * The within-stratum uniform permutation is the SAME single-key argsort the CPU path uses: a dense stratum
    rank ``z_rank`` (integer, 0,1,2,... over the sorted support blocks) plus a uniform key in [0,1) keeps every
    row's sort key in the half-open band ``[rank, rank+1)``, so ``argsort`` orders strictly by stratum then by
    random key within each block -- an independent uniform permutation inside every stratum. The keys are drawn
    on the DEVICE (``cupy.random.RandomState``) so there is no per-perm host->device key copy.
  * All ``nperm`` shuffled candidate columns are built into one ``(n, nperm)`` device matrix and scored in ONE
    ``batched_cmi_gpu`` workload (the existing GPU joint-entropy / CMI kernels), reusing the same Miller-Madow
    plug-in CMI the observed value uses.
  * The quantile (floor) and mean (bias estimate) of the permuted CMI distribution are reduced ON DEVICE; only
    the two resulting float64 scalars cross back to the host.

Correctness bar: SELECTION-EQUIVALENCE, not byte-identical. The permutation null is a RANDOM null; a device RNG
seeded from the SAME ``(seed, salt)`` the CPU path uses makes the draw reproducible across runs, and the floor /
null-mean are statistical estimates whose gate decision (CMI vs floor admission, the debiased-excess rel-bar)
matches the CPU path on F2. The device RNG draw sequence is NOT bit-identical to numpy's, so individual null
values differ at the noise scale, but the 0.95 quantile and the mean over 25 draws agree to within the gate's
razor tolerance and the F2 selection is unchanged.

NEVER call ``cp.get_default_memory_pool().free_all_blocks()`` here -- it nukes the resident pool (+47s regression
measured previously). The CPU path remains the default and the rare-GPU-failure fallback (the caller wraps this
in try/except -> CPU, debug-logged).

bench note (2026-06-28, CORRECTED): an earlier A/B reported this port +8s on F2 STRICT 300k and it was wrongly
filed bench-rejected. That A/B ran on a GPU SHARED with a concurrent job -- the walls were contention-inflated
(55-72s where the quiet-box fit is ~35s) and the cProfile cumtime delta (9.89->12.33s) was the async-CUDA
attribution artifact (the tottime actually DROPPED 0.424->0.240s). A synchronized micro-bench at the true perm-
null shapes settles it: over 44 calls the device path beats the CPU loop at every size EVEN with per-call H2D --
n=3000 GPU-host-input 0.20s vs CPU 0.67s; n=8000 0.24s vs 0.92s; n=15000 0.35s vs 1.18s -- and pre-resident is
faster again (0.17/0.21/0.29s). A clean full-fit flag-on vs flag-off A/B is within run-to-run noise (no +8s).
So the port is a WIN, not a regression; it is now DEFAULT ON under STRICT (opt-out MLFRAME_FE_CMI_PERM_NULL_GPU=0).
NEVER call free_all_blocks() here (it nukes the resident pool, +47s measured previously). The CPU path remains
the rare-GPU-failure fallback (the caller wraps this in try/except -> CPU, debug-logged).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def perm_null_gpu_resident_enabled() -> bool:
    """Whether the GPU-RESIDENT permutation null engages. DEFAULT ON under the resident FE path.

    Requires the resident FE path (``fe_gpu_strict_resident_enabled`` -> MLFRAME_FE_GPU_STRICT +
    MLFRAME_FE_GPU_STRICT_RESIDENT); ``MLFRAME_FE_CMI_PERM_NULL_GPU=0`` is the explicit opt-out.

    DEFAULT ON under the resident path; ``MLFRAME_FE_CMI_PERM_NULL_GPU=0`` is the explicit opt-out. The
    original +8s regression that motivated a separate opt-in was a CONTENTION artifact (the wall A/B ran on a
    GPU shared with another job); a synchronized micro-bench at the real perm-null shapes (n=3-15k, nperm=25,
    44 calls) shows the device path is 3-4x faster than the CPU loop EVEN with per-call H2D (GPU-host-input
    0.20-0.35s vs CPU 0.67-1.18s) and faster still pre-resident, and a clean full-fit A/B shows it within
    noise of the host-key path (no +8s). See the corrected module bench note."""
    if os.environ.get("MLFRAME_FE_CMI_PERM_NULL_GPU", "1").strip().lower() not in ("1", "true", "on", "yes"):
        return False
    try:
        from ._gpu_strict_fe import fe_gpu_strict_resident_enabled
        return bool(fe_gpu_strict_resident_enabled())
    except Exception:
        return False


def conditional_perm_null_gpu(
    x: np.ndarray,
    y_i: np.ndarray,
    z_i: Optional[np.ndarray],
    *,
    order: Optional[np.ndarray],
    z_rank: Optional[np.ndarray],
    n_permutations: int,
    quantile: float,
    seed: int,
    salt: int,
) -> tuple[float, float]:
    """GPU-resident within-stratum (or marginal, when ``z_i is None``) permutation null for ``CMI(x; y | z)``.

    Parameters mirror the host loop already prepared in ``_conditional_perm_null``:

    * ``x``        -- dense int candidate codes (n,), the column permuted within strata.
    * ``y_i``      -- dense int target codes (n,), shuffle-invariant.
    * ``z_i``      -- dense int support codes (n,) or ``None`` for the marginal (seed) null.
    * ``order``    -- ``np.argsort(z, kind='stable')`` (the conditional path's stratum grouping); the device
                      shuffle scatters ``x_sorted[within]`` back through this order. Unused on the marginal path.
    * ``z_rank``   -- dense float stratum rank over ``sorted_z`` (0,1,2,...); the single-key argsort base.
    * ``seed``/``salt`` -- folded into the device ``RandomState`` so each candidate draws an INDEPENDENT,
                      REPRODUCIBLE stream (same ``(seed, salt)`` the CPU path mixes via ``SeedSequence``).

    Returns ``(floor, null_mean)`` (host float64 scalars). Raises on ANY cupy error so the caller falls back to
    the exact CPU permutation loop.
    """
    import cupy as cp

    from ._fe_batched_mi import batched_cmi_gpu

    nperm = int(n_permutations)
    if nperm < 1:
        return 0.0, 0.0

    # Device RNG seeded from the SAME (seed, salt) the CPU path mixes -> reproducible per-candidate stream. The
    # XOR-fold keeps the two 32-bit components distinct as a single 64-bit seed (cupy RandomState takes one seed);
    # the exact stream differs from numpy's SeedSequence (acceptable: selection-equivalence, not byte-identity).
    dev_seed = (int(seed) & 0xFFFFFFFF) ^ ((int(salt) & 0xFFFFFFFF) << 1)
    rs = cp.random.RandomState(dev_seed & 0x7FFFFFFFFFFFFFFF)

    # x is held resident on device; y/z stay host (``batched_cmi_gpu`` consumes the (n,nperm) candidate matrix
    # on device but reads y/z as host ndarrays -- it does its own H2D for them, once per call, not per perm).
    # The candidate is the SAME content the per-candidate CMI scorer already uploaded, so the content-keyed
    # resident cache hits that resident copy (no extra H2D); a re-evaluated candidate never re-uploads.
    from ._fe_resident_operands import resident_operand
    dx = resident_operand(np.asarray(x).ravel(), "permnull_cand_x", dtype=np.int64)
    y_h = np.ascontiguousarray(y_i, dtype=np.int64).ravel()
    n = int(dx.size)

    if z_i is None:
        # MARGINAL null (seed step): free within-column shuffle -> null MARGINAL MI(x_perm; y). Draw an (n, nperm)
        # key matrix on device and argsort each column -> nperm independent free permutations, all resident.
        keys = rs.random_sample(size=(n, nperm))                 # device draw, no H2D
        perm = cp.argsort(keys, axis=0)                          # (n, nperm) free permutations
        Xp_d = dx[perm]                                          # (n, nperm) shuffled candidate columns
        nulls = np.asarray(batched_cmi_gpu(Xp_d, y_h, None), dtype=np.float64)
        return float(np.quantile(nulls, quantile)), float(np.mean(nulls))

    # CONDITIONAL null: within-stratum uniform shuffle via the single-key argsort (z_rank + key), SAME as CPU.
    # order / z_rank are per-support derivations (argsort + stratum-rank). Measured DISTINCT across the gate's
    # perm-null calls (the conditioning support grows as features are selected), so they are NOT resident-cached
    # (caching them would only pin per-call VRAM with no reuse); a fresh transient upload, freed below.
    z_h = np.ascontiguousarray(z_i, dtype=np.int64).ravel()
    order_d = cp.asarray(np.ascontiguousarray(order, dtype=np.int64).ravel())
    z_rank_d = cp.asarray(np.ascontiguousarray(z_rank, dtype=np.float64).ravel())[:, None]   # (n, 1)
    x_sorted_d = dx[order_d]                                     # x reordered into contiguous stratum blocks

    keys = rs.random_sample(size=(n, nperm))                     # device draw -> keys in [0,1), distinct per row
    within = cp.argsort(z_rank_d + keys, axis=0)                 # (n, nperm) within-stratum orders (no overlap)
    Xp_d = cp.empty((n, nperm), dtype=cp.int64)
    Xp_d[order_d, :] = x_sorted_d[within]                        # xp[order] = x_sorted[within], per perm
    del keys, within, x_sorted_d, z_rank_d, order_d            # free the (n,nperm) build + per-call order/zrank
    # (dx is resident-cached -> retained by the cache, not freed here)
    nulls = _batched_cmi_resident_chunked(Xp_d, y_h, z_h)
    return float(np.quantile(nulls, quantile)), float(np.mean(nulls))


def _batched_cmi_resident_chunked(Xp_d, y_h: np.ndarray, z_h: np.ndarray) -> np.ndarray:
    """CMI(perm_col; y | z) for every column of the RESIDENT (n, nperm) permutation matrix ``Xp_d``, chunking
    the perms so each ``batched_cmi_gpu`` call's dense joint fits free VRAM.

    ``batched_cmi_gpu``'s conditional path densifies a ``(nperm, Kx*Kyz)`` joint-count histogram (Kyz = the
    occupied (y,z) cells). At the FULL-n redundancy-gate calls the conditioning support is near-continuous
    (Kz ~ 1e5 -> Kyz multi-million), so the whole-batch joint is multi-GB (measured 7-11 GB at nperm=25) and
    OOMs a small card -- which is exactly why the resident null used to raise OOM and the gate fell back to the
    800 MB host-key path + the per-perm CPU loop (the residency leak). Each perm's CMI is INDEPENDENT of how
    many perms share a batched call, so splitting the perms into VRAM-sized chunks yields the SAME null values
    while keeping the null fully resident (device RNG, no per-perm candidate or key H2D). Adaptive: on OOM the
    chunk halves and retries, down to a single perm (only then re-raising, so the caller's CPU fallback still
    covers a genuinely-too-large single joint)."""
    import cupy as cp

    from ._fe_batched_mi import batched_cmi_gpu

    nperm = int(Xp_d.shape[1])
    # dense joint width per perm = Kx * Kyz; size the chunk from free VRAM (host-cheap cardinalities). The
    # permutation does not change x's value set, so Xp_d.max() == the unpermuted candidate's max.
    Kx = (int(Xp_d.max()) + 1) if Xp_d.size else 1
    Kz = (int(z_h.max()) + 1) if z_h.size else 1
    Kyz = (int((y_h * Kz + z_h).max()) + 1) if z_h.size else 1
    joint_bytes = max(1, Kx * Kyz * 8)
    try:
        free_b, _ = cp.cuda.runtime.memGetInfo()
    except Exception:
        free_b = 1 << 30
    # 0.30 of free VRAM leaves headroom for batched_cmi_gpu's sort / entropy temporaries beyond the dense joint.
    chunk = max(1, min(nperm, int(free_b * 0.30) // joint_bytes))
    nulls = np.empty(nperm, dtype=np.float64)
    s = 0
    while s < nperm:
        c = min(chunk, nperm - s)
        try:
            cols = cp.ascontiguousarray(Xp_d[:, s:s + c])       # contiguous (n, c) for the joint-hist kernels
            nulls[s:s + c] = np.asarray(batched_cmi_gpu(cols, y_h, z_h), dtype=np.float64)
            del cols
            s += c
        except cp.cuda.memory.OutOfMemoryError:
            if c == 1:
                raise                                            # one perm still won't fit -> caller -> CPU loop
            chunk = max(1, c // 2)                               # halve and retry this slice
    return nulls


__all__ = ["conditional_perm_null_gpu", "perm_null_gpu_resident_enabled"]
