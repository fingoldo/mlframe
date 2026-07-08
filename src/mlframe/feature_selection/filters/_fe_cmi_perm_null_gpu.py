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


def _floor_mean_from_nulls_dev(cp, nulls_dev, quantile: float) -> tuple[float, float]:
    """Reduce a RESIDENT null-CMI vector to ``(floor, null_mean)`` fully on-device, one D2H for the pair.

    Implements the module's documented design (the quantile + mean are reduced ON DEVICE; only the two scalars
    cross back) which the earlier code violated by ``np.quantile`` on a host-copied array. Uses a manual sort +
    linear interpolation quantile (numpy's default method) rather than ``cp.quantile``, which -- like
    ``cp.percentile`` -- does an internal host index read that would resync. ``quantile`` is a host float, so the
    interpolation position/indices are host-computed; only ``nulls_dev[lo]/[hi]`` and the mean are device reads,
    stacked into a single D2H. The null vector itself never leaves the device."""
    m = int(nulls_dev.size)
    if m == 0:
        return 0.0, 0.0
    vs = cp.sort(nulls_dev.ravel())
    pos = float(quantile) * (m - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, m - 1)
    frac = pos - lo
    q_dev = vs[lo] * (1.0 - frac) + vs[hi] * frac  # device 0-dim
    both = cp.asnumpy(cp.stack([q_dev, cp.mean(nulls_dev)]))  # single D2H for (floor, mean)
    return float(both[0]), float(both[1])


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
    # RESIDENT-INPUT fast path (device-born candidate-code foundation): the CMI-redundancy gate device-bins each
    # candidate ONCE (``_quantile_bin_gpu_resident``) and hands the ALREADY-RESIDENT int64 codes here -- use them
    # as-is so the candidate never re-crosses H2D at the ``permnull_cand_x`` site (and ``np.asarray`` on a cupy
    # array would raise, so the host path below must NOT run for a device input). A host candidate takes the
    # content-keyed cache path (uploaded once per fit; the cache hits the copy the per-candidate CMI scorer made).
    dx = x.astype(cp.int64, copy=False).ravel() if isinstance(x, cp.ndarray) else None
    if dx is None:
        from ._fe_resident_operands import resident_code_operand
        dx = resident_code_operand(np.asarray(x).ravel(), "permnull_cand_x")
    y_h = np.ascontiguousarray(y_i, dtype=np.int64).ravel()
    n = int(dx.size)

    if z_i is None:
        # MARGINAL null (seed step): free within-column shuffle -> null MARGINAL MI(x_perm; y). Draw an (n, nperm)
        # key matrix on device and argsort each column -> nperm independent free permutations, all resident.
        # INT32 keys (2026-07-02, nsys-driven): cub radix-sorts int32 in HALF the passes of f64 (the argsort was
        # the dominant per-call cost, micro-bench 305ms at n=1M nperm=25). A 32-bit random key ties within a
        # column with probability ~C(n,2)/2^32; a tie sorts in stable index order -- a microscopically
        # non-uniform but valid permutation (this is a NULL estimate; selection-equivalent, same bar as the
        # device-RNG stream itself).
        keys = rs.randint(0, np.iinfo(np.int32).max, size=(n, nperm), dtype=cp.int32)
        perm = cp.argsort(keys, axis=0)  # (n, nperm) free permutations
        Xp_d = dx[perm]  # (n, nperm) shuffled candidate columns
        nulls_dev = batched_cmi_gpu(Xp_d, y_h, None, return_device=True)  # null CMI vector stays RESIDENT
        return _floor_mean_from_nulls_dev(cp, nulls_dev, quantile)

    # CONDITIONAL null: within-stratum uniform shuffle via the single-key argsort (z_rank + key), SAME as CPU.
    # order / z_rank are per-support derivations (argsort + stratum-rank). Measured DISTINCT across the gate's
    # perm-null calls (the conditioning support grows as features are selected), so they are NOT resident-cached
    # (caching them would only pin per-call VRAM with no reuse); a fresh transient upload, freed below.
    # RESIDENT-SUPPORT: a device-born z_support (the CMI-redundancy gate's ``_renumber_joint_gpu`` of the
    # resident candidate codes) is handed in AS a cupy array -> derive the stratum grouping (``order``) and dense
    # stratum rank ON device so the support codes / order / z_rank never cross H2D (the ``cmi_z`` upload + the
    # ``permnull_cand_x`` order/z_rank uploads). ``cp.argsort(z)`` groups rows by stratum with a DIFFERENT
    # tie-order than the host ``np.argsort(kind='stable')``, but this GPU-resident null is ALREADY selection-
    # equivalent (its shuffle keys are device-RNG, not numpy's), and any valid stratum grouping yields the same
    # conditional null in distribution -> the floor/mean agree within the gate's razor tolerance. Host z_i keeps
    # the passed-in host order/z_rank (byte path unchanged).
    if isinstance(z_i, cp.ndarray):
        z_for_cmi = z_i.astype(cp.int64, copy=False).ravel()
        order_d = cp.argsort(z_for_cmi)  # stratum grouping on device (ties broken by sort)
        _sorted_z = z_for_cmi[order_d]
        z_rank_1d = cp.zeros(n, dtype=cp.float64)
        if n > 1:
            z_rank_1d[1:] = cp.cumsum((_sorted_z[1:] != _sorted_z[:-1]).astype(cp.float64))  # dense block index
        z_rank_d = z_rank_1d[:, None]
        del _sorted_z, z_rank_1d
    else:
        z_for_cmi = np.ascontiguousarray(z_i, dtype=np.int64).ravel()
        order_d = cp.asarray(np.ascontiguousarray(order, dtype=np.int64).ravel())
        z_rank_d = cp.asarray(np.ascontiguousarray(z_rank, dtype=np.float64).ravel())[:, None]  # (n, 1)
    x_sorted_d = dx[order_d]  # x reordered into contiguous stratum blocks

    # WITHIN-STRATUM shuffle keys. INT32 RANK-PACK fast path (2026-07-02, nsys-driven): the argsort of the
    # combined (z_rank + key) matrix was the dominant per-call cost (micro-bench 305ms of ~880ms at n=1M,
    # nperm=25) because cub radix-sorts f64 in 8 byte-passes. Pack the stratum rank into the HIGH bits of an
    # int32 and the random key into the remaining LOW bits -- blocks still never overlap (rank dominates), the
    # low bits give an independent uniform draw per row -- and radix-sort int32 in HALF the passes (~2x). Ties
    # (two rows drawing the same low bits within one stratum, ~2^-rand_bits per pair) sort in stable index
    # order: a microscopically non-uniform but valid permutation -- the same selection-equivalence bar as the
    # device RNG stream itself. Falls back to the exact f64 (rank + uniform) sum when the occupied stratum
    # count leaves fewer than 12 random bits (never at the gate's shapes).
    _nrank = int(z_rank_d[-1, 0]) + 1  # dense ranks are monotone over the sorted rows
    _rb = max(1, int(np.ceil(np.log2(max(2, _nrank)))))
    _rand_bits = 31 - _rb
    if _rand_bits >= 12:
        _zr32 = (z_rank_d[:, 0].astype(cp.int32) << np.int32(_rand_bits))[:, None]
        keys = _zr32 | rs.randint(0, 1 << _rand_bits, size=(n, nperm), dtype=cp.int32)
    else:
        keys = z_rank_d + rs.random_sample(size=(n, nperm))  # exact f64 fallback (huge stratum counts)
    within = cp.argsort(keys, axis=0)  # (n, nperm) within-stratum orders (no overlap)

    # SPARSE sort/run-length CMI (2026-07-02, design-agent + nsys): at the raw-redundancy gate the conditioning
    # support is near-continuous (Kz ~ 1e5 -> Kx*Kyz multi-million), so the dense (chunk, Kx*Kyz) joint of the
    # chunked scorer is multi-GB -- the chunk collapses to ~1 perm and the ~nperm dense passes serialize (the
    # gate's 14.7s). The plug-in CMI needs only OCCUPIED-cell counts, and in the z-SORTED domain (order_d /
    # z_rank already built above) each permuted column's occupied (x,z) / (x,y,z) cells are the RUNS of a
    # composite int64 key -- one batched cp.sort(axis=0) over ALL perms, O(n*nperm) memory, ZERO chunking.
    # Same occupied-cell definition -> same partition counts; fp order differs ~1e-15 (selection-equivalent).
    # Engaged only when the dense joint would exceed the sparse working set (huge-joint predicate); the dense
    # atomic path (faster at small joints) is otherwise unchanged. Opt-out MLFRAME_FE_CMI_PERM_NULL_SPARSE=0.
    if _perm_null_sparse_enabled():
        try:
            dyd = y_h.astype(cp.int64, copy=False).ravel() if isinstance(y_h, cp.ndarray) else None
            if dyd is None:
                from ._fe_resident_operands import resident_operand
                dyd = resident_operand(np.ascontiguousarray(y_h, dtype=np.int64).ravel(), "cmi_y", dtype=np.int64)
            Kx = int(dx.max()) + 1
            Ky = int(dyd.max()) + 1
            _Kz_est = _nrank  # occupied stratum count (dense rank max + 1)
            dense_joint_bytes = Kx * Ky * _Kz_est * 8  # span upper bound of the dense (Kx*Kyz) joint
            sparse_ws_bytes = 4 * n * nperm * 8  # sort + boundary/scan temporaries
            if dense_joint_bytes > max(sparse_ws_bytes, 64 << 20):
                inv_n = 1.0 / float(max(1, n))
                zr = z_rank_d[:, 0].astype(cp.int64)  # (n,) monotone stratum id (z-sorted domain)
                ys = dyd[order_d]  # fixed target in z-sorted order
                xs_p = x_sorted_d[within]  # (n, nperm) permuted candidate, z-sorted order
                key_xz = (zr * np.int64(Kx))[:, None] + xs_p
                h_xz, k_xz = _sparse_batched_entropy_k(cp, key_xz, inv_n)
                del key_xz
                key_xyz = (zr * np.int64(Kx * Ky))[:, None] + xs_p * np.int64(Ky) + ys[:, None]
                h_xyz, k_xyz = _sparse_batched_entropy_k(cp, key_xyz, inv_n)
                del key_xyz, xs_p
                # fixed y/z terms, ONCE per null: zr is already sorted (monotone) and (zr*Ky + ys) sorts cheap.
                h_z, k_z = _sparse_batched_entropy_k(cp, zr[:, None], inv_n)
                h_yz, k_yz = _sparse_batched_entropy_k(cp, (zr * np.int64(Ky) + ys)[:, None], inv_n)
                cmi = cp.maximum((h_xz + float(h_yz[0]) - float(h_z[0]) - h_xyz) - (k_xyz + float(k_z[0]) - k_xz - float(k_yz[0])) * (0.5 * inv_n), 0.0)
                return _floor_mean_from_nulls_dev(cp, cmi, quantile)  # cmi stays resident
        except Exception:
            logger.debug("sparse perm-null CMI failed; dense chunked path", exc_info=True)

    Xp_d = cp.empty((n, nperm), dtype=cp.int64)
    Xp_d[order_d, :] = x_sorted_d[within]  # xp[order] = x_sorted[within], per perm
    del keys, within, x_sorted_d, z_rank_d, order_d  # free the (n,nperm) build + per-call order/zrank
    # (dx is resident-cached -> retained by the cache, not freed here)
    nulls_dev = _batched_cmi_resident_chunked(Xp_d, y_h, z_for_cmi)
    return _floor_mean_from_nulls_dev(cp, nulls_dev, quantile)


def _sparse_batched_entropy_k(cp, keys2d, inv_n: float):
    """Plug-in entropy + occupied-cell count of EVERY column of the (n, nperm) int64 ``keys2d`` via
    sort/run-length -- O(n) memory per column, NO dense (Kx*Kyz) histogram. The occupied cells of a column
    are the RUNS of equal keys after a per-column sort; a run's length is the cell count. Returns
    ``(h[nperm] f64, k[nperm] i64)`` device vectors. Same occupied-cell definition as the dense bincount
    path -> identical partition counts; only the fp summation order differs (~1e-15)."""
    n = int(keys2d.shape[0])
    S = cp.sort(keys2d, axis=0)
    b = cp.empty(S.shape, dtype=cp.bool_)
    b[0, :] = True
    b[1:, :] = S[1:, :] != S[:-1, :]  # run starts
    idx = cp.arange(n, dtype=cp.int64)[:, None]
    rs = cp.maximum.accumulate(cp.where(b, idx, np.int64(0)), axis=0)  # run-start index per element
    is_end = cp.empty(S.shape, dtype=cp.bool_)
    is_end[-1, :] = True
    is_end[:-1, :] = b[1:, :]  # run-end elements
    re = cp.minimum.accumulate(cp.where(is_end, idx, np.int64(n))[::-1, :], axis=0)[::-1, :]
    p = (re - rs + 1).astype(cp.float64) * inv_n  # cell probability, valid at run starts
    h = -(cp.where(b, p * cp.log(p), 0.0)).sum(axis=0)
    k = b.sum(axis=0, dtype=cp.int64)
    return h, k


def _perm_null_sparse_enabled() -> bool:
    """bench-attempt-rejected as DEFAULT (2026-07-02, GTX 1050 Ti): the sparse sort/run-length null (below)
    measured F2 1M/30k wall 49.10s vs 45.82s for the dense chunked path WITH the precomp_yz hoist -- the
    batched (n, nperm) int64 sorts move ~10 passes of 200 MB each, more traffic than the hoisted dense
    joint at this card's bandwidth. Selection-equivalent and OOM-free, so it stays available (opt-in
    MLFRAME_FE_CMI_PERM_NULL_SPARSE=1) for cards/joints where the dense path cannot fit at all."""
    return os.environ.get("MLFRAME_FE_CMI_PERM_NULL_SPARSE", "0").strip().lower() in ("1", "true", "on", "yes")


def _batched_cmi_resident_chunked(Xp_d, y_h: np.ndarray, z):
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
    # permutation does not change x's value set, so Xp_d.max() == the unpermuted candidate's max. ``z`` is the
    # conditioning support: a RESIDENT cupy array (device-born support -> no cmi_z H2D; cardinalities computed on
    # device) or a host ndarray (legacy path). ``batched_cmi_gpu`` accepts either z form directly.
    Kx = (int(Xp_d.max()) + 1) if Xp_d.size else 1
    # Build the column-invariant y/z terms ONCE for the WHOLE null and thread them into every chunk call via
    # batched_cmi_gpu(precomp_yz=...) -- the chunked driver used to re-derive the identical z entropies +
    # yz key (fused 1M-row entropy launches + .max() syncs) INSIDE batched_cmi_gpu on every perm chunk (down
    # to 1 perm/chunk at the gate's huge joints => up to ~nperm recomputes per null). Same values -> the CMI
    # is bit-identical; only the redundant recomputation is gone.
    from ._fe_batched_mi import joint_entropy_gpu
    from ._fe_resident_operands import resident_operand
    if isinstance(y_h, cp.ndarray):
        _dy = y_h.astype(cp.int64, copy=False).ravel()
    else:
        _dy = resident_operand(np.ascontiguousarray(y_h, dtype=np.int64).ravel(), "cmi_y", dtype=np.int64)
    if isinstance(z, cp.ndarray):
        _dz = z.astype(cp.int64, copy=False).ravel()
    else:
        _dz = resident_operand(np.ascontiguousarray(np.asarray(z), dtype=np.int64).ravel(), "cmi_z", dtype=np.int64)
    n_rows = int(Xp_d.shape[0])
    _inv_n = 1.0 / float(max(1, n_rows))
    Kz = (int(_dz.max()) + 1) if _dz.size else 1
    _yzk = _dy * Kz + _dz
    Kyz = (int(_yzk.max()) + 1) if _dz.size else 1
    h_z, k_z = joint_entropy_gpu([_dz], [Kz], _inv_n)
    h_yz, k_yz = joint_entropy_gpu([_yzk], [Kyz], _inv_n)
    _precomp = (_dz, Kz, h_z, k_z, _yzk, Kyz, h_yz, k_yz)
    joint_bytes = max(1, Kx * Kyz * 4)  # int32 counts (2026-07-02) -> the dense joint is half its old size
    try:
        free_b, _ = cp.cuda.runtime.memGetInfo()
    except Exception:
        free_b = 1 << 30
    # 0.30 of free VRAM leaves headroom for batched_cmi_gpu's sort / entropy temporaries beyond the dense joint.
    chunk = max(1, min(nperm, int(free_b * 0.30) // joint_bytes))
    nulls_dev = cp.empty(nperm, dtype=cp.float64)  # assembled RESIDENT -> reduced on-device by caller
    s = 0
    while s < nperm:
        c = min(chunk, nperm - s)
        try:
            cols = cp.ascontiguousarray(Xp_d[:, s : s + c])  # contiguous (n, c) for the joint-hist kernels
            nulls_dev[s : s + c] = batched_cmi_gpu(cols, _dy, z, precomp_yz=_precomp, kx=Kx, return_device=True)
            del cols
            s += c
        except cp.cuda.memory.OutOfMemoryError:
            if c == 1:
                raise  # one perm still won't fit -> caller -> CPU loop
            chunk = max(1, c // 2)  # halve and retry this slice
    return nulls_dev


__all__ = ["conditional_perm_null_gpu", "perm_null_gpu_resident_enabled"]
