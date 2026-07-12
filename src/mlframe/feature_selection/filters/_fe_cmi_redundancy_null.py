"""Conditional-permutation / analytic chi-square CMI null for the FE redundancy gate.

Carved from _fe_cmi_redundancy_gate.py: the significance-floor + debias null computation
(``_conditional_perm_null``) plus the analytic-null size threshold. The gate facade re-imports
every name; all heavy MI/GPU primitives are lazy-imported inside the function so import stays cheap."""
from __future__ import annotations

import logging
import os as _os
from typing import Optional

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Conditional-permutation floor: within-stratum shuffles + the null-quantile used as the significance bar.
_CMI_FLOOR_PERMUTATIONS = 25
_CMI_FLOOR_QUANTILE = 0.95

# Minimum n at which the analytic chi-square CMI null replaces the within-stratum permutation null.
# Env-tunable via MLFRAME_CMI_ANALYTIC_NULL_MIN_N.
_CMI_ANALYTIC_NULL_MIN_N_DEFAULT = 20_000


def _cmi_analytic_null_min_n() -> int:
    """Minimum ``n`` at which the analytic chi-square CMI null replaces the within-stratum permutation null, from ``MLFRAME_CMI_ANALYTIC_NULL_MIN_N`` or the module default."""
    raw = _os.environ.get("MLFRAME_CMI_ANALYTIC_NULL_MIN_N", "").strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return _CMI_ANALYTIC_NULL_MIN_N_DEFAULT


def _conditional_perm_null(
    cand_bin: np.ndarray,
    y_bin: np.ndarray,
    z_support: Optional[np.ndarray],
    *,
    n_permutations: int = _CMI_FLOOR_PERMUTATIONS,
    quantile: float = _CMI_FLOOR_QUANTILE,
    seed: int = 0,
    salt: int = 0,
    precomp_cards: Optional[tuple] = None,
    z_support_dev=None,
) -> tuple[float, float]:
    """Conditional-permutation null for ``CMI(cand; y | z_support)``.

    ``precomp_cards`` (k_z, k_xz, k_yz, k_xyz): the OCCUPIED-cell cardinalities for THIS candidate, already
    computed by the round-level ``batched_cmi_gpu(return_cards=True)`` workload. When supplied (conditional
    path), the analytic-null df reuses them instead of recomputing the per-candidate ``joint_cardinalities_cupy``
    (the same occupied-cell definition -> bit-identical df), eliminating a redundant 4-histogram device call
    per fallback candidate.

    ``salt`` mixes a per-candidate value into the RNG (via ``SeedSequence([seed, salt])``) so each
    candidate's null is drawn from an INDEPENDENT stream. Without it every candidate in a greedy round
    reused the identical permutation key sequence, correlating their floors/null-means and making the
    per-candidate significance test (leg 1) not independent. Callers pass a stable per-candidate salt.

    Reuses the production within-stratum permutation infrastructure
    (``conditional_permutation_test``): the candidate column is permuted WITHIN
    each support stratum, preserving the ``cand | support`` distribution, so the
    null measures the CMI a candidate of the SAME conditional marginal would show
    by chance.

    Returns ``(floor, null_mean)``:

      * ``floor``  -- the ``quantile`` of the null distribution; the
                         data-derived SIGNIFICANCE bar (leg 1).
      * ``null_mean`` -- the MEAN of the null distribution; an estimate of the
                         candidate's finite-sample CMI BIAS at this n. Subtracted
                         from the observed CMI to form the n-invariant DEBIASED
                         EXCESS that the relative-gap leg (leg 2) compares (the
                         bias is present in both ``cmi_obs`` and ``null_mean`` and
                         cancels). The within-stratum shuffle reproduces the SAME
                         bias because it leaves ``cand | support`` (hence the
                         occupied-cell structure that drives the bias) intact.

    When ``z_support`` is ``None`` (the seed step, nothing to condition on) the
    null is the MARGINAL-permutation null: ``cand`` is freely shuffled and the
    null statistic is the marginal ``MI(cand_perm; y)``. This debiases the seed
    anchor on the SAME footing as the conditional candidates, so every admitted
    feature's relative-bar anchor is a debiased excess.
    """
    # Sparse, renumber-based plug-in CMI/MI -- the SAME estimator used for the
    # observed value (``_cmi_from_binned``), so the floor / null mean and the
    # point estimate are directly comparable, and the memory stays bounded by n
    # (no dense (K_x, K_y, K_z) contingency allocation when the frozen support's
    # joint cardinality climbs into the thousands).
    from ._mi_greedy_cmi_fe import _cmi_gpu_enabled, _entropy_from_classes, _renumber_joint, cmi_from_binned_fixed_yz, precompute_cmi_yz_terms, precompute_marginal_y_terms, marginal_mi_binned_fixed_y

    # ``cand_bin`` may be an ALREADY-RESIDENT cupy int64 code (device-born binning) OR a host int64 array. Keep
    # the resident handle ``cand_dev`` so the GPU-resident perm-null branch below consumes it WITHOUT a re-upload
    # (``conditional_perm_null_gpu`` has an ``isinstance(x, cp.ndarray)`` resident-input branch); the host ``x``
    # is derived LAZILY (one D2H) only when a genuine host path is reached (the analytic marginal-seed renumber,
    # or the batched / CPU permutation fallbacks). On the dominant large-n analytic conditional path with
    # ``precomp_cards`` supplied, ``x`` is never materialised at all.
    cand_dev = None
    try:
        import cupy as _cp_c
        if isinstance(cand_bin, _cp_c.ndarray):
            cand_dev = cand_bin.astype(_cp_c.int64, copy=False).ravel()
    except Exception:
        cand_dev = None

    _host_x_cache: list = [None]

    def _host_x():
        """Return the candidate codes as a host int64 array, D2H-transferring once and caching the result on the closure."""
        # D2H the resident code once (cached on the closure via the list cell) or contiguous-cast a host input.
        if _host_x_cache[0] is None:
            if cand_dev is not None:
                import cupy as _cp2
                _host_x_cache[0] = np.ascontiguousarray(_cp2.asnumpy(cand_dev), dtype=np.int64).ravel()
            else:
                _host_x_cache[0] = np.ascontiguousarray(cand_bin, dtype=np.int64).ravel()
        return _host_x_cache[0]

    y = np.ascontiguousarray(y_bin, dtype=np.int64).ravel()

    # ANALYTIC CMI NULL (2026-06-20). The 25 within-stratum permutations exist only to estimate
    # the null distribution of the plug-in CMI under conditional independence X _||_ Y | Z. That
    # null has a known asymptotic form: the likelihood-ratio (G) statistic ``2N * CMI`` is
    # chi-square with df = ``sum_z (Bx_z - 1)(By_z - 1)``, summed over the conditioning strata.
    # Using OCCUPIED-cell counts this df equals exactly ``k_xyz + k_z - k_xz - k_yz`` -- the SAME
    # quantity the Miller-Madow bias term in ``_cmi_from_binned`` already computes (the plug-in CMI
    # bias is ``-df/(2N)``). So the null is distributed as ``chi2(df)/(2N)``:
    #   * null_mean = E[chi2(df)/(2N)] = df/(2N)  -- IDENTICAL to what the permutation mean estimates
    #     (the candidate's finite-sample bias), so the debiased excess ``cmi - null_mean`` (leg 2) is
    #     selection-EQUIVALENT to the permutation path by construction (matched bias estimator).
    #   * floor    = chi2.ppf(quantile, df)/(2N)  -- the analytic ``quantile`` of the same null, the
    #     significance bar (leg 1), replacing the empirical 95th percentile of 25 draws.
    # Bias-consistent with the observed CMI's Miller-Madow correction and free of the 25-permutation
    # cost (the single largest steady-state redundancy-gate hotspot at large n). Gated on the same
    # safe-conditions as the pair-path analytic MI null: n >= the analytic-null threshold AND the
    # contingency cells are not sparse (avg expected count >= the min-cell floor). Below either, or
    # when scipy.chi2 is unavailable, falls back to the exact permutation null. Env off-switch:
    # MLFRAME_MI_ANALYTIC_NULL=0 routes everything to the permutation path.
    try:
        from ._analytic_mi_null import _HAVE_CHI2, _chi2, _min_expected_cell, analytic_null_enabled
    except Exception:
        _HAVE_CHI2 = False
    n_size = int(cand_dev.size) if cand_dev is not None else int(np.asarray(cand_bin).size)
    if _HAVE_CHI2 and analytic_null_enabled() and n_size >= _cmi_analytic_null_min_n():
        try:
            _n = float(max(1, n_size))
            if (z_support is None or z_support.size == 0) and z_support_dev is None:
                _xh = _host_x()
                _, k_x = _entropy_from_classes(_xh)
                _, k_y = _entropy_from_classes(y)
                _df = (int(k_x) - 1) * (int(k_y) - 1)
                _cells = max(1, int(k_x) * int(k_y))
            else:
                # z_support may be host OR None with a resident z_support_dev (device-born support). Keep _z host
                # only if a host consumer is actually reached (the no-precomp-cards renumber fallback below);
                # the precomp_cards + resident joint_cardinalities paths need no host z (no D2H).
                _z = np.ascontiguousarray(z_support, dtype=np.int64).ravel() if z_support is not None else None
                # GPU route (2026-06-25): the analytic-null df needs only the OCCUPIED-cell counts
                # (k_z/k_xz/k_yz/k_xyz) -> device cp.unique(...).size replaces the host renumber+entropy.
                # Label-invariant -> same df. Gated (STRICT / MLFRAME_CMI_GPU), falls back to CPU on error.
                _ks = None
                if precomp_cards is not None:
                    # round-batched cards (same occupied-cell definition -> bit-identical df); no per-cand call
                    _ks = precomp_cards
                elif _cmi_gpu_enabled():
                    try:
                        from ._mi_greedy_cmi_fe import joint_cardinalities_cupy
                        # RESIDENT candidate code + RESIDENT support (joint_cardinalities_cupy resident-input
                        # branch) -> no re-upload at the ``card_cand_x`` / ``cmi_z`` sites; host code otherwise.
                        _ks = joint_cardinalities_cupy(cand_dev if cand_dev is not None else _host_x(), y, _z if _z is not None else z_support_dev)
                    except Exception:
                        _ks = None
                if _ks is not None:
                    k_z, k_xz, k_yz, k_xyz = _ks
                else:
                    # Host renumber fallback (no precomp_cards + no resident cards): materialise the support on
                    # host from the device-born copy (rare -- a single D2H only when both device card paths miss).
                    if _z is None and z_support_dev is not None:
                        import cupy as _cp
                        _z = np.ascontiguousarray(_cp.asnumpy(z_support_dev), dtype=np.int64).ravel()
                    assert _z is not None  # the enclosing else-branch condition guarantees z_support or z_support_dev is present
                    _xh = _host_x()
                    _, k_xz = _renumber_joint(_xh, _z)
                    _, k_yz = _renumber_joint(y, _z)
                    _, k_xyz = _renumber_joint(_xh, y, _z)
                    _, k_z = _entropy_from_classes(_z)
                # df = sum_z (Bx_z - 1)(By_z - 1) over OCCUPIED strata = k_xyz - k_xz - k_yz + k_z
                # (occupied-cell expansion). This is EXACTLY the Miller-Madow CMI bias numerator
                # ``_cmi_from_binned`` uses (``cmi_bias = (k_xyz + k_z - k_xz - k_yz)/(2n)``), so
                # ``null_mean = df/(2N)`` matches the plug-in CMI's bias term sign-for-sign and is
                # always >= 0 for nested supports. (Prior form ``k_xz+k_yz-k_z-k_xyz`` was the NEGATED
                # quantity -> df<0 for every sparse high-cardinality joint, so the >0 guard below sent
                # ALL conditional calls to the permutation null and the analytic path never engaged.)
                _df = int(k_xyz) + int(k_z) - int(k_xz) - int(k_yz)
                _cells = max(1, int(k_xyz))
            # Sparse-cell safe-condition (chi-square "expected >= 5" rule): avg expected count over
            # the joint cells must clear the floor, else the asymptotic is unreliable -> permute.
            if _df > 0 and (_n / float(_cells)) >= _min_expected_cell():
                _null_mean = _df / (2.0 * _n)
                _floor = float(_chi2.ppf(float(quantile), _df)) / (2.0 * _n)
                if _floor < 0.0:
                    _floor = 0.0
                return float(_floor), float(_null_mean)
            # else: fall through to the permutation null (sparse cells / degenerate df).
        except Exception:
            logger.debug("analytic CMI null failed; using permutation null", exc_info=True)

    rng = np.random.default_rng(np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(salt) & 0xFFFFFFFF]))

    # RESIDENT-SUPPORT conditional perm-null: a device-born z_support is available -> run the GPU-resident
    # conditional null with order/z_rank DERIVED ON DEVICE from the resident z (conditional_perm_null_gpu), so
    # the support / order / z_rank / candidate never cross H2D. Only on a cupy fault does control fall through to
    # the host order/z_rank path below (which materialises z from the device copy). Selection-equivalent (this
    # GPU null already uses a device-RNG shuffle; the device stratum grouping is another valid grouping).
    if z_support_dev is not None and getattr(z_support_dev, "size", 0) > 0 and _cmi_gpu_enabled() and int(n_permutations) > 1:
        try:
            from ._fe_cmi_perm_null_gpu import perm_null_gpu_resident_enabled, conditional_perm_null_gpu
            if perm_null_gpu_resident_enabled():
                return conditional_perm_null_gpu(
                    cand_dev if cand_dev is not None else _host_x(), y, z_support_dev,
                    order=None, z_rank=None,
                    n_permutations=int(n_permutations), quantile=quantile, seed=seed, salt=salt,
                )
        except Exception:
            logger.debug("resident-support GPU conditional perm-null failed; host order/z_rank path", exc_info=True)
        # GPU path unavailable/failed -> materialise host z once for the host order/z_rank fallback below.
        if z_support is None and z_support_dev is not None:
            import cupy as _cp
            z_support = np.ascontiguousarray(_cp.asnumpy(z_support_dev), dtype=np.int64).ravel()

    if (z_support is None or z_support.size == 0) and z_support_dev is None:
        # Marginal-permutation null (seed step): free shuffle of the candidate
        # -> null MARGINAL MI(cand; y). Mean estimates the marginal MI bias at
        # this n; the seed's debiased excess = max(0, marginal_mi - this mean).
        nperm = int(n_permutations)
        # GPU-RESIDENT marginal null (default OFF -> MLFRAME_FE_CMI_PERM_NULL_GPU, requires the RESIDENT path).
        # Draws the shuffle KEYS on device (cupy RandomState, no per-perm key H2D) and reduces the floor/mean on
        # device; only the two scalars return. Selection-EQUIVALENT (device RNG stream != numpy's; same
        # (seed,salt)). BENCH-REJECTED on the 6-SM GTX 1050 Ti (regressed wall; see _fe_cmi_perm_null_gpu module
        # docstring) -> behind its own sub-flag so plain RESIDENT keeps the faster host-key path below. Falls back
        # to the host-key batched path then the CPU loop on any cupy error. Plain STRICT stays byte-identical.
        try:
            from ._fe_cmi_perm_null_gpu import perm_null_gpu_resident_enabled
            _resident = perm_null_gpu_resident_enabled()
        except Exception:
            _resident = False
        if _resident and _cmi_gpu_enabled() and nperm > 1:
            try:
                from ._fe_cmi_perm_null_gpu import conditional_perm_null_gpu
                # Pass the RESIDENT candidate code directly (conditional_perm_null_gpu resident-input branch) so
                # the marginal seed null never re-uploads the candidate; host code otherwise.
                return conditional_perm_null_gpu(
                    cand_dev if cand_dev is not None else _host_x(), y, None, order=None, z_rank=None,
                    n_permutations=nperm, quantile=quantile, seed=seed, salt=salt,
                )
            except Exception:
                logger.debug("GPU-resident marginal perm-null failed; using host/CPU path", exc_info=True)
        # BATCHED marginal null under STRICT (default OFF -> CPU loop): all nperm free-shuffled columns
        # into one (n, nperm) matrix (SAME rng draws) -> one batched_cmi_gpu(..., z=None) call.
        _xh = _host_x()
        if _cmi_gpu_enabled() and nperm > 1:
            try:
                import cupy as cp
                from ._fe_batched_mi import batched_cmi_gpu
                from ._fe_cmi_perm_null_gpu import _floor_mean_from_nulls_dev
                Xp = np.empty((_xh.size, nperm), dtype=np.int64)
                for i in range(nperm):
                    Xp[:, i] = _xh[rng.permutation(_xh.size)]
                # null CMI vector reduced on-device -> stays resident, one D2H for (floor, mean)
                nulls_dev = batched_cmi_gpu(Xp, y, None, return_device=True)
                return _floor_mean_from_nulls_dev(cp, nulls_dev, quantile)
            except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                logger.debug("suppressed in _fe_cmi_redundancy_null.py:276: %s", e)
                pass
        # y is FIXED across all marginal shuffles, so H(Y) and its occupied-cell count k_y are
        # invariant -- hoist the y-only block ONCE (precompute_marginal_y_terms) and reuse it per
        # perm via marginal_mi_binned_fixed_y, instead of re-binning y + recomputing H(Y) inside
        # every _cmi_from_binned(x_perm, y, None) call. Bit-identical (same plug-in entropies +
        # Miller-Madow bias, only the redundant per-perm H(Y) recompute + y int64 cast removed).
        y_i_m, h_y_m, k_y_m = precompute_marginal_y_terms(y)
        nulls = np.empty(nperm, dtype=np.float64)
        for i in range(nperm):
            x_perm = _xh[rng.permutation(_xh.size)]
            nulls[i] = float(marginal_mi_binned_fixed_y(x_perm, y_i_m, h_y_m, k_y_m))
        return float(np.quantile(nulls, quantile)), float(np.mean(nulls))

    z = np.ascontiguousarray(z_support, dtype=np.int64).ravel()
    # Group row indices by support stratum once; permute the CANDIDATE column
    # within each stratum (preserves the ``cand | support`` distribution -- the
    # conditional permutation null of Berrett et al. 2020).
    order = np.argsort(z, kind="stable")
    sorted_z = z[order]
    boundaries = np.flatnonzero(np.diff(sorted_z)) + 1
    groups = [g for g in np.split(order, boundaries) if g.size > 1]
    if not groups:
        return 0.0, 0.0
    # y and z are fixed across permutations (only x is reshuffled within strata),
    # so the H(Y,Z) / H(Z) block of the conditional CMI is invariant -- hoist it
    # out of the loop and recompute only the x-dependent xz / xyz terms per perm.
    y_i, z_i, h_yz, h_z, k_yz, k_z, n_f = precompute_cmi_yz_terms(y, z)
    # VECTORISED within-stratum permutation (perf, 2026-06-19). The previous per-stratum Python loop
    # ``for g in groups: x_perm[g] = x[g[rng.permutation(g.size)]]`` issued ONE rng.permutation PER
    # stratum PER perm; at n=100k with a high-cardinality conditioning support that is hundreds of
    # thousands of calls -- measured 697k calls / 14.4s, the single largest steady-state FE hotspot.
    # A uniform within-group shuffle is a single lexsort: draw one random key per row and sort by
    # (stratum, key); since ``sorted_z`` is already sorted, each contiguous stratum block is reordered
    # by its keys alone -> an independent uniform permutation within every stratum in one vectorised
    # pass (size-1 strata are fixed points, exactly as the old ``g.size > 1`` guard left them). This is
    # the SAME conditional permutation null (Berrett et al. 2020) -- only the RNG draw sequence changes,
    # so the floor/mean are unchanged in expectation (verified: identical feature selection on the
    # canonical recovery fit). ``order``/``sorted_z`` were computed once above.
    # ``x_sorted`` (x reordered into contiguous stratum blocks) is only consumed by the HOST fallbacks below
    # (the batched-device build and the per-perm CPU loop); the DEFAULT-ON GPU-resident branch does the reorder
    # on device from the resident ``cand_dev``. Defer it (via ``_host_x()``) so the resident path never D2Hs the
    # candidate just to build a host reorder it will not use.
    # SINGLE-KEY argsort within-stratum shuffle (perf, 2026-06-21). The per-perm
    # ``np.lexsort((keys, sorted_z))`` runs TWO stable radix passes (one per key) over all n
    # rows every permutation -- and at a high-cardinality conditioning support (thousands of
    # strata) it was the DOMINANT cost of the conditional null, above the CMI eval itself
    # (measured: 25-perm loop 224ms -> 75ms, 3.0x, at n=30k / 1500 strata). Since ``sorted_z``
    # is already grouped, replace it with a DENSE STRATUM RANK (0,1,2,... over the sorted
    # blocks) and a SINGLE ``argsort(z_rank + keys)``: ``keys`` lie in [0,1) and ``z_rank`` is
    # integer, so ``z_rank + keys`` stays in the half-open band ``[rank, rank+1)`` per stratum
    # -- blocks never overlap, so the argsort orders strictly by stratum then by key within
    # each block. This is the SAME within-stratum uniform permutation lexsort produced: for an
    # identical ``keys`` draw the resulting order is bit-identical (verified: ``sorted_z`` after
    # both reorderings is equal element-for-element), so the RNG draw sequence and every null
    # value are unchanged -- selection is bit-identical, not merely equivalent.
    z_rank = np.zeros(n_size, dtype=np.float64)
    if n_size > 1:
        z_rank[1:] = np.cumsum(sorted_z[1:] != sorted_z[:-1])
    _nperm = int(n_permutations)
    # GPU-RESIDENT conditional null (DEFAULT ON under the RESIDENT path -> opt-out MLFRAME_FE_CMI_PERM_NULL_GPU=0).
    # Holds the candidate / target / support codes resident on device, draws the within-stratum shuffle KEYS on
    # device (cupy RandomState -- no per-perm key H2D), builds all _nperm shuffled columns and scores CMI on the
    # device (VRAM-chunked over perms so the dense (chunk, Kx*Kyz) joint always fits a small card -- at the full-n
    # gate calls the support is near-continuous, Kz ~ 1e5, so the whole-batch joint is multi-GB), reducing the
    # floor/mean on device; only the two scalars return. Selection-EQUIVALENT (the device RNG stream differs from
    # numpy's, but the same (seed,salt) makes it reproducible and the quantile / mean over the draws agree within
    # the gate's razor tolerance -> identical F2 selection, verified). The earlier +8s "bench-rejection" was a
    # contention artifact (the A/B ran on a GPU shared with another job; see the corrected _fe_cmi_perm_null_gpu
    # module bench note). H2D audit of a 1M strict-resident fit: enabling this for ALL gate calls (the VRAM-chunk
    # removed the OOM that used to force the fallback) dropped total bulk H2D from 3528 to 1832 MB by killing the
    # 800 MB host-key keys + the per-perm candidate re-uploads. Falls back to the host-key batched path, then the
    # exact per-perm CPU loop, only on a genuine cupy error. Plain STRICT (RESIDENT off) is byte-identical -- this
    # branch is skipped entirely.
    try:
        from ._fe_cmi_perm_null_gpu import perm_null_gpu_resident_enabled
        _resident = perm_null_gpu_resident_enabled()
    except Exception:
        _resident = False
    if _resident and _cmi_gpu_enabled() and _nperm > 1:
        try:
            from ._fe_cmi_perm_null_gpu import conditional_perm_null_gpu
            # Pass the RESIDENT candidate code directly (conditional_perm_null_gpu resident-input branch reorders
            # ``dx[order]`` on device), so the candidate never re-crosses H2D at the ``permnull_cand_x`` site;
            # host code otherwise.
            return conditional_perm_null_gpu(
                cand_dev if cand_dev is not None else _host_x(), y_i, z_i, order=order, z_rank=z_rank,
                n_permutations=_nperm, quantile=quantile, seed=seed, salt=salt,
            )
        except Exception:
            logger.debug("GPU-resident conditional perm-null failed; using host/CPU path", exc_info=True)
    # BATCHED born-on-device null under STRICT (default OFF -> per-perm CPU loop below). y_i/z_i are
    # shuffle-invariant; only the within-stratum-shuffled candidate varies per perm -> build all _nperm
    # shuffled columns into one (n, _nperm) matrix (SAME rng draws as the loop) and score CMI(x_perm; y|z)
    # for every perm in ONE batched_cmi_gpu workload, replacing _nperm per-call cp.unique CMIs.
    if _cmi_gpu_enabled() and _nperm > 1:
        try:
            from ._fe_batched_mi import batched_cmi_gpu
            import cupy as cp
            # DEVICE within-stratum shuffle (2026-06-25). The host built the _nperm shuffled candidate
            # columns with a per-perm np.argsort (the dominant steady-state host cost on sparse conditional
            # joints, where the analytic null falls back to permutations). Move only the ARGSORT + gather +
            # CMI to the GPU while keeping the RNG KEY DRAW on the host (np ``rng.random``, cheap) so the
            # permutations are BIT-IDENTICAL to the per-perm CPU loop below: ``z_rank+keys`` has distinct
            # values per row (keys are distinct floats), so argsort has no ties and cp.argsort matches
            # np.argsort(kind="stable") element-for-element -> identical Xp -> identical nulls -> identical
            # floor/null-mean -> bit-identical selection (NOT merely statistically equivalent). One batched
            # (n,_nperm) device argsort replaces _nperm host argsorts; codes stay resident for the CMI.
            _xh = _host_x()
            x_sorted = _xh[order]
            keys = np.empty((n_size, _nperm), dtype=np.float64)
            for i in range(_nperm):
                keys[:, i] = rng.random(n_size)  # per-perm draw -> SAME sequence as the CPU loop
            z_rank_d = cp.asarray(z_rank)[:, None]
            within = cp.argsort(z_rank_d + cp.asarray(keys), axis=0)  # (n, _nperm) within-stratum orders
            x_sorted_d = cp.asarray(x_sorted)
            order_d = cp.asarray(order)
            Xp_d = cp.empty((n_size, _nperm), dtype=cp.int64)
            Xp_d[order_d, :] = x_sorted_d[within]  # xp[order] = x_sorted[within], per perm
            from ._fe_cmi_perm_null_gpu import _floor_mean_from_nulls_dev
            nulls_dev = batched_cmi_gpu(Xp_d, y_i, z_i, return_device=True)   # stays resident
            return _floor_mean_from_nulls_dev(cp, nulls_dev, quantile)
        except Exception:  # nosec B110 - optional/best-effort path, rationale documented
            pass  # any cupy error -> exact per-perm CPU loop below
    _xh = _host_x()
    x_sorted = _xh[order]
    # bench-attempt-rejected (2026-07-05): dropping the per-perm ``np.empty_like`` + scatter
    # ``x_perm[order] = x_sorted[within]`` by pre-permuting ``y_i``/``z_i`` by ``order`` once and
    # passing ``x_sorted[within]`` directly (histogram is row-order invariant) measured only
    # 1.01-1.07x (bench_perm_null_row_order_hoist.py) AND was NOT bit-identical -- reordering the
    # rows changes _renumber_joint's first-appearance cell labelling, so the entropy sum reduces in
    # a different order (~1e-13 drift). Not worth trading bit-identity for a win that vanishes to
    # 1.01x at n=30k (the njit CMI + argsort dominate, not the alloc/scatter).
    nulls = np.empty(_nperm, dtype=np.float64)
    for i in range(_nperm):
        keys = rng.random(n_size)
        within = np.argsort(z_rank + keys, kind="stable")  # within each (already-sorted) stratum block: random order
        x_perm = np.empty_like(_xh)
        x_perm[order] = x_sorted[within]
        nulls[i] = float(cmi_from_binned_fixed_yz(x_perm, y_i, z_i, h_yz, h_z, k_yz, k_z, n_f))
    return float(np.quantile(nulls, quantile)), float(np.mean(nulls))
