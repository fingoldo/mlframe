"""Westfall-Young maxT permutation-null gain floor for high-dimensional screening.

In a wide candidate pool (embedding / TF-IDF matrices, p >> sqrt(n)) the MAXIMUM
marginal MI over p pure-noise columns is a positive order statistic that grows
with p - the classic multiple-comparison selection bias. Per-candidate
Miller-Madow correction centres each column's EXPECTED bias near zero but does
NOT remove this max-over-p inflation: with 497 i.i.d. noise columns the best one
clears the absolute / relative gain floors purely by chance, so MRMR's greedy
admits a noise cloud (Layer-20 p=500: 15 noise dims survive a <=15 bound).

The fix is a permutation null on the SELECTION statistic itself. Shuffle y K
times (destroying every X-y dependency while preserving each column's marginal
distribution and the pool size p), and for each shuffle record the MAXIMUM
corrected marginal MI over the whole candidate pool. The q-th quantile of that
K-sample max-distribution is a gain floor that a genuine signal clears but the
best-of-p noise does not. It is the Westfall-Young maxT step-down null restricted
to its single most important quantity - the distribution of the best chance hit.

The floor is SELF-GATING: it is the chance ceiling for THIS pool, so it is tiny
at small p (a coef-0.4 weak signal at p=10 clears it) and large at high p (it
rejects the Layer-20 noise cloud). Computed once per ``screen_predictors`` call
on the order-1 pool, and only applied when the pool is wide enough that the
max-over-p selection bias actually bites - the narrow tabular suite (p below
``min_features``) stays byte-stable.

Why a marginal-MI null (not the Fleuret relevance-redundancy gain): at
interactions_order == 1 a candidate's Fleuret relevance IS its marginal MI, and
for the noise columns the redundancy term against the (few) genuine signals is
~0 (independent Gaussian noise), so the order-1 gain at the gate tracks the
marginal MI the null is built on. Pure-synergy features (XOR partners, ~0
marginal) are never order-1-selected - they enter via the FE synergy bootstrap
as an engineered JOINT whose own marginal MI is large - so a marginal floor at
order 1 cannot prune genuine synergy.
"""
from __future__ import annotations

import logging

import numba
import numpy as np
from numba import prange


@numba.njit(cache=True, parallel=True)
def _pooled_gain_floor_perms_njit(scaled_flat, offsets, joint_card, h_x, mm_bias, h_y, y_perms, inv_n):
    """Per-shuffle MAX corrected marginal MI over the candidate pool, fused into one njit pass.

    ``scaled_flat`` concatenates every candidate's ``x_code * nbins_y`` column (segment ``j`` is ``scaled_flat[offsets[j]:offsets[j+1]]``); ``y_perms[k]`` is the
    pre-generated k-th target shuffle (numpy RNG owns the draw sequence so the floor stays bit-identical to the per-shuffle ``rng.shuffle`` path). For each shuffle the joint
    H(x, y_perm) is the only term that changes, so ``h_x`` / ``h_y`` / ``mm_bias`` are precomputed once and threaded in. The per-cell ``-p*log(p)`` accumulates in
    code-ascending order; the only divergence from the numpy ``-(p*log(p)).sum()`` reduction is FP reduction-order (~1e-16, far below selection scale).

    The K-shuffle loop is ``prange``-parallel: each iteration ``k`` writes only ``maxes[k]`` from read-only shared inputs and owns a private ``counts`` scratch, so the result is
    BIT-IDENTICAL to the serial scan (per-shuffle entropy reduction order unchanged -- verified max|diff|=0.0, identical 0.95-quantile floor across n=10k/100k x p=50/200) while
    scaling with cores. ~8x at 22 threads (bench _benchmarks/bench_pooled_gain_floor_perms_prange.py: 6.8s -> 0.78s at n=100k/p=200/K=200)."""
    nperm = y_perms.shape[0]
    n = y_perms.shape[1]
    ncand = offsets.shape[0] - 1
    maxes = np.empty(nperm, dtype=np.float64)
    max_jc = 0
    for j in range(ncand):
        if joint_card[j] > max_jc:
            max_jc = joint_card[j]
    for k in prange(nperm):
        counts = np.empty(max_jc, dtype=np.float64)  # per-thread scratch (prange-private)
        yp = y_perms[k]
        best = 0.0
        for j in range(ncand):
            jc = joint_card[j]
            for t in range(jc):
                counts[t] = 0.0
            s0 = offsets[j]
            for i in range(n):
                counts[scaled_flat[s0 + i] + yp[i]] += 1.0
            h_xy = 0.0
            for t in range(jc):
                c = counts[t]
                if c > 0.0:
                    p = c * inv_n
                    h_xy -= p * np.log(p)
            mi = h_x[j] + h_y - h_xy - mm_bias[j]
            if mi > best:
                best = mi
        maxes[k] = best
    return maxes

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def target_oversplit_floor_applies(
    factors_nbins: np.ndarray,
    candidate_indices: np.ndarray,
    y_index: int,
    n: int,
    *,
    oversplit_ratio: float = 1.0,
    min_rows_per_joint_cell: float = 8.0,
) -> bool:
    """Return ``True`` when the maxT noise floor should fire on a NARROW pool because the target is plug-in-bias-inflated AND the floor is itself statistically reliable.

    The wide-pool gate (``len(pool) >= screen_fdr_min_features``) catches embedding / TF-IDF matrices where best-of-p selection bias dominates. It MISSES a different
    finite-sample-bias regime that surfaces on narrow tabular pools: a heavy-tailed (log-normal) regression target whose continuous quantile binning yields a HIGH-CARDINALITY
    target (~10 equal-frequency bins) while the noise features bin to the same ~10 levels. The plug-in MI bias ``(nbins_x-1)*(nbins_y-1)/(2n)`` then lifts pure-noise columns
    past the abs/rel gain floors AFTER the genuine signals are picked, so a noise column leaks in even though the pool is only ~9 columns wide.

    HISTORY: the original gate (2026-06-04) keyed ``over-split`` on ``nbins_y >= 3 * median(nbins_x)`` because the then-default adaptive ``nbins_strategy="mdlp"`` OVER-SPLIT
    the heavy-tailed target into ~30 bins (vs feat ~5). The 2026-06-10 TARGET REBIN GUARD (``_mrmr_fit_impl/_fit_impl_core.py``) re-bins the target back to the legacy
    equal-frequency ``quantization_nbins`` (~10) encoding -- correct, because supervised MDLP on the injected target is self-referential and degrades MI sensitivity -- but
    that removed the ~30-bin over-split signature, so the ``ratio>=3`` gate stopped firing in production and the lognormal noise re-leaked. The plug-in-bias mechanism that
    admits noise is unchanged (the maxT floor still cleanly separates signal >> floor > noise at nbins_y=10); only the *gate predicate* was keyed on a binning artifact the
    rebin guard now removes. The discriminator is the TARGET, expressed via two cheap, already-computed bin counts:

    * **high-cardinality target** -- ``nbins_y >= oversplit_ratio * median(nbins_x)`` with ``oversplit_ratio=1.0``: the target carries at least as many levels as the features.
      A continuous (regression) target quantile-binned to ``quantization_nbins`` (~10, matching feature cardinality) trips this; a low-cardinality CLASSIFICATION target
      (nbins_y in {2,3} << feat ~5-10) does NOT -- there the plug-in bias ``(nbins_x-1)*(nbins_y-1)/(2n)`` is small and the few target classes carry no over-fit headroom.

    * **reliable** -- ``n / (nbins_y * median(nbins_x)) >= min_rows_per_joint_cell``. The maxT floor is the q-quantile of the per-shuffle MAX corrected plug-in MI; when the
      (X, y) contingency table averages well under a handful of rows per cell the plug-in MI is itself dominated by finite-sample variance, so the floor explodes and would
      prune genuine weak signal. THIS predicate is the real safety rail separating the lognormal WIN from the diabetes NO-REGRESSION: at the production quantile-10 binning
      diabetes (n=330, nbins_y=10, feat=10 -> 3.3 rows per joint cell) FAILS it, so the floor stays OFF and its 10 weak features survive; lognormal (n=2500, nbins_y=10,
      feat=10 -> ~25-50 rows per joint cell) passes, so the floor is trustworthy and fires, rejecting the 6 noise columns (max noise corrected-MI ~0.005 < floor ~0.006-0.007
      while the 3 signals score 0.025-0.53).

    Both predicates must hold. The defaults (``oversplit_ratio=1.0``, ``min_rows_per_joint_cell=8.0``) sit with comfortable margin between the fire / no-fire cases measured at
    the production equal-frequency binning (lognormal ~25-50 rows/cell vs diabetes 3.3 rows/cell). Returns ``False`` on a degenerate pool (n too small, single-class target,
    no scorable feature) so the caller can treat it as "floor off".
    """
    if n < 8:
        return False
    y_idx = int(y_index)
    nbins_y = int(factors_nbins[y_idx])
    if nbins_y < 2:
        return False
    feat_nbins = []
    for c in candidate_indices:
        ci = int(c)
        if ci == y_idx:
            continue
        nb = int(factors_nbins[ci])
        if nb >= 2:
            feat_nbins.append(nb)
    if not feat_nbins:
        return False
    median_feat_nbins = float(np.median(feat_nbins))
    if median_feat_nbins < 1.0:
        return False
    # High-cardinality target: nbins_y at least matches the feature cardinality (a continuous
    # regression target), distinguishing it from a low-card classification target whose few
    # classes carry negligible plug-in-bias headroom. (Pre-rebin-guard this keyed on a 3x MDLP
    # over-split; the 2026-06-10 target-rebin guard removed that signature -- see docstring.)
    high_cardinality_target = nbins_y >= float(oversplit_ratio) * median_feat_nbins
    if not high_cardinality_target:
        return False
    rows_per_joint_cell = n / (nbins_y * median_feat_nbins)
    return rows_per_joint_cell >= float(min_rows_per_joint_cell)


def pooled_permutation_null_gain_floor(
    factors_data: np.ndarray,
    factors_nbins: np.ndarray,
    candidate_indices: np.ndarray,
    y_index: int,
    *,
    n_permutations: int = 200,
    quantile: float = 0.95,
    cardinality_bias_correction: bool = True,
    random_seed: int | None = None,
) -> float:
    """Return the maxT permutation-null gain floor for an order-1 candidate pool.

    REUSE-AUDIT RU-5 disposition (2026-06-19): sharing the permuted-y draws across the order-1/2/3 null floors
    + the seeder self-gate was evaluated and REJECTED. The only cost sharing would save is generating the K
    shuffles -- measured 8.76ms for K=25 at n=20000 -- which is negligible against the per-shuffle histogram
    RESCORING that dominates each floor, and the floor statistics differ per order so they must each be
    computed anyway. Sharing the same draws across orders would also correlate the per-order null estimates
    (they are intended independent one-sided thresholds). Not worth it.

    ``factors_data`` must be the ordinal-encoded (int) screening matrix;
    ``candidate_indices`` are the column indices of the single-feature candidates
    (the ``y_index`` column is skipped if present). The returned scalar is the
    ``quantile``-th quantile of the per-shuffle MAX corrected marginal MI over the
    pool - a gain a genuine signal exceeds and chance-max noise does not.

    The floor is the ``quantile`` (default 0.95) of the EXTREME (per-shuffle MAX) over ``n_permutations``
    shuffles. The default is 200 (not 25): a 95th-percentile order statistic places only ~K*(1-q) draws
    above it, so K=25 leaves ~1.25 draws in the estimated tail and the floor wobbles badly run-to-run
    (high quantile-estimator variance); K=200 puts ~10 draws in the tail and the run-to-run std of the
    floor drops several-fold on a fixed null (see _benchmarks/bench_maxt_floor_stability.py). A lower-
    variance null floor is the correct behavior, so the default favors floor stability over the marginally
    cheaper rescore -- each extra shuffle pays a full pool-rescore, but the floor is computed once per
    screen and 200 stays sub-second at production widths. Drop ``n_permutations`` to the legacy 25 only to
    reproduce a pre-2026-06 floor (note the floor MOVES with K, so re-validate downstream selection).

    Returns ``0.0`` (a no-op floor) when the pool is degenerate (n too small,
    fewer than two scorable candidates, or a constant target), so callers can
    unconditionally compare ``gain >= floor`` without special-casing.
    """
    n = int(factors_data.shape[0])
    if n < 8 or n_permutations < 1:
        return 0.0

    y_idx = int(y_index)
    nbins_y = int(factors_nbins[y_idx])
    if nbins_y < 2:
        return 0.0

    inv_n = 1.0 / n
    y_codes = np.ascontiguousarray(factors_data[:, y_idx]).astype(np.int64)

    # H(y) is invariant under permutation (relabelling only); H(x) too. Only the
    # joint H(x, y_perm) changes per shuffle, so precompute the invariants once.
    y_counts = np.bincount(y_codes, minlength=nbins_y).astype(np.float64)
    py = y_counts[y_counts > 0] * inv_n
    h_y = float(-(py * np.log(py)).sum())
    # Idea-#9 (backlog #4): Miller-Madow bias uses the EFFECTIVE OCCUPIED bin
    # count, not the nominal cardinality. Heavy-tailed engineered columns
    # (e.g. a**2/b) bin to ~8 occupied cells out of nbins=16; the MM term
    # ``(k_x-1)(k_y-1)/2n`` with nominal k OVER-corrects (it charges bias for
    # empty cells that contribute no plug-in inflation), pushing the chance
    # floor down and understating it. Using k_eff = #occupied bins is the
    # statistically correct MM and tracks the true (null=0) MI 3.1-3.2x tighter
    # on heavy-tailed cols (residual |MI|: nominal 0.045/0.018 -> occupied
    # 0.014/0.006 at n=2000/5000). y's occupied count is invariant under the
    # relabelling shuffle, so it is fixed here.
    ky_eff = int(py.shape[0])

    scaled_codes = []   # x_codes * nbins_y  (so joint = scaled + y_perm)
    joint_card = []     # nbins_x * nbins_y
    h_x = []            # marginal entropy of each candidate
    mm_bias = []        # Miller-Madow bias subtracted from each candidate's MI
    for c in candidate_indices:
        ci = int(c)
        if ci == y_idx:
            continue
        nb = int(factors_nbins[ci])
        if nb < 2:
            continue
        xc = np.ascontiguousarray(factors_data[:, ci]).astype(np.int64)
        xcounts = np.bincount(xc, minlength=nb).astype(np.float64)
        px = xcounts[xcounts > 0] * inv_n
        kx_eff = int(px.shape[0])  # occupied bins of this candidate (idea-#9)
        # int32 codes (not int64): the joint code ``x_code*nbins_y + y_code`` is < nbins_x*nbins_y,
        # always within int32 -- so this is BIT-IDENTICAL but halves the (n_cand x n) ``scaled_flat``
        # and (nperm x n) ``y_perms`` pools below, which reach GBs at n=1M and drove an OOM
        # (_permutation_null.py:239, (n_cand*n,) int64). Used only as histogram indices downstream.
        scaled_codes.append((xc * nbins_y).astype(np.int32))
        joint_card.append(nb * nbins_y)
        h_x.append(float(-(px * np.log(px)).sum()))
        mm_bias.append(((kx_eff - 1) * (ky_eff - 1) / (2.0 * n)) if cardinality_bias_correction else 0.0)

    n_cand = len(scaled_codes)
    if n_cand < 2:
        return 0.0

    nperm = int(n_permutations)
    rng = np.random.default_rng(random_seed)
    # Pre-generate the K target shuffles in numpy (it owns the RNG draw sequence,
    # matching the legacy per-shuffle ``rng.shuffle(y_perm)`` order exactly) so the
    # fused njit MI pass below stays bit-identical to the pure-Python loop.
    # y codes are < nbins_y -> int32 holds them; the (nperm x n) pool halves vs int64. shuffle still
    # runs on an int32 buffer (numpy RNG draw order preserved -> floor stays bit-identical).
    import os as _os
    _fdr_dt = np.int64 if _os.environ.get("MLFRAME_FDR_NULL_INT32", "") == "0" else np.int32
    y_perm = y_codes.astype(_fdr_dt)
    y_perms = np.empty((nperm, n), dtype=_fdr_dt)
    for k in range(nperm):
        rng.shuffle(y_perm)
        y_perms[k] = y_perm
    scaled_flat = np.concatenate(scaled_codes).astype(_fdr_dt)
    offsets = np.arange(n_cand + 1, dtype=np.int64) * n  # int64: true flat indices, can exceed 2^31
    _jc = np.asarray(joint_card, dtype=np.int64)
    _hx = np.asarray(h_x, dtype=np.float64)
    _mm = np.asarray(mm_bias, dtype=np.float64)

    # RESIDENT-GPU CROSSOVER (iter16, 2026-06-23): the per-shuffle MAX corrected MI over the pool is a
    # (nperm x n_cand x n) histogram+MI loop -- ONE batched workload (no per-pair launches, the iter13 trap).
    # Route it to the resident cupy twin ONLY where a per-host KTC sweep MEASURED it faster than this njit
    # kernel (wide pools p>=64 and/or large n on a capable card); the narrow tabular floor stays on CPU where
    # the njit is already sub-second. Selection-equivalent (per-cell entropy differs only in FP reduction
    # order ~1e-15; the host owns the final np.quantile). Any cupy/device error falls back to njit -- the
    # floor is NEVER broken by a GPU problem (correctness first).
    maxes = None
    try:
        from ._permutation_null_resident_ktc import permnull_use_resident
        if permnull_use_resident(n, n_cand, nperm):
            from ._gpu_policy import gpu_globally_disabled
            if not gpu_globally_disabled():
                from ._permutation_null_resident import pooled_gain_floor_perms_cupy
                maxes = pooled_gain_floor_perms_cupy(
                    scaled_flat, offsets, _jc, _hx, _mm, float(h_y), y_perms, float(inv_n),
                )
    except Exception:
        maxes = None  # fall through to the exact njit kernel
    if maxes is None:
        maxes = _pooled_gain_floor_perms_njit(
            scaled_flat, offsets, _jc, _hx, _mm, float(h_y), y_perms, float(inv_n),
        )

    return float(np.quantile(maxes, float(quantile)))


@numba.njit(cache=True)
def _pairwise_occupied_joint_k_njit(factors_data, pair_a, pair_b, nbins):
    """njit twin of :func:`_pairwise_occupied_joint_k`: counts the SAME distinct joint
    codes ``a*nbins_b + b`` per pair via a flat boolean ``seen`` buffer of length
    ``nbins_a*nbins_b`` instead of a Python ``set``. BIT-IDENTICAL by construction (it
    enumerates the identical code per row and counts first-occurrences). ~90-240x faster
    than the interpreter set-per-pair loop (bench bench_pairwise_occupied_joint_k.py)."""
    n = factors_data.shape[0]
    n_pairs = pair_a.shape[0]
    out = np.empty(n_pairs, dtype=np.int64)
    for p in range(n_pairs):
        a = pair_a[p]; b = pair_b[p]
        nb_a = nbins[a]; nb_b = nbins[b]
        seen = np.zeros(nb_a * nb_b, dtype=np.uint8)
        cnt = 0
        for i in range(n):
            code = factors_data[i, a] * nb_b + factors_data[i, b]
            if seen[code] == 0:
                seen[code] = 1
                cnt += 1
        out[p] = cnt
    return out


def _pairwise_occupied_joint_k(
    factors_data: np.ndarray, pair_a: np.ndarray, pair_b: np.ndarray, nbins: np.ndarray,
) -> np.ndarray:
    """Per-pair OCCUPIED joint-bin count of ``(x_a, x_b)`` -- the cardinality the
    plug-in joint MI :func:`batch_pair_mi_prange` actually sees (backlog #4).

    Permutation-INVARIANT (depends only on the X-columns, never on y), so it is
    precomputed ONCE and reused across all shuffles. Returns an ``int64`` array of
    length ``len(pair_a)``; entry ``k`` is ``#{distinct (a*nbins_b + b) codes}`` for
    pair ``k``. Used to subtract the per-pair Miller-Madow MI bias term consistently
    from BOTH the floor's per-shuffle joint MIs AND the gate's observed ``pair_mi``.

    Delegates to the njit boolean-seen kernel (bit-identical distinct-count, ~90-240x
    over the prior Python set-per-pair loop). Only the index/bin arrays are normalised
    to int64; ``factors_data`` keeps its native (typically int32) dtype so a wide
    screening matrix is NOT int64-copied (RAM rule) -- numba indexes it directly."""
    pa = np.ascontiguousarray(pair_a, dtype=np.int64)
    pb = np.ascontiguousarray(pair_b, dtype=np.int64)
    nb = np.ascontiguousarray(nbins, dtype=np.int64)
    return _pairwise_occupied_joint_k_njit(factors_data, pa, pb, nb)


def pooled_pair_permutation_null_joint_mi_floor(
    factors_data: np.ndarray,
    nbins: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    *,
    n_permutations: int = 25,
    quantile: float = 0.95,
    random_seed: int | None = None,
    mm_debias: bool = False,
) -> float:
    """Return the ORDER-2 maxT permutation-null floor for a prospective-pair pool.

    The FE step ranks prospective engineered PAIRS by the JOINT MI of
    ``(x_i, x_j; y)`` over an O(p^2) candidate pool. At high p the MAXIMUM joint
    MI over PURE-NOISE pairs is a positive order statistic that grows with the
    pool size -- the SAME best-of-p selection bias the order-1 floor rejects, now
    at order 2. The per-pair prevalence gates (``fe_min_pair_mi_prevalence`` /
    ``fe_synergy_min_prevalence``) are PER-PAIR; they centre each pair's own
    finite-sample bias but do NOT account for the max-over-pool selection, so a
    wide noise matrix still surfaces "synergistic-looking" noise pairs whose
    joint MI is merely the best chance hit.

    This is the Westfall-Young maxT step-down null on the SELECTION statistic:
    shuffle the discretised target ``classes_y`` K times (destroying every X-y
    dependency while preserving each column's marginal distribution AND the pool
    size), and for each shuffle record the MAXIMUM joint MI over the WHOLE
    candidate pair pool via the SAME batched plug-in joint-MI estimator the FE
    screen scores ``pair_mi`` with (:func:`batch_pair_mi_prange`), so the floor
    is on the exact same scale as the values it gates. The ``quantile``-th
    quantile of that K-sample max-distribution is a joint-MI floor a genuine
    synergy pair (XOR / product / bilinear -- joint MI FAR above chance) clears
    and the best-of-p noise pair does not.

    ``classes_y`` is the per-row ordinal target code (the array the FE joint-MI
    sweep scores against); ``freqs_y`` its class-probability vector, INVARIANT
    under permutation (relabelling rows only), so it is reused across shuffles.
    The floor is applied IN ADDITION to the existing per-pair prevalence gates.

    Returns ``0.0`` (a no-op floor) when the pool is degenerate (n too small,
    fewer than two candidate pairs, single-class target, or no permutations
    requested), so callers can unconditionally compare ``pair_mi >= floor``.

    MM-DEBIAS (2026-06-09, backlog #1 IRON RULE). When ``mm_debias`` the FE
    joint-prevalence RATIO gate downstream subtracts the Miller-Madow joint-MI
    bias from its ``pair_mi`` denominator, which LOWERS the bar and admits more
    pairs. To keep this floor (the outer best-of-pool guard) on the SAME scale --
    so lowering the ratio bar does NOT weaken the floor -- we subtract the per-pair
    Miller-Madow MI bias ``(k_joint-1)(k_y-1)/2n`` (occupied joint-K, #4) from EACH
    pair's joint MI BEFORE the per-shuffle max. The per-pair bias is permutation-
    invariant (X-columns + k_y unchanged under y-shuffle), so it is precomputed once
    and applied to every shuffle, and the caller subtracts the IDENTICAL per-pair
    term from the observed ``pair_mi`` before the ``>= floor`` comparison (consistent
    debias on both sides). ``->`` the raw floor as ``n -> inf``.
    """
    n = int(factors_data.shape[0])
    n_pairs = int(pair_a.shape[0])
    if n < 8 or n_permutations < 1 or n_pairs < 2:
        return 0.0
    k_y = int(np.asarray(freqs_y).shape[0])
    if k_y < 2:
        return 0.0

    # Reuse the exact batched plug-in joint-MI kernel the FE pair screen uses
    # (CPU njit prange -- deterministic, GPU-independent for the floor compute),
    # so the per-shuffle max is on the same scale as the gated ``pair_mi``.
    from .info_theory import batch_pair_mi_prange

    pa = np.ascontiguousarray(pair_a, dtype=np.int64)
    pb = np.ascontiguousarray(pair_b, dtype=np.int64)
    nb = np.ascontiguousarray(nbins)
    fy = np.ascontiguousarray(freqs_y, dtype=np.float64)

    # Per-pair Miller-Madow bias vector (occupied joint-K), permutation-invariant.
    mm_bias = None
    if mm_debias:
        k_joint = _pairwise_occupied_joint_k(factors_data, pa, pb, nb)
        mm_bias = (k_joint - 1).astype(np.float64) * float(k_y - 1) / (2.0 * n)
        mm_bias[k_joint <= 1] = 0.0

    rng = np.random.default_rng(random_seed)
    y_perm = np.ascontiguousarray(classes_y).copy()
    maxes = np.empty(int(n_permutations), dtype=np.float64)
    for k in range(int(n_permutations)):
        rng.shuffle(y_perm)  # in-place uniform permutation of the target codes
        mis = batch_pair_mi_prange(factors_data, pa, pb, nb, y_perm, fy)
        if mm_bias is not None and mis.size:
            mis = mis - mm_bias
        maxes[k] = float(np.max(mis)) if mis.size else 0.0

    return float(np.quantile(maxes, float(quantile)))


def pooled_triple_permutation_null_joint_mi_floor(
    factors_data: np.ndarray,
    nbins: np.ndarray,
    triple_a: np.ndarray,
    triple_b: np.ndarray,
    triple_c: np.ndarray,
    classes_y: np.ndarray,
    freqs_y: np.ndarray,
    *,
    n_permutations: int = 25,
    quantile: float = 0.95,
    random_seed: int | None = None,
) -> float:
    """Return the ORDER-3 maxT permutation-null floor for a candidate-TRIPLE pool.

    The mandatory safety rail (backlog #7) for any 3-way interaction proposer (the
    surrogate-GBM split-co-occurrence seeder #6, the CMI-lattice #10, ...): the
    triplet / quadruplet FE modules historically seed by univariate top-N and lack
    an order-MATCHED permutation floor, so OPENING 3-way generation WILL surface
    chance-max noise triples -- a wide noise matrix's MAXIMUM 3-D joint MI over the
    proposed triple pool is a positive order statistic that grows with the pool size,
    the SAME best-of-pool selection bias the order-1 / order-2 floors reject, now at
    order 3 (and STRONGER: the 3-way joint cardinality inflates the plug-in joint MI
    further).

    This is the Westfall-Young maxT step-down null on the order-3 selection statistic:
    shuffle the discretised target ``classes_y`` K times (destroying every X-y
    dependency while preserving each column's marginal AND the pool size), and for each
    shuffle record the MAXIMUM 3-way joint MI over the WHOLE candidate-triple pool via
    the SAME batched plug-in joint-MI estimator the screen scores a triple with
    (:func:`batch_triple_mi_prange`, which dense-renumbers the 3-way joint so its
    cardinality stays <= n). The ``quantile``-th quantile of that K-sample
    max-distribution is a joint-MI floor a genuine 3-way needle (XOR / triple product
    -- joint MI FAR above chance) clears and the best-of-pool noise triple does not.

    ``classes_y`` is the per-row ordinal target code; ``freqs_y`` its class-probability
    vector, INVARIANT under permutation, so it is reused across shuffles. The floor is
    applied IN ADDITION to any per-triple gates, mirroring the order-2 floor's role.

    Returns ``0.0`` (a no-op floor) when the pool is degenerate (n too small, fewer than
    two candidate triples, single-class target, or no permutations requested), so callers
    can unconditionally compare ``triple_mi >= floor``. ``->`` the raw chance ceiling as
    the pool shrinks (a proposer-pruned small triple pool is LESS punishing than all
    C(p, 3) -- tighter multiple-comparison correction, the architectural through-line).
    """
    n = int(factors_data.shape[0])
    n_triples = int(triple_a.shape[0])
    if n < 8 or n_permutations < 1 or n_triples < 2:
        return 0.0
    k_y = int(np.asarray(freqs_y).shape[0])
    if k_y < 2:
        return 0.0

    # Reuse the exact batched plug-in 3-way joint-MI kernel a triple is scored with
    # (CPU njit prange -- deterministic, GPU-independent for the floor compute), so the
    # per-shuffle max is on the same scale as the gated ``triple_mi``.
    from .info_theory import batch_triple_mi_prange

    ta = np.ascontiguousarray(triple_a, dtype=np.int64)
    tb = np.ascontiguousarray(triple_b, dtype=np.int64)
    tc = np.ascontiguousarray(triple_c, dtype=np.int64)
    nb = np.ascontiguousarray(nbins)
    fy = np.ascontiguousarray(freqs_y, dtype=np.float64)

    rng = np.random.default_rng(random_seed)
    y_perm = np.ascontiguousarray(classes_y).copy()
    maxes = np.empty(int(n_permutations), dtype=np.float64)
    for k in range(int(n_permutations)):
        rng.shuffle(y_perm)  # in-place uniform permutation of the target codes
        mis = batch_triple_mi_prange(factors_data, ta, tb, tc, nb, y_perm, fy)
        maxes[k] = float(np.max(mis)) if mis.size else 0.0

    return float(np.quantile(maxes, float(quantile)))


def pairwise_mm_joint_bias(
    factors_data: np.ndarray,
    pair_a: np.ndarray,
    pair_b: np.ndarray,
    nbins: np.ndarray,
    k_y: int,
) -> np.ndarray:
    """Per-pair Miller-Madow joint-MI bias ``(k_joint-1)(k_y-1)/2n`` (occupied
    joint-K) for a candidate-pair pool. Public wrapper around
    :func:`_pairwise_occupied_joint_k` so the FE gate can subtract the SAME per-pair
    term from the observed ``pair_mi`` that the MM-debiased maxT floor subtracted
    from its per-shuffle joint MIs (IRON RULE: consistent debias on both sides).
    Returns an all-zero vector when ``k_y <= 1`` (degenerate target)."""
    n = int(factors_data.shape[0])
    pa = np.ascontiguousarray(pair_a, dtype=np.int64)
    pb = np.ascontiguousarray(pair_b, dtype=np.int64)
    nb = np.ascontiguousarray(nbins)
    if int(k_y) <= 1 or n <= 0:
        return np.zeros(int(pa.shape[0]), dtype=np.float64)
    k_joint = _pairwise_occupied_joint_k(factors_data, pa, pb, nb)
    bias = (k_joint - 1).astype(np.float64) * float(int(k_y) - 1) / (2.0 * n)
    bias[k_joint <= 1] = 0.0
    return bias
