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

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def target_oversplit_floor_applies(
    factors_nbins,
    candidate_indices,
    y_index: int,
    n: int,
    *,
    oversplit_ratio: float = 3.0,
    min_rows_per_joint_cell: float = 8.0,
) -> bool:
    """Return ``True`` when the maxT noise floor should fire on a NARROW pool because the target is plug-in-bias-inflated AND the floor is itself statistically reliable.

    The wide-pool gate (``len(pool) >= screen_fdr_min_features``) catches embedding / TF-IDF matrices where best-of-p selection bias dominates. It MISSES a different
    finite-sample-bias regime that surfaces on narrow tabular pools: a heavy-tailed (log-normal) regression target that the supervised MDLP binner OVER-SPLITS into many
    bins (~30) while the features bin to ~5. The plug-in MI bias ``(nbins_x-1)*(nbins_y-1)/(2n)`` then lifts pure-noise columns past the abs/rel gain floors AFTER the
    genuine signals are picked, so a noise column leaks in even though the pool is only ~9 columns wide.

    A blunt pool-size drop cannot separate this from a dense weak-signal regression pool (e.g. sklearn diabetes: 10 genuine-but-weak features) where the SAME narrow pool
    must keep ALL features -- dropping the size gate there over-prunes 10 -> 2 and regresses R^2. The discriminator is the TARGET, expressed via two cheap, already-computed
    bin counts:

    * **over-split** -- ``nbins_y >= oversplit_ratio * median(nbins_x)``. The MDLP binner has split the target into many more bins than the features carry, which is the
      precondition for the ``(nbins_x-1)*(nbins_y-1)/(2n)`` plug-in bias to materially inflate noise MI. Heavy-tailed lognormal trips this (nbins_y~30 vs feat~5, ratio ~6);
      a linear / bimodal regression target (nbins_y~10-14 vs feat~5, ratio ~2) and any low-cardinality classification target (nbins_y in {2,3}) do NOT.

    * **reliable** -- ``n / (nbins_y * median(nbins_x)) >= min_rows_per_joint_cell``. The maxT floor is the q-quantile of the per-shuffle MAX corrected plug-in MI; when the
      (X, y) contingency table averages well under a handful of rows per cell the plug-in MI is itself dominated by finite-sample variance, so the floor explodes and would
      prune genuine weak signal. Diabetes (n=331, nbins_y=53, feat=5 -> ~1.2 rows per joint cell) FAILS this predicate -- so the floor stays OFF there and the legacy
      narrow-pool behaviour (no floor) is preserved exactly, keeping its 10 weak features. Lognormal (n=2500 -> ~16 rows per joint cell) passes, so the floor is trustworthy
      and fires.

    Both predicates must hold. The defaults (``oversplit_ratio=3.0``, ``min_rows_per_joint_cell=8.0``) sit with comfortable margin between the fire / no-fire cases measured
    across the regression + classification benchmark suite (lognormal over-split ratio ~6 and ~16 rows/cell vs the nearest no-fire cases: california_housing ratio 1.5,
    diabetes 1.2 rows/cell). Returns ``False`` on a degenerate pool (n too small, single-class target, no scorable feature) so the caller can treat it as "floor off".
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
    over_split = nbins_y >= float(oversplit_ratio) * median_feat_nbins
    if not over_split:
        return False
    rows_per_joint_cell = n / (nbins_y * median_feat_nbins)
    return rows_per_joint_cell >= float(min_rows_per_joint_cell)


def pooled_permutation_null_gain_floor(
    factors_data: np.ndarray,
    factors_nbins,
    candidate_indices,
    y_index: int,
    *,
    n_permutations: int = 25,
    quantile: float = 0.95,
    cardinality_bias_correction: bool = True,
    random_seed=None,
) -> float:
    """Return the maxT permutation-null gain floor for an order-1 candidate pool.

    ``factors_data`` must be the ordinal-encoded (int) screening matrix;
    ``candidate_indices`` are the column indices of the single-feature candidates
    (the ``y_index`` column is skipped if present). The returned scalar is the
    ``quantile``-th quantile of the per-shuffle MAX corrected marginal MI over the
    pool - a gain a genuine signal exceeds and chance-max noise does not.

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
        scaled_codes.append(xc * nbins_y)
        joint_card.append(nb * nbins_y)
        h_x.append(float(-(px * np.log(px)).sum()))
        mm_bias.append(((nb - 1) * (nbins_y - 1) / (2.0 * n)) if cardinality_bias_correction else 0.0)

    n_cand = len(scaled_codes)
    if n_cand < 2:
        return 0.0

    rng = np.random.default_rng(random_seed)
    y_perm = y_codes.copy()
    maxes = np.empty(int(n_permutations), dtype=np.float64)
    for k in range(int(n_permutations)):
        rng.shuffle(y_perm)  # in-place uniform permutation of the target labels
        best = 0.0
        for j in range(n_cand):
            jc = np.bincount(scaled_codes[j] + y_perm, minlength=joint_card[j]).astype(np.float64)
            pj = jc[jc > 0] * inv_n
            h_xy = -(pj * np.log(pj)).sum()
            mi = h_x[j] + h_y - h_xy - mm_bias[j]
            if mi > best:
                best = mi
        maxes[k] = best

    return float(np.quantile(maxes, float(quantile)))


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
    random_seed=None,
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
    """
    n = int(factors_data.shape[0])
    n_pairs = int(pair_a.shape[0])
    if n < 8 or n_permutations < 1 or n_pairs < 2:
        return 0.0
    if int(np.asarray(freqs_y).shape[0]) < 2:
        return 0.0

    # Reuse the exact batched plug-in joint-MI kernel the FE pair screen uses
    # (CPU njit prange -- deterministic, GPU-independent for the floor compute),
    # so the per-shuffle max is on the same scale as the gated ``pair_mi``.
    from .info_theory import batch_pair_mi_prange

    pa = np.ascontiguousarray(pair_a, dtype=np.int64)
    pb = np.ascontiguousarray(pair_b, dtype=np.int64)
    nb = np.ascontiguousarray(nbins)
    fy = np.ascontiguousarray(freqs_y, dtype=np.float64)

    rng = np.random.default_rng(random_seed)
    y_perm = np.ascontiguousarray(classes_y).copy()
    maxes = np.empty(int(n_permutations), dtype=np.float64)
    for k in range(int(n_permutations)):
        rng.shuffle(y_perm)  # in-place uniform permutation of the target codes
        mis = batch_pair_mi_prange(factors_data, pa, pb, nb, y_perm, fy)
        maxes[k] = float(np.max(mis)) if mis.size else 0.0

    return float(np.quantile(maxes, float(quantile)))
