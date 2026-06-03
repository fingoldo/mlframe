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

import numpy as np


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
