"""Per-fold-majority acceptance criterion for greedy feature elimination/addition decisions.

A greedy feature-selection step (RFECV's drop-worst-feature, or any forward/backward elimination loop) that
accepts a change purely on AGGREGATE (mean) CV delta can be fooled by a feature that happens to help one
lucky fold a lot while hurting the rest -- the mean looks positive, but the decision doesn't generalize. A
Home-Credit-3rd-place refinement: only accept a change if it improves the score in a MAJORITY of individual
folds, not just on average. This module provides that acceptance criterion (plus optional multi-seed
averaging per fold to fight decision noise, per a Porto-Seguro-3rd-place companion trick) as a standalone,
reusable primitive -- usable inside any greedy elimination loop, including as a custom acceptance callback
for mlframe's existing RFECV.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
from scipy import stats as _scipy_stats


def _wilson_lower_bound(fraction_improved: float, n_folds: int, confidence: float) -> float:
    """Wilson score interval lower bound on a binomial proportion.

    Unlike the raw ``fraction_folds_improved``, this discounts the estimate by sample size -- the same 3/5
    fraction is treated as much less trustworthy than 5/5, and both are treated as less trustworthy than the
    identical fraction observed over more folds. Used as a continuous "agreement score" so callers can pick
    their own strictness instead of only getting the hard-coded ``min_fraction`` majority-vote verdict.
    """
    z = float(_scipy_stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    n = float(n_folds)
    denom = 1.0 + z * z / n
    center = fraction_improved + z * z / (2.0 * n)
    adj = z * np.sqrt(fraction_improved * (1.0 - fraction_improved) / n + z * z / (4.0 * n * n))
    return float(max(0.0, (center - adj) / denom))


def per_fold_majority_accept(
    baseline_fold_scores: Sequence[float],
    candidate_fold_scores: Sequence[float],
    maximize: bool = True,
    min_fraction: float = 0.6,
    compute_agreement_score: bool = False,
    agreement_confidence: float = 0.95,
) -> dict:
    """Accept a candidate feature change only if it improves the score in a majority of individual folds.

    Parameters
    ----------
    baseline_fold_scores, candidate_fold_scores
        Per-fold scores (same length, paired by fold) for the current-best and the candidate respectively.
    maximize
        ``True`` when higher is better.
    min_fraction
        The candidate is accepted when the fraction of folds it improves reaches this threshold (default
        ``0.6`` -- a genuine majority with margin, not a bare 50%+1 tie).
    compute_agreement_score
        Opt-in. When ``True``, also returns a continuous ``agreement_score`` (Wilson score interval lower
        bound on ``fraction_folds_improved``) so callers can tune their own strictness threshold instead of
        only consuming the binary ``accept`` verdict. Default-``False`` behavior (and the returned dict's
        keys/values under default args) is unchanged.
    agreement_confidence
        Confidence level for the Wilson interval used by ``agreement_score`` (only used when
        ``compute_agreement_score`` is ``True``).

    Returns
    -------
    dict
        ``fraction_folds_improved``, ``mean_delta`` (candidate mean - baseline mean, sign per ``maximize``),
        ``accept`` (bool), plus ``agreement_score`` when ``compute_agreement_score`` is ``True``.
    """
    baseline = np.asarray(baseline_fold_scores, dtype=np.float64)
    candidate = np.asarray(candidate_fold_scores, dtype=np.float64)
    if baseline.shape != candidate.shape:
        raise ValueError("per_fold_majority_accept: baseline_fold_scores and candidate_fold_scores must have the same shape")
    if baseline.shape[0] == 0:
        raise ValueError("per_fold_majority_accept: fold score arrays must be non-empty")

    improved = (candidate > baseline) if maximize else (candidate < baseline)
    fraction_improved = float(np.mean(improved))
    mean_delta = float(np.mean(candidate) - np.mean(baseline))
    if not maximize:
        mean_delta = -mean_delta

    result = {
        "fraction_folds_improved": fraction_improved,
        "mean_delta": mean_delta,
        "accept": fraction_improved >= min_fraction,
    }
    if compute_agreement_score:
        result["agreement_score"] = _wilson_lower_bound(fraction_improved, baseline.shape[0], agreement_confidence)
    return result


def seed_averaged_fold_scores(
    score_fn: Callable[[Optional[int]], Sequence[float]],
    n_repeats: int = 4,
    base_seed: Optional[int] = None,
) -> np.ndarray:
    """Average per-fold scores across ``n_repeats`` independent seeds to reduce noisy accept/reject decisions.

    Parameters
    ----------
    score_fn
        ``score_fn(seed) -> (n_folds,) per-fold scores`` for one seeded run (e.g. re-splitting/re-fitting
        with that seed).
    n_repeats
        Number of independent seeds to average over.
    base_seed
        First seed used; subsequent repeats use ``base_seed + 1, base_seed + 2, ...``. ``None`` lets
        ``score_fn`` pick its own default seed each call (only sensible if ``score_fn`` itself randomizes).

    Returns
    -------
    np.ndarray
        ``(n_folds,)`` mean per-fold scores across the ``n_repeats`` seeded runs.
    """
    if n_repeats < 1:
        raise ValueError(f"seed_averaged_fold_scores: n_repeats must be >= 1; got {n_repeats}")
    runs = []
    for i in range(n_repeats):
        seed = None if base_seed is None else base_seed + i
        runs.append(np.asarray(score_fn(seed), dtype=np.float64))
    return np.asarray(np.mean(np.stack(runs, axis=0), axis=0), dtype=np.float64)


__all__ = ["per_fold_majority_accept", "seed_averaged_fold_scores"]
