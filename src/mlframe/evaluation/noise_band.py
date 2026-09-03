"""CV score noise-band estimation: distinguish real improvements from resampling noise.

Automated model/feature-selection loops (RFECV, MRMR, greedy forward search) compare a candidate's CV score
against the current best and accept on any improvement, however small. When the comparison is smaller than the
CV estimator's own sampling noise, the loop is chasing variance, not signal — repeated across hundreds of
candidates this measurably overfits the selection process itself to the particular fold split. This module
gives that noise floor a name: ``cv_score_equivalence_band`` estimates the standard error of a set of per-fold
(or per-seed) CV scores, and ``is_within_noise_band`` answers "are these two candidates practically equal?".
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import stats

_SQRT_2 = float(np.sqrt(2.0))


@lru_cache(maxsize=64)
def _two_sided_t(alpha: float, df: int) -> float:
    """Cached ``t_{df, 1-alpha/2}``.

    The standard error this multiplies is ESTIMATED from ``df + 1`` fold scores, not known, so the correct
    quantile is Student-t, not normal. At the usual k=5 the difference is not cosmetic: ``t_{4,0.975} = 2.776``
    against ``z_{0.975} = 1.960``, so a z-based band is 29% too narrow and covers ~86% rather than the
    documented 95%. An under-wide band ACCEPTS noise, which is the one failure this module exists to prevent.
    ``calibration/policy._heldout_ece_ci`` already made this same correction for the same reason.

    Caching is unchanged in kind: ``df`` is as stable as ``alpha`` inside a selection loop (the fold count does
    not vary), and scipy's ``ppf`` dominated ~76% of this module's wall time when called per comparison
    (profiled: 40k calls at n_folds=5 -> 10.9s of 14.3s).
    """
    return float(stats.t.ppf(1.0 - alpha / 2.0, df))


def cv_score_equivalence_band(
    fold_scores: np.ndarray,
    alpha: float = 0.05,
    method: str = "sem",
    n_comparisons: int = 1,
) -> float:
    """Estimate the "practically equal" noise band for a set of per-fold/per-seed CV scores.

    Parameters
    ----------
    fold_scores
        1D array of per-fold (or per-seed-repeat) scores for ONE candidate. At least 2 values are required;
        the band is undefined (returns ``0.0``) for a single score, since there is then no variance to estimate.
    alpha
        Two-sided miscoverage for the ``"sem"`` method (0.05 -> the band is the half-width of a 95% CI on the
        mean fold score). Ignored by ``"std"``.
    method
        ``"sem"`` (default) — ``t_{n-1, 1-alpha/2} * standard_error_of_the_mean``. This is the natural band for
        comparing two candidates' MEAN CV scores (what selection loops actually compare): a difference smaller
        than this is statistically indistinguishable from resampling noise at the given confidence level.
        ``"std"`` — the raw (ddof=1) standard deviation of the fold scores. More conservative (typically
        ``sqrt(n_folds)`` wider than SEM); use when comparing a SINGLE new fold's score against history rather
        than two multi-fold means.
    n_comparisons
        Opt-in Bonferroni-style multiple-comparisons correction. Default ``1`` reproduces the exact single-call
        behavior (``alpha`` used as given). A long automated selection loop (RFECV/MRMR greedy search) runs the
        noise-band test once per candidate; treating every single test at the nominal ``alpha`` lets the
        FAMILY-WISE false-accept rate across the whole search climb toward 1 as the candidate count grows, even
        though each individual test is correctly calibrated in isolation. Passing the number of candidate
        comparisons already run (or planned) divides the per-test ``alpha`` by ``n_comparisons`` (classic
        Bonferroni correction), widening the band so the family-wise false-accept rate across the WHOLE search
        stays bounded near the original ``alpha`` instead of accumulating. Must be a positive integer.

    Returns
    -------
    float
        The noise-band epsilon, in the same units as ``fold_scores``. Two candidates whose mean scores differ
        by less than this band should be treated as tied.
    """
    if n_comparisons < 1:
        raise ValueError(f"cv_score_equivalence_band: n_comparisons must be a positive integer; got {n_comparisons!r}")
    fold_scores = np.asarray(fold_scores, dtype=np.float64).ravel()
    n = fold_scores.shape[0]
    if n < 2:
        return 0.0
    std = float(np.std(fold_scores, ddof=1))
    if method == "std":
        return std
    if method != "sem":
        raise ValueError(f"cv_score_equivalence_band: method must be 'sem' or 'std'; got {method!r}")
    sem = std / float(np.sqrt(n))
    corrected_alpha = alpha / float(n_comparisons)
    return _two_sided_t(corrected_alpha, n - 1) * sem


def two_sample_score_band(
    fold_scores_a: np.ndarray,
    fold_scores_b: np.ndarray,
    alpha: float = 0.05,
    n_comparisons: int = 1,
) -> float:
    """Half-width of the band for the DIFFERENCE of two fold-score means, from both candidates' fold scores.

    :func:`cv_score_equivalence_band` is the band for ONE mean. A selection loop compares two of them, and the
    difference of two independent means has standard error ``sqrt(var_a/n + var_b/n)`` -- up to ``sqrt(2)`` wider
    than either one alone. Holding a difference to a one-mean band makes it too easy to clear, and clearing the
    band is what makes a candidate "actionable", so the error runs in the direction of accepting noise.

    This is the classic equal-size two-sample t band: the pooled standard error above with
    ``t_{2n-2, 1-alpha/2}``. It deliberately does NOT use the per-fold differences, which would be exact under
    pairing but collapse to a zero band whenever a candidate beats the baseline by the same amount in every
    fold -- a real pattern on quantised metrics, and one that would then make every delta actionable and zero
    out any multiplier applied to the band downstream.

    Returns ``0.0`` when fewer than 2 folds are supplied, matching :func:`cv_score_equivalence_band`.
    """
    a = np.asarray(fold_scores_a, dtype=np.float64).ravel()
    b = np.asarray(fold_scores_b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"two_sample_score_band: fold score arrays must have the same shape; got {a.shape} and {b.shape}")
    if n_comparisons < 1:
        raise ValueError(f"two_sample_score_band: n_comparisons must be a positive integer; got {n_comparisons!r}")
    n = a.shape[0]
    if n < 2:
        return 0.0
    se = float(np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / n))
    return _two_sided_t(alpha / float(n_comparisons), 2 * n - 2) * se


def is_within_noise_band(
    score_a: float,
    score_b: float,
    fold_scores: np.ndarray,
    alpha: float = 0.05,
    method: str = "sem",
    n_comparisons: int = 1,
    both_estimated: bool = True,
) -> bool:
    """``True`` when ``|score_a - score_b|`` is not distinguishable from CV resampling noise.

    ``fold_scores`` should be the per-fold scores of whichever candidate (typically the current best) is used
    to estimate the noise band — the band is a property of the CV scheme's variance, not of the specific
    comparison, so either candidate's fold scores are a reasonable proxy as long as they were produced by the
    same splitter/data/model family. ``n_comparisons`` is passed straight through to
    :func:`cv_score_equivalence_band`.

    ``both_estimated`` (default ``True``) says that ``score_a`` and ``score_b`` are BOTH means estimated from
    CV folds, which is the usual case in a selection loop. The difference of two such means has standard error
    ``sqrt(2)`` times that of either one, so the one-mean band :func:`cv_score_equivalence_band` returns is
    scaled accordingly — without it the comparison is held to a band 29% narrower than the quantity it is
    testing, and an under-wide band accepts noise. Set it to ``False`` only when ``score_b`` is a fixed
    reference rather than an estimate. When BOTH candidates' per-fold scores are available, prefer
    :func:`mlframe.evaluation.cv_delta_triage.triage_cv_delta`: it uses the per-fold differences and so is
    exact rather than assuming the two are independent.
    """
    band = cv_score_equivalence_band(fold_scores, alpha=alpha, method=method, n_comparisons=n_comparisons)
    if both_estimated:
        band *= _SQRT_2
    return bool(abs(score_a - score_b) <= band)


__all__ = ["cv_score_equivalence_band", "is_within_noise_band", "two_sample_score_band"]
