"""CV-delta triage: is a candidate's CV improvement actually LB/OOS-actionable?

Home-Credit-style writeup finding: CV deltas below ~0.001-0.005 correlated poorly with leaderboard/OOS
outcome, deltas above ~0.01 correlated well, and improvements sourced from feature engineering were far more
trustworthy than improvements from hyperparameter tuning at the SAME CV delta magnitude. ``cv_score_equivalence_band``
(:mod:`mlframe.evaluation.noise_band`) already answers "is this delta bigger than resampling noise?" — this
module adds the second axis the writeup flags as separately predictive: WHERE the delta came from.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from mlframe.evaluation.noise_band import _two_sided_z, cv_score_equivalence_band

ChangeSource = Literal["feature_engineering", "hyperparameter"]


class CVDeltaHistory:
    """Opt-in rolling accumulator of CV-noise evidence across many ``triage_cv_delta`` calls.

    A single-call ``cv_score_equivalence_band`` estimates fold-score variance from whatever ``n_folds`` (often
    3-10) that one call happens to see -- a small-sample variance estimate that itself has high sampling noise.
    Across many experiment comparisons on the same project/metric/CV scheme, the true noise scale is shared, so
    pooling per-call sample variances (classic pooled-variance / ANOVA-style estimator, weighted by degrees of
    freedom) converges to a far more accurate noise-scale estimate than any single call's. Callers persist one
    instance per project and pass it into successive ``triage_cv_delta`` calls; the accumulator has no effect
    unless a caller explicitly creates and passes one in.
    """

    def __init__(self) -> None:
        self._pooled_ss: float = 0.0
        self._pooled_dof: int = 0
        self.n_updates: int = 0

    def update(self, fold_scores: np.ndarray) -> None:
        """Fold a new set of per-fold scores into the pooled variance estimate."""
        fold_scores = np.asarray(fold_scores, dtype=np.float64).ravel()
        n = fold_scores.shape[0]
        if n < 2:
            return
        var = float(np.var(fold_scores, ddof=1))
        self._pooled_ss += (n - 1) * var
        self._pooled_dof += n - 1
        self.n_updates += 1

    @property
    def pooled_dof(self) -> int:
        return self._pooled_dof

    @property
    def pooled_std(self) -> Optional[float]:
        """Pooled fold-score standard deviation across all accumulated history, or ``None`` before any update."""
        if self._pooled_dof == 0:
            return None
        return float(np.sqrt(self._pooled_ss / self._pooled_dof))

    def pooled_band(self, n_folds: int, alpha: float = 0.05) -> Optional[float]:
        """Noise band for a comparison with ``n_folds`` folds, using the pooled (cross-history) std, or
        ``None`` before any update.
        """
        std = self.pooled_std
        if std is None:
            return None
        sem = std / float(np.sqrt(n_folds))
        return _two_sided_z(alpha) * sem


def triage_cv_delta(
    baseline_fold_scores: np.ndarray,
    candidate_fold_scores: np.ndarray,
    change_source: ChangeSource,
    alpha: float = 0.05,
    hyperparameter_band_multiplier: float = 2.0,
    history: Optional[CVDeltaHistory] = None,
    min_history_dof: int = 20,
) -> dict:
    """Classify a candidate's CV improvement as actionable, noise, or (for hyperparameter deltas) suspect.

    Parameters
    ----------
    baseline_fold_scores, candidate_fold_scores
        Per-fold scores (higher assumed better) for the current-best and the candidate respectively; equal
        length, paired by fold.
    change_source
        ``"feature_engineering"`` deltas are trusted at the plain noise band; ``"hyperparameter"`` deltas are
        held to a stricter (``hyperparameter_band_multiplier``x wider) band, since the source writeup found
        hyperparameter-driven CV gains far less likely to generalize to LB/OOS than FE-driven gains of the
        same nominal size.
    alpha
        Passed through to :func:`cv_score_equivalence_band` (and to :meth:`CVDeltaHistory.pooled_band`).
    hyperparameter_band_multiplier
        How much wider the noise band must be cleared for a hyperparameter-sourced delta to be trusted.
    history
        Optional :class:`CVDeltaHistory` accumulator. When omitted (default), behavior is unchanged: the band
        is derived solely from this call's ``baseline_fold_scores``, as before. When supplied, ``baseline_fold_scores``
        is folded into the accumulator's pooled-variance estimate, and once the accumulator has seen at least
        ``min_history_dof`` pooled degrees of freedom the (tighter, cross-experiment) pooled band replaces the
        single-call band -- the project's own noise-band threshold auto-calibrates as more comparisons accrue.
    min_history_dof
        Minimum pooled degrees of freedom (summed ``n_folds - 1`` across accumulated calls) before the pooled
        band from ``history`` is trusted over the single-call band. Ignored when ``history`` is ``None``.

    Returns
    -------
    dict
        ``delta`` (candidate mean - baseline mean), ``band`` (the band actually applied), ``actionable`` (bool),
        ``reason`` (short human-readable classification), ``band_source`` (``"single_call"`` or ``"history"``).
    """
    baseline_fold_scores = np.asarray(baseline_fold_scores, dtype=np.float64).ravel()
    candidate_fold_scores = np.asarray(candidate_fold_scores, dtype=np.float64).ravel()
    if baseline_fold_scores.shape != candidate_fold_scores.shape:
        raise ValueError("triage_cv_delta: baseline_fold_scores and candidate_fold_scores must have the same shape")
    if change_source not in ("feature_engineering", "hyperparameter"):
        raise ValueError(f"triage_cv_delta: change_source must be 'feature_engineering' or 'hyperparameter'; got {change_source!r}")

    delta = float(np.mean(candidate_fold_scores) - np.mean(baseline_fold_scores))

    band_source = "single_call"
    pooled_band: Optional[float] = None
    if history is not None:
        history.update(baseline_fold_scores)
        if history.pooled_dof >= min_history_dof:
            pooled_band = history.pooled_band(n_folds=baseline_fold_scores.shape[0], alpha=alpha)
    if pooled_band is not None:
        band = pooled_band
        band_source = "history"
    else:
        band = cv_score_equivalence_band(baseline_fold_scores, alpha=alpha, method="sem")

    if change_source == "hyperparameter":
        band *= hyperparameter_band_multiplier

    if abs(delta) <= band:
        reason = f"|delta|={abs(delta):.5f} within {change_source} noise band ({band:.5f}, {band_source}) -- not LB/OOS-actionable"
        actionable = False
    else:
        reason = f"|delta|={abs(delta):.5f} exceeds {change_source} noise band ({band:.5f}, {band_source}) -- likely LB/OOS-actionable"
        actionable = True

    return {"delta": delta, "band": band, "actionable": actionable, "reason": reason, "band_source": band_source}


__all__ = ["triage_cv_delta", "CVDeltaHistory"]
