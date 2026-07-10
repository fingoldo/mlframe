"""CV-delta triage: is a candidate's CV improvement actually LB/OOS-actionable?

Home-Credit-style writeup finding: CV deltas below ~0.001-0.005 correlated poorly with leaderboard/OOS
outcome, deltas above ~0.01 correlated well, and improvements sourced from feature engineering were far more
trustworthy than improvements from hyperparameter tuning at the SAME CV delta magnitude. ``cv_score_equivalence_band``
(:mod:`mlframe.evaluation.noise_band`) already answers "is this delta bigger than resampling noise?" — this
module adds the second axis the writeup flags as separately predictive: WHERE the delta came from.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from mlframe.evaluation.noise_band import cv_score_equivalence_band

ChangeSource = Literal["feature_engineering", "hyperparameter"]


def triage_cv_delta(
    baseline_fold_scores: np.ndarray,
    candidate_fold_scores: np.ndarray,
    change_source: ChangeSource,
    alpha: float = 0.05,
    hyperparameter_band_multiplier: float = 2.0,
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
        Passed through to :func:`cv_score_equivalence_band`.
    hyperparameter_band_multiplier
        How much wider the noise band must be cleared for a hyperparameter-sourced delta to be trusted.

    Returns
    -------
    dict
        ``delta`` (candidate mean - baseline mean), ``band`` (the band actually applied), ``actionable`` (bool),
        ``reason`` (short human-readable classification).
    """
    baseline_fold_scores = np.asarray(baseline_fold_scores, dtype=np.float64).ravel()
    candidate_fold_scores = np.asarray(candidate_fold_scores, dtype=np.float64).ravel()
    if baseline_fold_scores.shape != candidate_fold_scores.shape:
        raise ValueError("triage_cv_delta: baseline_fold_scores and candidate_fold_scores must have the same shape")

    delta = float(np.mean(candidate_fold_scores) - np.mean(baseline_fold_scores))
    band = cv_score_equivalence_band(baseline_fold_scores, alpha=alpha, method="sem")
    if change_source == "hyperparameter":
        band *= hyperparameter_band_multiplier
    elif change_source != "feature_engineering":
        raise ValueError(f"triage_cv_delta: change_source must be 'feature_engineering' or 'hyperparameter'; got {change_source!r}")

    if abs(delta) <= band:
        reason = f"|delta|={abs(delta):.5f} within {change_source} noise band ({band:.5f}) -- not LB/OOS-actionable"
        actionable = False
    else:
        reason = f"|delta|={abs(delta):.5f} exceeds {change_source} noise band ({band:.5f}) -- likely LB/OOS-actionable"
        actionable = True

    return {"delta": delta, "band": band, "actionable": actionable, "reason": reason}


__all__ = ["triage_cv_delta"]
