"""Decision threshold calibration: sweep the classification cutoff on a validation fold, not just probability shape.

Probability calibration (Platt/isotonic) fixes the SHAPE of predicted probabilities but still leaves the
default 0.5 decision threshold in place -- for imbalanced problems, 0.5 is rarely the operating point that
maximizes F1, or minimizes a custom cost matrix. Threshold optimization is a lightweight, often-overlooked
alternative/complement to resampling: sweep candidate thresholds on a validation fold and pick the one that
optimizes the metric that actually matters for deployment.

Two opt-in extensions on top of the single global threshold:
    - ``groups``: fit a separate threshold per cohort/segment when the optimal operating point genuinely
      differs across segments (different class balance or score distribution per segment) -- a single
      global threshold is a compromise that is suboptimal for every segment individually.
    - ``cv``: a cross-validated threshold-stability report -- how much the chosen threshold moves across
      folds. A high-variance threshold means the sweep is overfitting the threshold choice to one
      particular validation fold rather than finding a value the deployed model can rely on.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import KFold


def _best_threshold_for(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
) -> Tuple[float, float]:
    """Sweep ``thresholds`` and return ``(best_threshold, best_score)`` -- helper shared by the group/cv paths."""
    scores = np.empty(thresholds.shape[0], dtype=np.float64)
    for i, t in enumerate(thresholds):
        y_pred = (y_proba >= t).astype(np.int64)
        scores[i] = float(metric_fn(y_true, y_pred))
    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx]), float(scores[best_idx])


def _threshold_stability_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_splits: int,
    seed: int,
    stability_cv_threshold: float,
) -> Dict[str, Any]:
    """Fit the threshold independently on each of ``n_splits`` folds and report how much it moves.

    A high coefficient of variation (``std / mean`` across folds) signals the threshold choice is
    overfit to whichever fold happened to be used for the sweep, not a stable operating point.
    """
    n = y_true.shape[0]
    if n < 2 * n_splits:
        return {"fold_thresholds": np.array([]), "mean": float("nan"), "std": float("nan"), "cv": float("nan"), "is_stable": False, "n_splits": n_splits, "reason": "insufficient_samples"}

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_thresholds = np.empty(n_splits, dtype=np.float64)
    for i, (_, fold_idx) in enumerate(kfold.split(np.arange(n))):
        y_true_fold = y_true[fold_idx]
        y_proba_fold = y_proba[fold_idx]
        if len(np.unique(y_true_fold)) < 2:
            fold_thresholds[i] = float("nan")
            continue
        fold_thresholds[i] = _best_threshold_for(y_true_fold, y_proba_fold, thresholds, metric_fn)[0]

    valid = fold_thresholds[~np.isnan(fold_thresholds)]
    if valid.shape[0] == 0:
        return {"fold_thresholds": fold_thresholds, "mean": float("nan"), "std": float("nan"), "cv": float("nan"), "is_stable": False, "n_splits": n_splits, "reason": "no_valid_folds"}

    mean = float(np.mean(valid))
    std = float(np.std(valid))
    coeff_of_variation = float(std / mean) if mean != 0 else (0.0 if std < 1e-9 else float("inf"))
    is_stable = coeff_of_variation <= stability_cv_threshold
    return {
        "fold_thresholds": fold_thresholds,
        "mean": mean,
        "std": std,
        "cv": coeff_of_variation,
        "is_stable": is_stable,
        "n_splits": n_splits,
    }


def optimize_decision_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_thresholds: int = 200,
    threshold_range: Tuple[float, float] = (0.0, 1.0),
    groups: Optional[np.ndarray] = None,
    min_group_size: int = 20,
    cv: Optional[int] = None,
    cv_seed: int = 0,
    stability_cv_threshold: float = 0.15,
) -> dict:
    """Sweep candidate decision thresholds and return the one maximizing ``metric_fn`` on binary predictions.

    Parameters
    ----------
    y_true
        ``(n,)`` binary ground truth.
    y_proba
        ``(n,)`` predicted probabilities.
    metric_fn
        ``metric_fn(y_true, y_pred_binary) -> float``, HIGHER is better (e.g. F1, a custom cost-weighted
        score). For a cost MATRIX, wrap it into a callable returning ``-total_cost`` so higher is still better.
    n_thresholds
        Number of candidate thresholds swept, evenly spaced over ``threshold_range``.
    threshold_range
        ``(low, high)`` bounds for the sweep.
    groups
        Opt-in per-segment thresholding. When given, ``(n,)`` cohort/group labels aligned with ``y_true``.
        In addition to the global result, a per-group threshold is fit on each segment's own subset via
        ``_best_threshold_for``; segments smaller than ``min_group_size`` fall back to the global threshold
        (too few samples for a reliable segment-local sweep). Adds ``group_thresholds`` and ``group_results``
        to the return dict. Does not change the top-level ``best_threshold``/``best_score`` fields.
    min_group_size
        Minimum samples a group needs to get its own fitted threshold (see ``groups``).
    cv
        Opt-in cross-validated threshold-stability report. When given (``>= 2``), the threshold is
        refit independently on ``cv`` random folds of ``(y_true, y_proba)`` and the fold-to-fold spread is
        reported under ``cv_report`` (and per group under ``cv_report["per_group"]`` when ``groups`` is
        also given). Does not change the top-level ``best_threshold``/``best_score`` fields.
    cv_seed
        Random seed for the fold split used by ``cv``.
    stability_cv_threshold
        Coefficient-of-variation cutoff (``std / mean`` across folds) below which a threshold is
        reported ``is_stable`` in the ``cv`` report.

    Returns
    -------
    dict
        ``best_threshold``, ``best_score``, ``thresholds`` ``(n_thresholds,)``, ``scores`` ``(n_thresholds,)``
        (the full sweep, for inspection/plotting) -- unchanged when ``groups``/``cv`` are omitted. Plus,
        opt-in: ``group_thresholds``, ``group_results`` (when ``groups`` given) and ``cv_report`` (when
        ``cv`` given).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)

    scores = np.empty(n_thresholds, dtype=np.float64)
    for i, t in enumerate(thresholds):
        y_pred = (y_proba >= t).astype(np.int64)
        scores[i] = float(metric_fn(y_true, y_pred))

    best_idx = int(np.argmax(scores))
    result: Dict[str, Any] = {
        "best_threshold": float(thresholds[best_idx]),
        "best_score": float(scores[best_idx]),
        "thresholds": thresholds,
        "scores": scores,
    }

    if groups is not None:
        groups_arr = np.asarray(groups)
        group_thresholds: Dict[Any, float] = {}
        group_results: Dict[Any, Dict[str, float]] = {}
        for g in np.unique(groups_arr):
            mask = groups_arr == g
            if int(mask.sum()) >= min_group_size:
                g_threshold, g_score = _best_threshold_for(y_true[mask], y_proba[mask], thresholds, metric_fn)
                group_results[g] = {"best_threshold": g_threshold, "best_score": g_score, "n_samples": int(mask.sum())}
                group_thresholds[g] = g_threshold
            else:
                group_thresholds[g] = result["best_threshold"]  # too few samples: fall back to the global threshold
        result["group_thresholds"] = group_thresholds
        result["group_results"] = group_results

    if cv is not None:
        cv_report = _threshold_stability_report(y_true, y_proba, thresholds, metric_fn, n_splits=cv, seed=cv_seed, stability_cv_threshold=stability_cv_threshold)
        if groups is not None:
            groups_arr = np.asarray(groups)
            per_group_cv: Dict[Any, Dict[str, Any]] = {}
            for g in np.unique(groups_arr):
                mask = groups_arr == g
                if int(mask.sum()) >= max(min_group_size, 2 * cv):
                    per_group_cv[g] = _threshold_stability_report(
                        y_true[mask], y_proba[mask], thresholds, metric_fn, n_splits=cv, seed=cv_seed, stability_cv_threshold=stability_cv_threshold
                    )
                else:
                    per_group_cv[g] = {"reason": "insufficient_samples", "is_stable": False}
            cv_report["per_group"] = per_group_cv
        result["cv_report"] = cv_report

    return result


def apply_decision_threshold(
    y_proba: np.ndarray,
    threshold: float,
    groups: Optional[np.ndarray] = None,
    group_thresholds: Optional[Dict[Any, float]] = None,
) -> np.ndarray:
    """Binarize ``y_proba`` at ``threshold`` (``>= threshold`` -> 1).

    Opt-in per-segment mode: when both ``groups`` (``(n,)`` cohort labels aligned with ``y_proba``) and
    ``group_thresholds`` (as returned by ``optimize_decision_threshold(..., groups=...)``) are given, each
    sample is binarized at its own group's threshold; a group missing from ``group_thresholds`` falls back
    to ``threshold``. Omitting ``groups``/``group_thresholds`` reproduces the original single-threshold
    behavior exactly.
    """
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if groups is None or group_thresholds is None:
        return np.asarray((y_proba >= threshold).astype(np.int64))

    groups_arr = np.asarray(groups)
    preds = np.empty(y_proba.shape[0], dtype=np.int64)
    for g in np.unique(groups_arr):
        mask = groups_arr == g
        t = group_thresholds.get(g, threshold)
        preds[mask] = (y_proba[mask] >= t).astype(np.int64)
    return preds


__all__ = ["optimize_decision_threshold", "apply_decision_threshold"]
