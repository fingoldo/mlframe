"""Decision threshold calibration: sweep the classification cutoff on a validation fold, not just probability shape.

Probability calibration (Platt/isotonic) fixes the SHAPE of predicted probabilities but still leaves the
default 0.5 decision threshold in place -- for imbalanced problems, 0.5 is rarely the operating point that
maximizes F1, or minimizes a custom cost matrix. Threshold optimization is a lightweight, often-overlooked
alternative/complement to resampling: sweep candidate thresholds on a validation fold and pick the one that
optimizes the metric that actually matters for deployment.
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def optimize_decision_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_thresholds: int = 200,
    threshold_range: Tuple[float, float] = (0.0, 1.0),
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

    Returns
    -------
    dict
        ``best_threshold``, ``best_score``, ``thresholds`` ``(n_thresholds,)``, ``scores`` ``(n_thresholds,)``
        (the full sweep, for inspection/plotting).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=np.float64)
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)

    scores = np.empty(n_thresholds, dtype=np.float64)
    for i, t in enumerate(thresholds):
        y_pred = (y_proba >= t).astype(np.int64)
        scores[i] = float(metric_fn(y_true, y_pred))

    best_idx = int(np.argmax(scores))
    return {
        "best_threshold": float(thresholds[best_idx]),
        "best_score": float(scores[best_idx]),
        "thresholds": thresholds,
        "scores": scores,
    }


def apply_decision_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """Binarize ``y_proba`` at ``threshold`` (``>= threshold`` -> 1)."""
    return np.asarray((np.asarray(y_proba, dtype=np.float64) >= threshold).astype(np.int64))


__all__ = ["optimize_decision_threshold", "apply_decision_threshold"]
