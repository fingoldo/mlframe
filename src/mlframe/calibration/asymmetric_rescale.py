"""``fit_asymmetric_rescale``/``apply_asymmetric_rescale``: sign-conditional scalar rescaling.

Source: 8th_ubiquant-market-prediction.md -- "multiplying all negative predictions by 1.4 and dividing all
positive predictions by 1.4... going through a bunch of different values from 1 to 2 and seeing how that
affected the CV." Applicable when a prediction's SIGN carries most of the decision-relevant information but
its magnitude is systematically miscalibrated in an asymmetric way between the positive and negative regimes
(e.g. trading signals scored by a sign-weighted metric).

Caveat (documented explicitly, per this idea's own known risk): a single scalar tuned by a 1-D CV sweep is
genuinely prone to overfitting the validation fold, especially on noisy targets -- use a metric-appropriate
CV scheme (multiple folds, not a single holdout) and treat a large factor (far from 1.0) as a red flag for
validation-set noise rather than a real, generalizable correction.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


def fit_asymmetric_rescale(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    factor_range: Tuple[float, float] = (1.0, 2.0),
    n_factors: int = 50,
) -> Dict[str, float]:
    """Grid-search a scalar ``factor`` maximizing ``metric_fn`` after ``negative *= factor, positive /= factor``.

    Parameters
    ----------
    y_true, y_pred
        ``(n,)`` validation-slice ground truth and predictions.
    metric_fn
        ``metric_fn(y_true, y_pred) -> float``, HIGHER is better.
    factor_range
        ``(low, high)`` sweep bounds for the rescale factor (``1.0`` = no-op).
    n_factors
        Number of grid points.

    Returns
    -------
    dict
        ``{"factor": best_factor, "metric": best_metric_value}``.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    best_factor = 1.0
    best_metric = float(metric_fn(y_true_arr, y_pred_arr))
    for factor in np.linspace(factor_range[0], factor_range[1], n_factors):
        rescaled = apply_asymmetric_rescale(y_pred_arr, float(factor))
        metric_value = float(metric_fn(y_true_arr, rescaled))
        if metric_value > best_metric:
            best_metric = metric_value
            best_factor = float(factor)

    return {"factor": best_factor, "metric": best_metric}


def apply_asymmetric_rescale(y_pred: np.ndarray, factor: float) -> np.ndarray:
    """``negative predictions *= factor``, ``positive predictions /= factor`` (``factor=1.0`` is a no-op)."""
    pred_arr = np.asarray(y_pred, dtype=np.float64)
    return np.asarray(np.where(pred_arr < 0, pred_arr * factor, pred_arr / factor))


__all__ = ["fit_asymmetric_rescale", "apply_asymmetric_rescale"]
