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

from typing import Any, Callable, Dict, List, Tuple

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


def cross_validate_asymmetric_rescale(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_folds: int = 5,
    factor_range: Tuple[float, float] = (1.0, 2.0),
    n_factors: int = 50,
    instability_cv_threshold: float = 0.15,
    seed: int = 0,
) -> Dict[str, Any]:
    """K-fold check that a single-holdout ``fit_asymmetric_rescale`` factor isn't overfit noise.

    Addresses this module's own documented risk (a 1-D grid-searched scalar fit on one validation slice is
    prone to overfitting, especially on noisy targets): fits a factor on each fold's train split, applies it to
    that fold's held-out split, and reports the fold-to-fold variance of the fitted factors. A genuine,
    generalizable asymmetric miscalibration produces similar factors across folds (low coefficient of
    variation); pure validation-set noise produces wildly different factors per fold (high CV) -- a red flag
    that applying the globally-fit factor to new data is risky.

    Parameters
    ----------
    y_true, y_pred
        ``(n,)`` ground truth and predictions to cross-validate over.
    metric_fn
        ``metric_fn(y_true, y_pred) -> float``, HIGHER is better.
    n_folds
        Number of CV folds (``>= 2``).
    factor_range, n_factors
        Forwarded to ``fit_asymmetric_rescale`` on each fold's training split.
    instability_cv_threshold
        Fold-factor coefficient of variation (``std / mean``) above which the fit is flagged unstable.
    seed
        Seed for the fold-shuffling RNG (deterministic fold assignment).

    Returns
    -------
    dict
        ``factor`` (mean across folds, for downstream use), ``fold_factors`` (per-fold list),
        ``factor_std``, ``factor_cv``, ``is_stable`` (bool, ``factor_cv < instability_cv_threshold``),
        ``fold_metrics`` (per-fold held-out metric under that fold's factor), ``mean_fold_metric``.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    rng = np.random.default_rng(seed)
    shuffled_idx = rng.permutation(len(y_true_arr))
    folds = np.array_split(shuffled_idx, n_folds)

    fold_factors: List[float] = []
    fold_metrics: List[float] = []
    for fold_i in range(n_folds):
        test_idx = folds[fold_i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != fold_i])
        fold_fit = fit_asymmetric_rescale(
            y_true_arr[train_idx], y_pred_arr[train_idx], metric_fn, factor_range=factor_range, n_factors=n_factors
        )
        fold_factors.append(fold_fit["factor"])
        rescaled_test = apply_asymmetric_rescale(y_pred_arr[test_idx], fold_fit["factor"])
        fold_metrics.append(float(metric_fn(y_true_arr[test_idx], rescaled_test)))

    fold_factors_arr = np.asarray(fold_factors, dtype=np.float64)
    factor_mean = float(fold_factors_arr.mean())
    factor_std = float(fold_factors_arr.std())
    factor_cv = float(factor_std / factor_mean) if factor_mean != 0.0 else float("inf")

    return {
        "factor": factor_mean,
        "fold_factors": fold_factors,
        "factor_std": factor_std,
        "factor_cv": factor_cv,
        "is_stable": factor_cv < instability_cv_threshold,
        "fold_metrics": fold_metrics,
        "mean_fold_metric": float(np.mean(fold_metrics)),
    }


__all__ = ["fit_asymmetric_rescale", "apply_asymmetric_rescale", "cross_validate_asymmetric_rescale"]
