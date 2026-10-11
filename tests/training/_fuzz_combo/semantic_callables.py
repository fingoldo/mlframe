"""Reference callables for the callable-typed config fields the fuzz harness varies.

Each one matches the signature its production call site actually uses, and lives at module scope so it pickles by reference (the ensemble
scorer ships ``custom_ice_metric`` / ``custom_rice_metric`` to loky workers, and fitted models carrying a ``metamodel_func`` wrapper get dumped).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import numpy as np

__all__ = [
    "passthrough_metamodel",
    "arcsinh_target_metamodel",
    "brier_integral_error",
    "binned_ece_integral_error",
]


def passthrough_metamodel(model: Any) -> Any:
    """``metamodel_func`` that returns the estimator unchanged, so the wrapping call path runs without altering the model."""
    return model


def arcsinh_target_metamodel(model: Any) -> Any:
    """``metamodel_func`` wrapping regressors in a variance-stabilising target transform; classifiers pass through.

    ``arcsinh`` (not ``log1p``) because it is defined on every real target, including the negative ones the fuzz frames produce.
    """
    from sklearn.base import is_regressor
    from sklearn.compose import TransformedTargetRegressor

    if not is_regressor(model):
        return model
    return TransformedTargetRegressor(regressor=model, func=np.arcsinh, inverse_func=np.sinh, check_inverse=False)


def _as_indicator_matrix(y_true: Any, y_score: np.ndarray) -> np.ndarray:
    """``y_true`` as a ``(N, K)`` 0/1 matrix aligned with the ``(N, K)`` score matrix: a 2-D label matrix as is, 1-D labels one-hot encoded."""
    y = np.asarray(y_true)
    if y.ndim == 2:
        return y.astype(np.float64)
    k = y_score.shape[1]
    if np.issubdtype(y.dtype, np.integer) and y.size and y.min() >= 0 and y.max() < k:
        codes = y.astype(np.int64)
    else:
        classes = np.unique(y)
        codes = np.searchsorted(classes, y)
    out = np.zeros((y.shape[0], k), dtype=np.float64)
    valid = codes < k
    out[np.flatnonzero(valid), codes[valid]] = 1.0
    return out


def _per_column(y_true: Any, y_score: Any, column_error) -> Tuple[float, Dict[int, float]]:
    """Mean of ``column_error`` over the score columns, plus the per-column values keyed by column index."""
    s = np.asarray(y_score, dtype=np.float64)
    if s.ndim == 1:
        err = float(column_error(np.asarray(y_true).astype(np.float64).ravel(), s))
        return err, {0: err}
    y = _as_indicator_matrix(y_true, s)
    per = {j: float(column_error(y[:, j], s[:, j])) for j in range(s.shape[1])}
    return float(np.mean(list(per.values()))), per


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared gap between a 0/1 label column and its probability column."""
    return float(np.mean((p - y) ** 2)) if y.size else 0.0


def _binned_ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over ``n_bins`` equal-width probability bins."""
    if not y.size:
        return 0.0
    bins = np.clip((p * n_bins).astype(np.int64), 0, n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins).astype(np.float64)
    gap = np.abs(np.bincount(bins, weights=p, minlength=n_bins) - np.bincount(bins, weights=y, minlength=n_bins))
    return float(gap.sum() / counts.sum())


def brier_integral_error(y_true: Any, y_score: Any, return_per_class: bool = False) -> Union[float, Tuple[float, Dict[int, float]]]:
    """``custom_ice_metric`` / ``custom_rice_metric``: class-averaged Brier score; ``return_per_class`` exercises the batched per-class contract."""
    total, per = _per_column(y_true, y_score, _brier)
    return (total, per) if return_per_class else total


def binned_ece_integral_error(y_true: Any, y_score: Any) -> float:
    """``custom_ice_metric`` / ``custom_rice_metric`` without ``return_per_class``, so the reporter's ``TypeError`` fallback path runs."""
    return _per_column(y_true, y_score, _binned_ece)[0]
