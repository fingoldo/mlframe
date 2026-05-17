"""Auto-pick a robust regression loss for heavy-tailed targets.

Pack H. Production failure mode (2026-05-17 TVT-linres-TVT_prev): the composite residual had ``excess_kurt = +2.40`` (Laplace-like) but the inner CatBoost / XGBoost / LightGBM were trained with the default RMSE objective. RMSE on Laplace-distributed residuals collapses the gradient near zero (outliers dominate), the model gives up at iter=4-10, and the composite gain over raw is left on the table.

The fix is policy: when the target distribution moments indicate Laplace / Student-t / contaminated tails, switch the inner boosting objective to L1 (Laplace MLE) or Huber. This module exposes the policy as a pure function that takes a 1-D target array and returns a per-backend recommendation:

    {"cb": "MAE", "lgb": "regression_l1", "xgb": "reg:absoluteerror",
     "rationale": "...", "skew": ..., "excess_kurt": ...}

Wiring: callers thread the returned per-backend loss name into the model factory's ``common_params`` / ``loss_function`` / ``objective`` argument BEFORE building the estimator. The recommended call site is ``_phase_train_one_target`` just after the per-target ``y_train`` is materialised; the recommendation is a one-shot O(n) compute on the target array. Composite-target paths feed the residual T directly (``y_T = transform.forward(y, base)``) so the loss matches the actual distribution the inner sees.

Thresholds. The cutoffs come from the empirical excess-kurtosis ladder verified in ``regression_residual_audit.py``:
- ``excess_kurt < 1.5`` -> near-Gaussian, keep RMSE (default).
- ``1.5 <= excess_kurt <= 10`` -> Laplace / Student-t. MAE / L1 (Laplace MLE).
- ``excess_kurt > 10`` -> contaminated outliers. Huber where supported (LightGBM ``huber``, CatBoost ``Huber``); XGBoost falls back to ``reg:pseudohubererror``.

If the target is too small to estimate moments reliably (``n_finite < 30``) the recommendation is conservative (RMSE, the standard default).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np


_EXCESS_KURT_HEAVY: float = 1.5
"""Threshold matching ``regression_residual_audit.EXCESS_KURT_HEAVY``. Above this the residual is Laplace-or-heavier; RMSE under-fits the median."""

_EXCESS_KURT_EXTREME: float = 10.0
"""Threshold matching ``regression_residual_audit.EXCESS_KURT_EXTREME``. Above this the residual is contaminated (Student-t / mixture); Huber's bounded influence function is the safe default."""

_MIN_SAMPLE_N: int = 30
"""Below this we don't have enough mass to estimate kurtosis reliably; recommendation falls back to the standard default."""


def _safe_moments(y: np.ndarray) -> tuple[int, float, float, float]:
    """Return ``(n_finite, mean, std, excess_kurt)`` from finite values of ``y``.

    Uses the population definition of kurtosis - 3 so a Normal sample yields ~0; matches ``scipy.stats.kurtosis(..., fisher=True)`` to four decimal places without paying the import. Returns ``excess_kurt=0.0`` on a constant target (var=0) and ``(0, nan, nan, nan)`` when no finite values remain.
    """
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    n = int(finite.size)
    if n == 0:
        return 0, float("nan"), float("nan"), float("nan")
    mean = float(finite.mean())
    if n < 2:
        return n, mean, 0.0, 0.0
    centered = finite - mean
    var = float((centered * centered).mean())
    std = float(np.sqrt(var))
    if var <= 0.0:
        return n, mean, 0.0, 0.0
    # Population excess kurtosis: E[(x-mu)^4] / sigma^4 - 3.
    m4 = float((centered ** 4).mean())
    excess_kurt = m4 / (var * var) - 3.0
    return n, mean, std, excess_kurt


def recommend_boosting_regression_loss(y_target: Any) -> Dict[str, Any]:
    """Recommend per-backend regression loss based on target tail shape.

    Parameters
    ----------
    y_target : 1-D array-like
        The target distribution the inner boosting will FIT (raw ``y`` for raw-target, residual ``T`` for composite-target). NaN / Inf values are ignored.

    Returns
    -------
    dict
        Keys: ``cb`` (CatBoost ``loss_function``), ``lgb`` (LightGBM ``objective``), ``xgb`` (XGBoost ``objective``), ``rationale``, ``excess_kurt``, ``n_finite``. Use ``cb``/``lgb``/``xgb`` to override the corresponding model param; ``rationale`` is a one-line human-readable explanation for logging.
    """
    n_finite, _, _, excess_kurt = _safe_moments(np.asarray(y_target))
    if n_finite < _MIN_SAMPLE_N or not np.isfinite(excess_kurt):
        return {
            "cb": "RMSE",
            "lgb": "regression",
            "xgb": "reg:squarederror",
            "rationale": f"n_finite={n_finite} < {_MIN_SAMPLE_N} -- insufficient sample for kurtosis estimate; RMSE default.",
            "excess_kurt": excess_kurt,
            "n_finite": n_finite,
        }
    if excess_kurt > _EXCESS_KURT_EXTREME:
        return {
            "cb": "Huber:delta=1.345",
            "lgb": "huber",
            "xgb": "reg:pseudohubererror",
            "rationale": f"excess_kurt={excess_kurt:.2f} > {_EXCESS_KURT_EXTREME} -- contaminated tails; Huber bounded-influence loss.",
            "excess_kurt": excess_kurt,
            "n_finite": n_finite,
        }
    if excess_kurt > _EXCESS_KURT_HEAVY:
        return {
            "cb": "MAE",
            "lgb": "regression_l1",
            "xgb": "reg:absoluteerror",
            "rationale": f"excess_kurt={excess_kurt:.2f} > {_EXCESS_KURT_HEAVY} -- Laplace/Student-t tails; L1 (MAE) is the Laplace MLE.",
            "excess_kurt": excess_kurt,
            "n_finite": n_finite,
        }
    return {
        "cb": "RMSE",
        "lgb": "regression",
        "xgb": "reg:squarederror",
        "rationale": f"excess_kurt={excess_kurt:.2f} within Gaussian tolerance; RMSE default.",
        "excess_kurt": excess_kurt,
        "n_finite": n_finite,
    }


__all__ = [
    "recommend_boosting_regression_loss",
]
