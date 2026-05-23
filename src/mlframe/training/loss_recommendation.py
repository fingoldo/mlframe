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


_EXCESS_KURT_HEAVY: float = 3.0
"""Threshold above which the residual is genuinely Laplace-or-heavier
and L1/MAE is the right MLE.

Default raised 1.5 -> 3.0 (2026-05-23 TVT-rerun audit): on production TVT
composite residuals with ``excess_kurt=6.37`` the old 1.5 threshold
triggered MAE objective, and the MAE gradient on a target where 99% of
mass is near zero is just ``sign(noise)`` -- constant-magnitude random
signal. CatBoost early-stopped at iter=1 on TVT-addres-TVT_prev;
LightGBM at iter=5 on TVT-diff-TVT_prev. The Huber branch (excess_kurt>3.0,
new default) handles the ``excess_kurt in (1.5, 3.0)`` range with
bounded-influence loss that retains a useful gradient on small residuals
AND remains robust on tails. Above 3.0 the distribution is sharp enough
that pure L1 is genuinely appropriate."""

_EXCESS_KURT_EXTREME: float = 10.0
"""Threshold matching ``regression_residual_audit.EXCESS_KURT_EXTREME``. Above this the residual is contaminated (Student-t / mixture); Huber's bounded influence function is the safe default."""

_EXCESS_KURT_MEDIUM: float = 1.5
"""Lower-bound for the Huber band. ``excess_kurt in (_EXCESS_KURT_MEDIUM,
_EXCESS_KURT_HEAVY)`` triggers Huber loss (was the old MAE band).
1.5 retains the original threshold for the FIRST departure from Gaussian
tolerance; Huber's gradient stays useful on small residuals AND attenuates
outlier influence, so it dominates plain L1 in the medium-kurt regime."""

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
    c2 = centered * centered
    var = float(c2.mean())
    std = float(np.sqrt(var))
    if var <= 0.0:
        return n, mean, 0.0, 0.0
    # Population excess kurtosis: E[(x-mu)^4] / sigma^4 - 3.
    # c2 * c2 over centered ** 4 avoids np.power's general dispatch
    # (~3x faster at n=50k..5M; same antipattern as iter138 fixed in
    # _target_distribution_analyzer._excess_kurtosis).
    m4 = float((c2 * c2).mean())
    excess_kurt = m4 / (var * var) - 3.0
    return n, mean, std, excess_kurt


def recommend_boosting_regression_loss(
    y_target: Any,
    *,
    target_quantile: float | None = None,
) -> Dict[str, Any]:
    """Recommend per-backend regression loss based on target tail shape.

    Parameters
    ----------
    y_target : 1-D array-like
        The target distribution the inner boosting will FIT (raw ``y`` for raw-target, residual ``T`` for composite-target). NaN / Inf values are ignored.
    target_quantile : float in (0, 1), optional
        When supplied, OVERRIDES the kurt-based heuristic and returns a
        quantile loss configured at this alpha for all three backends.
        Use for asymmetric-cost regression (e.g. user prefers
        under-prediction over over-prediction): ``alpha=0.7`` puts 70%
        of the penalty weight on over-prediction.

    Returns
    -------
    dict
        Keys: ``cb`` (CatBoost ``loss_function``), ``lgb`` (LightGBM ``objective``), ``xgb`` (XGBoost ``objective``), ``rationale``, ``excess_kurt``, ``n_finite``. When ``target_quantile`` is set, also includes ``quantile_alpha`` and the per-backend value strings carry the alpha (e.g. ``Quantile:alpha=0.7``).
    """
    n_finite, _, _, excess_kurt = _safe_moments(np.asarray(y_target))

    # Quantile-loss override: when caller specifies an asymmetric cost,
    # ignore the kurt-based heuristic. Quantile loss IS robust to heavy
    # tails (it's a quantile estimator, not a mean estimator) so the
    # heavy-tail switch would be redundant.
    if target_quantile is not None:
        alpha = float(target_quantile)
        if not (0.0 < alpha < 1.0):
            raise ValueError(
                f"target_quantile must be in (0, 1); got {alpha}"
            )
        return {
            "cb": f"Quantile:alpha={alpha:.3f}",
            "lgb": "quantile",
            "lgb_extra_params": {"alpha": alpha},
            # XGBoost >=2.0 introduces ``reg:quantileerror`` with ``quantile_alpha`` kwarg.
            "xgb": "reg:quantileerror",
            "xgb_extra_params": {"quantile_alpha": alpha},
            "quantile_alpha": alpha,
            "rationale": (
                f"target_quantile={alpha:.3f} -- asymmetric-cost regression, "
                f"using quantile loss with alpha={alpha:.3f}."
            ),
            "excess_kurt": excess_kurt,
            "n_finite": n_finite,
        }

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
    if excess_kurt > _EXCESS_KURT_MEDIUM:
        # Mild leptokurtic (Laplace-LIKE but not pure Laplace): Huber
        # retains useful gradient on small residuals AND attenuates tail
        # influence. Avoids the MAE-gradient-is-noise pathology
        # (production TVT 2026-05-23: composite residuals with kurt~6
        # got pure MAE objective and boosters stopped at iter=1-5).
        return {
            "cb": "Huber:delta=1.345",
            "lgb": "huber",
            "xgb": "reg:pseudohubererror",
            "rationale": (
                f"excess_kurt={excess_kurt:.2f} in "
                f"({_EXCESS_KURT_MEDIUM}, {_EXCESS_KURT_HEAVY}] -- "
                "mildly leptokurtic; Huber bounded-influence loss keeps "
                "the gradient informative on small residuals while "
                "attenuating outlier influence."
            ),
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
