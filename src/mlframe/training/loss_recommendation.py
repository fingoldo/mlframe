"""Auto-pick a robust regression loss for heavy-tailed targets.

Pack H. Production failure mode: a composite residual (e.g. ``y-linres-lag1``) had ``excess_kurt = +2.40`` (Laplace-like) but the inner CatBoost / XGBoost / LightGBM were trained with the default RMSE objective. RMSE on Laplace-distributed residuals collapses the gradient near zero (outliers dominate), the model gives up at iter=4-10, and the composite gain over raw is left on the table.

The fix is policy: when the target distribution moments indicate Laplace / Student-t / contaminated tails, switch the inner boosting objective to L1 (Laplace MLE) or Huber. This module exposes the policy as a pure function that takes a 1-D target array and returns a per-backend recommendation:

    {"cb": "MAE", "lgb": "regression_l1", "xgb": "reg:absoluteerror",
     "rationale": "...", "skew": ..., "excess_kurt": ...}

Wiring: callers thread the returned per-backend loss name into the model factory's ``common_params`` / ``loss_function`` / ``objective`` argument BEFORE building the estimator. The recommended call site is ``_phase_train_one_target`` just after the per-target ``y_train`` is materialised; the recommendation is a one-shot O(n) compute on the target array. Composite-target paths feed the residual T directly (``y_T = transform.forward(y, base)``) so the loss matches the actual distribution the inner sees.

Thresholds. The cutoffs come from the empirical excess-kurtosis ladder verified in ``regression_residual_audit.py`` and 2026-05-26 production observation on a +42.67-kurt address composite:
- ``excess_kurt < 1.5`` -> near-Gaussian, keep RMSE (default).
- ``excess_kurt in [1.5, 20]`` -> heavy tails (Laplace, Student-t,
  contaminated). Huber where supported (LightGBM ``huber``, CatBoost
  ``Huber:delta=1.345``); XGBoost falls back to ``reg:pseudohubererror``.
- ``excess_kurt > 20`` -> extreme-kurt: Huber itself collapses
  (``delta * sign(r)`` approx 0 when most rows have r ~ 0; booster
  ES at iter=0/1). Revert to RMSE so training proceeds; the
  ``_maybe_refit_on_degenerate_best_iter`` safety net handles
  residual edge cases below this threshold.

Note: the previous (1.5, 10] MAE band was collapsed into the single Huber band. Pure L1 / MAE was triggering the "MAE-gradient-is-noise" pathology on heavy-kurtosis residuals (kurt=6.37 -> CB es_best_iter=1, LGB es_best_iter=5) -- the constant-magnitude sign-gradient stops the boosting iteration almost immediately. Huber retains a useful gradient on small residuals AND attenuates outlier influence within the moderate-heavy range. The 2026-05-26 extreme-kurt band is the upper bound where Huber's gradient itself disappears. Operators who explicitly want the Laplace MLE call with ``target_quantile=0.5``.

If the target is too small to estimate moments reliably (``n_finite < 30``) the recommendation is conservative (RMSE, the standard default).
"""
from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

_EXCESS_KURT_HEAVY: float = 1.5
"""Threshold above which Gaussian assumptions break and a robust loss
(Huber by default; L1 reachable via ``prefer_l1_above_kurt``) wins.

History:
* Originally: 1.5 -> pure L1/MAE.
* Round 3: raised 1.5 -> 3.0 because the MAE gradient on
  near-zero residuals (heavy-kurt composite, ``excess_kurt=6.37``)
  is just ``sign(noise)`` -- constant magnitude noise; CatBoost early-
  stopped at iter=1. New Huber band on ``excess_kurt in (1.5, 3.0]``.
* Round 5 (this fix): collapsed the (3.0, 10.0] MAE band
  too. The kurt=6.37 residuals STILL hit L1 with the previous
  ladder and CB/LGB STILL underfit (es_best_iter=1 / 5). Pure L1 is
  rare in practice; Huber dominates on the full kurt > 1.5 range with
  bounded-influence loss that retains a useful gradient on small
  residuals AND attenuates outlier influence. Threshold lowered back
  to 1.5 with the understanding that ALL the "heavy" cases above 1.5
  route through Huber, not L1. ``EXCESS_KURT_EXTREME`` is preserved
  for the original extreme-tail Huber-with-larger-delta path; the
  default Huber-delta-1.345 covers most leptokurtic distributions."""

_EXCESS_KURT_EXTREME: float = 10.0
"""Threshold matching ``regression_residual_audit.EXCESS_KURT_EXTREME``. Above this the residual is contaminated (Student-t / mixture); Huber's bounded influence function is the safe default."""

_EXCESS_KURT_HUBER_FAILS: float = 20.0
"""Upper threshold above which Huber itself collapses on CB / LGB / XGB and we revert to RMSE.

The 2026-05-26 production failure on a TVT-addres-TVT_prev composite
target (excess_kurt=+42.67, skew=-4.96, sample residual heavy LEFT
tail with most rows ~ 0) hit CB ``Huber:delta=1.345`` + IncToDec
overfit-detector (od_pval=1e-5) and STILL early-stopped at iter=0:
the Huber gradient ``delta * sign(r)`` is approximately zero when most
rows have ``r ~ 0``, so no candidate split improves the loss enough
for the detector to register progress. The model returned a constant
train-mean prediction (MAE=14.10, R^2=-5.05).

Above this threshold (loosely ~ 20: covers Cauchy-like and contaminated
mixture distributions where every bounded-influence loss collapses on
the tail) we revert to RMSE. RMSE is less robust to outliers but the
``2*r`` gradient always carries signal -- a non-trivial fit beats a
constant-prediction collapse. The complementary safety net at
``_training_loop._maybe_refit_on_degenerate_best_iter`` catches the
remaining cases where Huber fires (kurt in (1.5, 20]) but still
collapses on a specific dataset shape (cross-checks ``best_iter < 3``
post-fit, refits with RMSE)."""

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
            raise ValueError(f"target_quantile must be in (0, 1); got {alpha}")
        return {
            "cb": f"Quantile:alpha={alpha:.3f}",
            "lgb": "quantile",
            "lgb_extra_params": {"alpha": alpha},
            # XGBoost >=2.0 introduces ``reg:quantileerror`` with ``quantile_alpha`` kwarg.
            "xgb": "reg:quantileerror",
            "xgb_extra_params": {"quantile_alpha": alpha},
            "quantile_alpha": alpha,
            "rationale": (f"target_quantile={alpha:.3f} -- asymmetric-cost regression, " f"using quantile loss with alpha={alpha:.3f}."),
            "excess_kurt": excess_kurt,
            "n_finite": n_finite,
        }

    if n_finite < _MIN_SAMPLE_N or not math.isfinite(excess_kurt):
        return {
            "cb": "RMSE",
            "lgb": "regression",
            "xgb": "reg:squarederror",
            "rationale": f"n_finite={n_finite} < {_MIN_SAMPLE_N} -- insufficient sample for kurtosis estimate; RMSE default.",
            "excess_kurt": excess_kurt,
            "n_finite": n_finite,
        }
    # Above _EXCESS_KURT_HUBER_FAILS even Huber collapses (gradient
    # delta*sign(r) approx 0 on a near-zero residual distribution with
    # heavy tail; CB / LGB / XGB ES at iter=0). Recommend RMSE
    # directly so we skip the doomed Huber fit + refit cycle instead
    # of letting ``_maybe_refit_on_degenerate_best_iter`` clean up.
    if excess_kurt > _EXCESS_KURT_HUBER_FAILS:
        return {
            "cb": "RMSE",
            "lgb": "regression",
            "xgb": "reg:squarederror",
            "rationale": (
                f"excess_kurt={excess_kurt:.2f} > {_EXCESS_KURT_HUBER_FAILS} "
                f"-- Huber gradient collapses on extreme-kurt residual "
                f"(delta*sign(r) approx 0 when most rows have r ~ 0); "
                f"booster early-stops at iter=0/1. Reverting to RMSE: "
                f"less robust to outliers but always trainable. Consider "
                f"y_quantile_clip or similar preprocessing if outlier "
                f"impact on the fit becomes a concern."
            ),
            "excess_kurt": excess_kurt,
            "n_finite": n_finite,
        }
    if excess_kurt > _EXCESS_KURT_HEAVY:
        _extreme = excess_kurt > _EXCESS_KURT_EXTREME
        # MAD-calibrated Huber slope. The canon 1.345 is robust-stats
        # tuned to MAD UNITS (~0.67*sigma for a Normal sample). LGB
        # ``huber`` and CB ``Huber:delta=1.345`` are documented to
        # operate in MAD units. XGB ``reg:pseudohubererror`` takes
        # ``huber_slope`` in RAW residual units; defaulting to 1.0 on a
        # T-scale residual with std~13 (observed in a heavy-residual composite)
        # means slope < 10% of std -> pseudo-Huber is quadratic over
        # essentially every realistic residual -> tree splits chase
        # heavy-tail outliers -> pred range blows from |T|<=50 to +340.
        # Scale to MAD(target) so XGB matches LGB/CB regimes.
        try:
            arr = np.asarray(y_target, dtype=np.float64).reshape(-1)
            arr_f = arr[np.isfinite(arr)]
            mad = float(np.median(np.abs(arr_f - np.median(arr_f))))
        except (TypeError, ValueError):
            mad = 0.0
        xgb_huber_slope = max(1.0, 1.345 * mad) if mad > 0 else 1.0
        # CB: switch overfit detector from Iter (constant-magnitude
        # gradient stops ES at iter=1 on small-residual composite
        # targets) to IncToDec with a small p-value so tiny absolute
        # improvements still count as progress.
        # NOTE: CB canonizes ``od_wait`` and ``early_stopping_rounds``
        # into the SAME parameter group and raises CatBoostError when
        # both are passed. The base CB params already set
        # ``early_stopping_rounds`` (from helpers.CB_GENERAL_PARAMS), so
        # we use that key here -- semantically equivalent to od_wait,
        # no collision. Pre-fix this dict set ``od_wait`` directly and
        # blew up with "only one of the parameters od_wait,
        # early_stopping_rounds should be initialized" on every CB fit.
        cb_extra = {
            "od_type": "IncToDec",
            "od_pval": 1e-5,
            "early_stopping_rounds": 100,
        }
        return {
            "cb": "Huber:delta=1.345",
            "lgb": "huber",
            "xgb": "reg:pseudohubererror",
            "xgb_extra_params": {"huber_slope": xgb_huber_slope},
            "cb_extra_params": cb_extra,
            "rationale": (
                f"excess_kurt={excess_kurt:.2f} > {_EXCESS_KURT_HEAVY}"
                + (f" (extreme tails, > {_EXCESS_KURT_EXTREME})" if _extreme else "")
                + " -- Huber bounded-influence loss keeps the "
                f"gradient informative on small residuals while "
                f"attenuating outlier influence. XGB huber_slope="
                f"{xgb_huber_slope:.4g} (MAD-calibrated). CB od_pval=1e-5 "
                f"+ early_stopping_rounds=100 (prevent ES at iter=1)."
            ),
            "excess_kurt": excess_kurt,
            "n_finite": n_finite,
            "mad": mad,
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
