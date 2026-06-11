"""Split-conformal prediction intervals for ``CompositeTargetEstimator``.

The wrapper already produces honest y-scale point predictions; conformal adds a
distribution-free, finite-sample-valid prediction INTERVAL on top. Given a
held-out calibration set (rows the inner never trained on -- the suite's val
split, or an OOF fold), we compute the empirical quantile of the absolute
calibration residuals and widen every point prediction by it:

    interval(x) = [ y_hat(x) - q, y_hat(x) + q ]

with ``q`` the ``ceil((n+1)(1-alpha))/n`` empirical quantile of
``|y_cal - y_hat(x_cal)|`` (the standard split-conformal level with the
finite-sample +1 correction). Under exchangeability of the calibration and test
rows this guarantees marginal coverage >= 1 - alpha for ANY underlying model --
no Gaussian / homoscedastic assumption. The interval is symmetric in y-scale
(absolute-residual nonconformity); for strongly heteroscedastic targets a
normalised score is a future refinement, noted in ``calibrate_conformal``.

Design choices mirroring the rest of the package:
- The calibration quantile(s) are stored per-alpha in ``self._conformal_q_`` --
  a plain dict of floats, so ``sklearn.clone`` / pickle stay clean and the
  wrapper carries no captured frames.
- Calibration consumes the inner's y-scale ``predict`` (the full
  transform-and-invert path), so the interval is on the ORIGINAL y scale the
  user cares about, not the T scale.
"""
from __future__ import annotations

import math

import numpy as np


def conformal_quantile(residuals: np.ndarray, alpha: float) -> float:
    """Split-conformal radius: the finite-sample ``(1-alpha)`` quantile of the
    absolute residuals.

    Uses the conservative rank ``ceil((n+1)(1-alpha))`` (the smallest residual
    that guarantees marginal coverage >= 1-alpha). Returns ``+inf`` when the
    requested rank exceeds ``n`` (too few calibration points for the level) so
    the interval is uninformative-but-valid rather than silently under-covering.
    """
    r = np.abs(np.asarray(residuals, dtype=np.float64).reshape(-1))
    r = r[np.isfinite(r)]
    n = int(r.size)
    if n == 0:
        return float("inf")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"conformal alpha must be in (0, 1), got {alpha!r}")
    # 1-indexed rank of the order statistic that bounds 1-alpha mass.
    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
        # Not enough calibration points to certify this level at finite n.
        return float("inf")
    r_sorted = np.sort(r)
    return float(r_sorted[rank - 1])


def calibrate_conformal(self, X_cal, y_cal, alpha=0.1):
    """Fit the split-conformal radius from a held-out calibration set.

    ``X_cal`` / ``y_cal`` MUST be rows the inner estimator did NOT train on
    (the suite val split, or an OOF fold) -- conformal validity rests on the
    calibration rows being exchangeable with the test rows, which in-sample
    rows are not. Stores ``self._conformal_q_[round(alpha, 6)]`` and returns
    ``self`` (sklearn-style).

    ``alpha`` may be a scalar or an iterable of levels; each is calibrated and
    cached so ``predict_interval`` can serve any pre-calibrated level cheaply.

    Heteroscedastic targets: this uses the plain absolute-residual score, which
    yields a CONSTANT-width band. A normalised (e.g. /sigma_hat(x)) score gives
    variable-width bands -- a future option, not wired here.
    """
    if not hasattr(self, "estimator_"):
        from sklearn.exceptions import NotFittedError
        raise NotFittedError(
            "CompositeTargetEstimator.calibrate_conformal called before fit."
        )
    y_true = np.asarray(y_cal, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(self.predict(X_cal), dtype=np.float64).reshape(-1)
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(
            "calibrate_conformal: predict produced "
            f"{y_pred.shape[0]} rows but y_cal has {y_true.shape[0]}"
        )
    residuals = y_true - y_pred
    alphas = [alpha] if np.isscalar(alpha) else list(alpha)
    if not hasattr(self, "_conformal_q_") or self._conformal_q_ is None:
        self._conformal_q_ = {}
    for a in alphas:
        self._conformal_q_[round(float(a), 6)] = conformal_quantile(residuals, float(a))
    self._conformal_n_cal_ = int(np.isfinite(residuals).sum())
    return self


def predict_interval(self, X, alpha=0.1):
    """Return ``(lower, upper)`` y-scale prediction intervals of marginal
    coverage ``>= 1 - alpha``.

    Requires a prior :meth:`calibrate_conformal` at this ``alpha`` (a clear
    error otherwise -- the radius cannot be invented from train rows without
    breaking conformal validity). The band is ``predict(X) +/- q`` where ``q``
    is the calibrated radius; it inherits the wrapper's post-inverse y-clip via
    ``predict`` on the point estimate, then the band is clipped to the same
    train envelope so the interval never claims an unphysical value.
    """
    key = round(float(alpha), 6)
    q = getattr(self, "_conformal_q_", {}) or {}
    if key not in q:
        raise RuntimeError(
            f"predict_interval: no conformal radius calibrated for alpha={alpha}. "
            f"Call calibrate_conformal(X_cal, y_cal, alpha={alpha}) on a held-out "
            f"set first (calibrated levels: {sorted(q.keys())})."
        )
    radius = q[key]
    point = np.asarray(self.predict(X), dtype=np.float64).reshape(-1)
    lower = point - radius
    upper = point + radius
    # Keep the band inside the same train envelope the point estimate uses.
    params = getattr(self, "fitted_params_", {}) or {}
    lo_b = params.get("y_clip_low", float("-inf"))
    hi_b = params.get("y_clip_high", float("inf"))
    lower = np.clip(lower, lo_b, hi_b)
    upper = np.clip(upper, lo_b, hi_b)
    return lower, upper
