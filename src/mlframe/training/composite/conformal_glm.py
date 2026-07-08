"""Variance-scaled split-conformal prediction intervals for ``CompositeGLMEstimator``.

The GLM composite predicts a positive mean ``mu_hat(x)`` on the original scale but
carries no uncertainty band. For count / positive targets the residual spread GROWS
with the mean (a Poisson with mean 50 scatters far more than one with mean 0.5), so a
constant-width band (the Gaussian ``conformal.py`` recipe) badly mis-shapes the
interval -- too wide near zero, too tight in the bulk. The right nonconformity score
divides the residual by the family's standard deviation ``sqrt(V(mu_hat))``:

    s_i = |y_i - mu_hat_i| / sqrt(V(mu_hat_i))

with the family VARIANCE FUNCTION ``V``:

    Poisson  V(mu) = mu
    Gamma    V(mu) = mu^2
    Tweedie  V(mu) = mu^p   (p in (1, 2))

The split-conformal finite-sample ``(1-alpha)`` quantile ``Q`` of those standardized
scores then gives a HETEROSCEDASTIC band whose half-width scales with the local
standard deviation::

    interval(x) = [ max(0, mu_hat(x) - Q * sqrt(V(mu_hat(x)))),  mu_hat(x) + Q * sqrt(V(mu_hat(x))) ]

so the band is wider where ``mu_hat`` is larger (by construction) and clipped at 0 so
it never claims a negative count / rate. Under exchangeability of the calibration and
test rows this guarantees marginal coverage ``>= 1 - alpha`` for ANY mean model -- no
distributional assumption beyond the variance-function shape used to normalise.

Design choices mirror ``conformal.py``:
- Calibration MUST run on HELD-OUT rows the inner never trained on (the suite val
  split, or an OOF fold); conformal validity rests on calibration/test exchangeability.
- The radius is stored per-alpha in a plain ``self._glm_conformal_q_`` dict of floats,
  so ``sklearn.clone`` / pickle stay clean and the wrapper captures no frames.
"""
from __future__ import annotations

import math

import numpy as np


def _variance_function(mu: np.ndarray, family: str, tweedie_power: float) -> np.ndarray:
    """Family variance ``V(mu)``: Poisson ``mu``, Gamma ``mu^2``, Tweedie ``mu^p``.

    ``mu`` is the GLM mean on the original scale (already floored positive by the
    estimator). The returned variance is floored at a tiny epsilon so the std
    ``sqrt(V)`` used to normalise residuals is never exactly zero (which would make
    the standardized score blow up / divide-by-zero on a degenerate all-zero mean).
    """
    mu = np.asarray(mu, dtype=np.float64).reshape(-1)
    if family == "poisson":
        v = mu
    elif family == "gamma":
        v = mu * mu
    elif family == "tweedie":
        v = np.power(np.maximum(mu, 0.0), float(tweedie_power))
    else:
        raise ValueError(f"conformal_glm: unknown family {family!r}; choose poisson / gamma / tweedie.")
    return np.maximum(v, 1e-12)


def standardized_conformal_quantile(
    residuals: np.ndarray, std: np.ndarray, alpha: float,
) -> float:
    """Finite-sample ``(1-alpha)`` quantile of the standardized scores ``|r| / std``.

    Uses the conservative rank ``ceil((n+1)(1-alpha))`` (the smallest standardized
    score that guarantees marginal coverage ``>= 1-alpha``); returns ``+inf`` when
    that rank exceeds ``n`` (too few calibration points for the level) so the band is
    valid-but-uninformative rather than silently under-covering -- the same tiny-n
    contract as the Gaussian ``conformal_quantile``.
    """
    r = np.abs(np.asarray(residuals, dtype=np.float64).reshape(-1))
    s = np.asarray(std, dtype=np.float64).reshape(-1)
    if s.shape[0] != r.shape[0]:
        raise ValueError(f"standardized_conformal_quantile: {s.shape[0]} std values for " f"{r.shape[0]} residuals")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"conformal alpha must be in (0, 1), got {alpha!r}")
    scores = r / s
    scores = scores[np.isfinite(scores)]
    n = int(scores.size)
    if n == 0:
        return float("inf")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        return float("inf")
    return float(np.sort(scores)[rank - 1])


def calibrate_conformal_glm(self, X_cal, y_cal, alpha=0.1):
    """Fit the variance-scaled split-conformal radius from a HELD-OUT calibration set.

    ``X_cal`` / ``y_cal`` MUST be rows the inner estimator did NOT train on (the suite
    val split, or an OOF fold) -- conformal validity rests on the calibration rows
    being exchangeable with the test rows, which in-sample rows are not.

    Computes the standardized score ``|y - mu_hat| / sqrt(V(mu_hat))`` (V = the family
    variance function) on the calibration rows and stores its finite-sample
    ``(1-alpha)`` quantile in ``self._glm_conformal_q_[round(alpha, 6)]``. ``alpha``
    may be a scalar or an iterable of levels; each is calibrated and cached so
    :func:`predict_interval_glm` can serve any pre-calibrated level cheaply. Returns
    ``self`` (sklearn-style).
    """
    if not hasattr(self, "estimator_"):
        from sklearn.exceptions import NotFittedError
        raise NotFittedError("CompositeGLMEstimator.calibrate_conformal_glm called before fit.")
    y_true = np.asarray(y_cal, dtype=np.float64).reshape(-1)
    mu = np.asarray(self.predict(X_cal), dtype=np.float64).reshape(-1)
    if mu.shape[0] != y_true.shape[0]:
        raise ValueError("calibrate_conformal_glm: predict produced " f"{mu.shape[0]} rows but y_cal has {y_true.shape[0]}")
    residuals = y_true - mu
    std = np.sqrt(_variance_function(mu, self.family, self.tweedie_power))
    alphas = [alpha] if np.isscalar(alpha) else list(alpha)
    if not hasattr(self, "_glm_conformal_q_") or self._glm_conformal_q_ is None:
        self._glm_conformal_q_ = {}
    for a in alphas:
        self._glm_conformal_q_[round(float(a), 6)] = standardized_conformal_quantile(  # type: ignore[arg-type]  # a is a scalar drawn from alphas (float | Sequence[float])
            residuals, std, float(a),  # type: ignore[arg-type]  # a is a scalar drawn from alphas (float | Sequence[float])
        )
    self._glm_conformal_n_cal_ = int(np.isfinite(residuals / std).sum())
    return self


def predict_interval_glm(self, X, alpha=0.1):
    """Return variance-scaled ``(lower, upper)`` original-scale intervals of marginal
    coverage ``>= 1 - alpha``, with band width that GROWS with the predicted mean.

    Requires a prior :func:`calibrate_conformal_glm` at this ``alpha`` (a clear error
    otherwise -- the radius cannot be invented from train rows without breaking
    conformal validity). The half-width is ``Q * sqrt(V(mu_hat(x)))`` (Q = the
    calibrated standardized quantile, V = the family variance function), so larger
    ``mu_hat`` => wider band. The lower bound is clipped at 0 so the interval never
    claims a negative count / rate.
    """
    key = round(float(alpha), 6)
    q = getattr(self, "_glm_conformal_q_", {}) or {}
    if key not in q:
        raise RuntimeError(
            f"predict_interval_glm: no conformal radius calibrated for alpha={alpha}. "
            f"Call calibrate_conformal_glm(X_cal, y_cal, alpha={alpha}) on a held-out "
            f"set first (calibrated levels: {sorted(q.keys())})."
        )
    radius = q[key]
    mu = np.asarray(self.predict(X), dtype=np.float64).reshape(-1)
    std = np.sqrt(_variance_function(mu, self.family, self.tweedie_power))
    half = radius * std
    lower = np.maximum(mu - half, 0.0)
    upper = mu + half
    return lower, upper
