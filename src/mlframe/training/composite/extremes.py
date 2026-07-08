"""Extreme-value (Peaks-Over-Threshold / GPD) tail composite estimator.

``TailCompositeEstimator`` targets the heavy-tailed regression case where the
mean and the central quantiles capture the BODY of the conditional distribution
well, but the far upper tail (q -> 1, e.g. 0.99 / 0.999) is genuinely heavier
than the empirical quantile can express. The empirical quantile of a finite
sample SATURATES at the largest observed residual: it can never exceed the max,
so it systematically UNDER-estimates the true tail quantile of a heavy-tailed
(Pareto / Student-t) target. Extreme-value theory gives the principled fix.

Design.

1. A usual point composite (:class:`CompositeTargetEstimator`) is fit for the
   body of the distribution -- this gives ``mu_hat(X)``, the conditional point
   prediction on the y-scale.
2. On a TRAIN/HELD-OUT split the absolute residuals ``r = |y - mu_hat(X)|`` are
   formed. A high threshold ``u`` is taken as the ``threshold_pct`` quantile of
   those residuals (default the 95th percentile). The Pickands-Balkema-de Haan
   theorem states that, for a broad class of distributions, the exceedances
   ``r - u | r > u`` converge to a Generalized Pareto Distribution (GPD) as
   ``u`` grows. We fit a GPD (shape ``xi``, scale ``beta``) to those exceedances
   by MLE (``scipy.stats.genpareto``, ``floc=0``) or method-of-moments.
3. :meth:`predict_tail_quantile(X, q)` for a high ``q`` (above the threshold's
   coverage ``F(u) = threshold_pct``) returns the POT tail formula

       Q(q) = mu_hat(X) +/- ( u + (beta/xi) * ( ((1-q)/(1-F_u))**(-xi) - 1 ) )

   i.e. the GPD inverse-survival extrapolation OFFSET around the point
   prediction, rather than an empirical residual quantile that caps at the
   observed max. For ``q`` at/below the threshold coverage it falls back to the
   empirical residual quantile (the body region where EVT gives no advantage).

Train-only / held-out fit (CRITICAL). The point composite, the threshold ``u``,
and the GPD ``(xi, beta)`` are ALL fit on the training rows only (or an explicit
held-out residual split passed by the caller). ``predict_tail_quantile`` reads
only the FITTED tail params + the body composite's point prediction -- it never
re-estimates anything from the query ``X``.

Memory. No frame copy: the body composite already pulls the narrow base column
via its inner heads; this wrapper only ever touches the resulting ``(n,)`` point
prediction vector and the 1-D residual array. The GPD fit is on the (typically
~5% of n) exceedances, a small 1-D array.

cProfile (fit n=20k + 200 tail-quantile calls, stub inner): the wrapper work
outside the single inner composite fit is one ``np.abs`` residual pass, one
``np.quantile`` for the threshold, and a single ``scipy.stats.genpareto.fit`` on
~1k exceedances (~3 ms) -- dwarfed by the inner boosting fit. ``predict_tail_-
quantile`` is a closed-form vectorised expression over the point vector: <0.1 ms
per call. No actionable wrapper-side speedup; cost is the inner composite fit.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone

from .estimator import CompositeTargetEstimator

logger = logging.getLogger(__name__)

# Default high threshold for the POT split: the 95th percentile of the absolute
# residuals. Above this the Balkema-de Haan GPD limit is a good approximation for
# heavy tails while still leaving enough exceedances (~5% of n) to fit (xi, beta).
_DEFAULT_THRESHOLD_PCT: float = 0.95

# Minimum number of threshold exceedances required for a stable GPD MLE. Below
# this the GPD fit is unreliable (the shape parameter is high-variance on a
# handful of points), so we fall back to the empirical residual quantile.
_MIN_EXCEEDANCES: int = 20


def fit_gpd_exceedances(
    exceedances: np.ndarray,
    method: str = "mle",
) -> "tuple[float, float]":
    """Fit a GPD ``(shape xi, scale beta)`` to non-negative threshold exceedances.

    The exceedances are ``r - u`` for ``r > u`` (so all ``>= 0``); the GPD
    location is pinned at 0. Two estimators:

    - ``"mle"`` -- ``scipy.stats.genpareto.fit(exceedances, floc=0)``; the
      maximum-likelihood shape + scale. Default (most accurate for n >= ~50).
    - ``"mom"`` -- method of moments: with sample mean ``m`` and variance ``s2``,
      ``xi = 0.5 * (1 - m*m/s2)`` and ``beta = 0.5 * m * (m*m/s2 + 1)`` (Hosking &
      Wallis 1987). Cheap + robust for small n; no optimiser.

    Returns ``(xi, beta)`` with ``beta`` forced strictly positive.
    """
    ex = np.asarray(exceedances, dtype=np.float64)
    ex = ex[np.isfinite(ex)]
    if ex.size == 0:
        return 0.0, 1.0
    if method == "mom":
        m = float(np.mean(ex))
        s2 = float(np.var(ex))
        if s2 <= 0.0 or m <= 0.0:
            return 0.0, max(m, 1e-12)
        ratio = m * m / s2
        xi = 0.5 * (1.0 - ratio)
        beta = 0.5 * m * (ratio + 1.0)
        return float(xi), float(max(beta, 1e-12))
    # MLE via scipy; floc=0 pins the GPD location at the threshold origin.
    xi, _loc, beta = stats.genpareto.fit(ex, floc=0.0)
    return float(xi), float(max(beta, 1e-12))


def gpd_tail_quantile(
    q: float,
    threshold: float,
    threshold_cov: float,
    xi: float,
    beta: float,
) -> float:
    """POT/GPD inverse-survival extrapolation of the ``q``-quantile residual.

    Given the exceedance probability ``1 - threshold_cov = P(r > u)``, the GPD
    quantile function maps a tail level ``q`` (with ``q > threshold_cov``) to the
    residual offset

        u + (beta/xi) * ( ((1-q)/(1-threshold_cov))**(-xi) - 1 )      (xi != 0)
        u + beta * ( -ln( (1-q)/(1-threshold_cov) ) )                 (xi == 0)

    For ``q <= threshold_cov`` the formula is undefined (the body region); the
    caller routes those to the empirical residual quantile instead.
    """
    surv_ratio = (1.0 - q) / (1.0 - threshold_cov)
    if abs(xi) < 1e-8:
        return float(threshold + beta * (-np.log(surv_ratio)))
    return float(threshold + (beta / xi) * (surv_ratio ** (-xi) - 1.0))


class TailCompositeEstimator(BaseEstimator, RegressorMixin):
    """Heavy-tail composite: body point composite + POT/GPD residual tail.

    Parameters
    ----------
    base_estimator
        Unfitted inner regressor prototype for the BODY point composite (cloned
        into a :class:`CompositeTargetEstimator`). Any sklearn-style regressor.
    transform_name, base_column, base_columns, group_column
        Forwarded verbatim to the inner :class:`CompositeTargetEstimator` (the
        composite-target transform wiring for the body point model).
    threshold_pct
        Residual quantile used as the high POT threshold ``u`` (default 0.95 ->
        the 95th percentile of ``|y - mu_hat|``). The GPD is fit to exceedances
        above ``u``; ``predict_tail_quantile`` uses the GPD only for ``q`` above
        this coverage.
    gpd_method
        ``"mle"`` (default) or ``"mom"`` -- the GPD shape/scale estimator.
    min_exceedances
        Minimum exceedance count for a GPD fit; below it the tail falls back to
        the empirical residual quantile. Default 20.
    two_sided
        When True (default) the absolute residuals model a symmetric tail and
        ``predict_tail_quantile`` adds the offset to ``mu_hat`` for an UPPER tail
        quantile. When False only positive residuals (``y - mu_hat > 0``) feed
        the GPD (a one-sided upper-tail model).

    Attributes set at fit
    ---------------------
    body_estimator_
        The fitted body :class:`CompositeTargetEstimator`.
    threshold_
        The fitted high threshold ``u`` (residual scale).
    threshold_cov_
        Coverage ``F(u) = threshold_pct`` actually used (a float in (0, 1)).
    gpd_shape_, gpd_scale_
        The fitted GPD ``(xi, beta)``; ``None`` when the GPD fallback fired.
    residuals_sorted_
        Sorted absolute (or signed-positive) residuals, for the empirical-quantile
        body fallback.
    gpd_fitted_
        True when a GPD tail was fit; False when too-few-exceedances fallback fired.
    feature_names_in_
        Inherited from the body composite (best effort).
    """

    def __init__(
        self,
        base_estimator: Any = None,
        transform_name: str = "diff",
        base_column: str = "",
        base_columns: "Any | None" = None,
        group_column: str = "",
        threshold_pct: float = _DEFAULT_THRESHOLD_PCT,
        gpd_method: str = "mle",
        min_exceedances: int = _MIN_EXCEEDANCES,
        two_sided: bool = True,
    ) -> None:
        self.base_estimator = base_estimator
        self.transform_name = transform_name
        self.base_column = base_column
        self.base_columns = base_columns
        self.group_column = group_column
        self.threshold_pct = threshold_pct
        self.gpd_method = gpd_method
        self.min_exceedances = min_exceedances
        self.two_sided = two_sided

    # -- fit -----------------------------------------------------------------

    def fit(
        self,
        X: Any,
        y: Any,
        residual_X: "Any | None" = None,
        residual_y: "Any | None" = None,
        **fit_params: Any,
    ) -> "TailCompositeEstimator":
        """Fit the body composite then the GPD tail on the (held-out) residuals.

        The body point composite is fit on ``(X, y)``. The residuals used for the
        threshold + GPD fit are computed on ``(residual_X, residual_y)`` when an
        explicit held-out split is given (the recommended train-only/held-out
        pattern -- avoids the in-sample residual being optimistically small), else
        on the training ``(X, y)`` themselves.
        """
        if not 0.0 < float(self.threshold_pct) < 1.0:
            raise ValueError(f"threshold_pct must be in (0, 1), got {self.threshold_pct!r}")

        self.body_estimator_ = CompositeTargetEstimator(
            base_estimator=clone(self.base_estimator) if self.base_estimator is not None else None,
            transform_name=self.transform_name,
            base_column=self.base_column,
            base_columns=self.base_columns,
            group_column=self.group_column,
        )
        self.body_estimator_.fit(X, y, **fit_params)

        # Residuals on the held-out split when given (honest tail), else in-sample.
        r_X = residual_X if residual_X is not None else X
        r_y = residual_y if residual_y is not None else y
        mu = np.asarray(self.body_estimator_.predict(r_X), dtype=np.float64)
        y_arr = np.asarray(r_y, dtype=np.float64)
        signed = y_arr - mu
        resid = np.abs(signed) if self.two_sided else signed
        resid = resid[np.isfinite(resid)]
        if not self.two_sided:
            resid = resid[resid > 0.0]

        self.threshold_cov_ = float(self.threshold_pct)
        if resid.size == 0:
            # Degenerate: no usable residuals -> zero tail offset.
            self.threshold_ = 0.0
            self.residuals_sorted_ = np.zeros(1, dtype=np.float64)
            self.gpd_shape_ = None
            self.gpd_scale_ = None
            self.gpd_fitted_ = False
        else:
            self.residuals_sorted_ = np.sort(resid)
            self.threshold_ = float(np.quantile(self.residuals_sorted_, self.threshold_cov_))
            exceed = resid[resid > self.threshold_] - self.threshold_
            if exceed.size >= int(self.min_exceedances):
                xi, beta = fit_gpd_exceedances(exceed, method=self.gpd_method)
                self.gpd_shape_ = xi
                self.gpd_scale_ = beta
                self.gpd_fitted_ = True
            else:
                # Too few exceedances for a stable GPD -> empirical fallback.
                self.gpd_shape_ = None
                self.gpd_scale_ = None
                self.gpd_fitted_ = False
                logger.debug(
                    "TailCompositeEstimator: %d exceedances < min %d; " "GPD tail disabled (empirical-quantile fallback).",
                    exceed.size,
                    int(self.min_exceedances),
                )

        names = getattr(self.body_estimator_, "feature_names_in_", None)
        if names is not None:
            try:
                self.feature_names_in_ = list(names)
            except Exception as e:  # pragma: no cover
                logger.debug("swallowed exception in extremes.py: %s", e)
                pass
        cols = getattr(X, "columns", None)
        if cols is not None:
            self.n_features_in_ = len(cols)
        elif getattr(X, "shape", None) is not None and len(X.shape) >= 2:
            self.n_features_in_ = int(X.shape[1])
        return self

    # -- predict -------------------------------------------------------------

    def predict(self, X: Any) -> "np.ndarray":
        """Body point prediction ``mu_hat(X)`` (thin pass-through to the body)."""
        self._check_fitted()
        return self.body_estimator_.predict(X)

    def _empirical_residual_quantile(self, q: float) -> float:
        """Empirical ``q``-quantile of the fitted residuals (body fallback)."""
        return float(np.quantile(self.residuals_sorted_, q))

    def tail_residual_offset(self, q: float) -> float:
        """The residual-scale ``q``-quantile offset (GPD tail or empirical body).

        For ``q`` above the threshold coverage AND a fitted GPD, returns the POT
        extrapolation; otherwise the empirical residual quantile. This is the
        scalar offset added to ``mu_hat`` by :meth:`predict_tail_quantile`.
        """
        self._check_fitted()
        if not 0.0 < float(q) < 1.0:
            raise ValueError(f"q must be in (0, 1), got {q!r}")
        if self.gpd_fitted_ and float(q) > self.threshold_cov_:
            assert self.gpd_shape_ is not None and self.gpd_scale_ is not None  # set together with gpd_fitted_=True
            return gpd_tail_quantile(
                float(q),
                self.threshold_,
                self.threshold_cov_,
                float(self.gpd_shape_),
                float(self.gpd_scale_),
            )
        return self._empirical_residual_quantile(float(q))

    def predict_tail_quantile(self, X: Any, q: float) -> "np.ndarray":
        """Conditional upper-tail ``q``-quantile of ``y`` at the rows of ``X``.

        Returns ``mu_hat(X) + offset(q)`` where ``offset`` is the GPD-extrapolated
        residual quantile for high ``q`` (``q`` above the threshold coverage and a
        fitted GPD), else the empirical residual quantile. Because the GPD inverse
        survival is unbounded above (for ``xi >= 0``), the extrapolated 0.999
        quantile can EXCEED the largest observed residual -- the whole point: the
        empirical quantile saturates at the sample max and under-estimates a heavy
        tail, while the GPD extrapolates beyond it.
        """
        self._check_fitted()
        mu = np.asarray(self.body_estimator_.predict(X), dtype=np.float64)
        offset = self.tail_residual_offset(float(q))
        return mu + offset

    @property
    def tail_params_(self) -> "dict[str, Any]":
        """The fitted tail parameters as a plain dict (for serving / reporting)."""
        self._check_fitted()
        return {
            "threshold": self.threshold_,
            "threshold_cov": self.threshold_cov_,
            "gpd_shape": self.gpd_shape_,
            "gpd_scale": self.gpd_scale_,
            "gpd_fitted": self.gpd_fitted_,
            "gpd_method": self.gpd_method,
            "two_sided": self.two_sided,
        }

    def _check_fitted(self) -> None:
        """Raise ``NotFittedError`` unless ``fit`` has already run (probed via presence of ``body_estimator_``)."""
        if not hasattr(self, "body_estimator_"):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("TailCompositeEstimator is not fitted; call fit() first.")
