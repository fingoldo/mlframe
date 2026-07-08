"""Heteroscedastic (predictive-variance) composite: per-row predictive std / interval on the y-scale.

Point composites emit only ``y_hat``; on a target whose noise scale depends on the inputs (heteroscedastic --
the common case in real regressions) they cannot say WHERE the prediction is uncertain. :class:`HeteroscedasticCompositeEstimator`
models BOTH the conditional mean AND the conditional variance on the transform (``T``) scale, then inverts to y.

Mean + variance method.

- The mean is a :class:`CompositeTargetEstimator` fit as usual (all transform fit / inverse / base-extraction /
  domain / clip / fallback machinery is reused verbatim -- NOT duplicated here). Its inner predicts the conditional
  mean of ``T``; the wrapper inverts to y.
- The variance is estimated on the SAME ``T`` scale. Two backends, dispatched by availability:
  - ``ngboost`` (preferred when installed): an ``NGBRegressor`` with a Normal distribution fit on ``T`` gives a
    native per-row ``scale`` (predictive std of ``T``) from its second natural-gradient boosted head -- no separate
    residual model. Used only when ``prefer_ngboost`` and the import succeeds.
  - two-model fallback (always available): a SECOND regressor is fit on the LOG of the squared ``T``-residuals
    ``log(max(r^2, floor))`` of the mean model, so ``sigma_T = exp(0.5 * pred)`` is positive by construction and the
    log-scale target keeps the variance head well-behaved (NGBoost-style mean/variance split without the dependency).
- Either backend's ``sigma_T`` is CALIBRATED by a single global factor ``c = sqrt(mean((r / sigma_T)^2))`` measured on
  the fit residuals, which removes the log-chi-square bias of the two-model head (and re-centres NGBoost's scale) so a
  central Gaussian interval is well-calibrated.

Propagation to y. The predictive band on the ``T`` scale ``[T_hat - z*sigma_T, T_hat + z*sigma_T]`` (``z`` the Normal
quantile for the requested coverage) is pushed through the SHARED transform inverse row-by-row, so the y-scale interval
respects the (possibly nonlinear, monotone) transform exactly. The per-row predictive std on the y-scale is read off the
inverted band as ``(y_hi - y_lo) / (2 z)`` -- for the additive-residual transforms (``linear_residual`` / ``diff``) the
inverse is a pure per-row shift, so this reduces to ``sigma_T``; for ``ratio`` / ``logratio`` it correctly widens/narrows
with the base.

Surface: sklearn ``fit`` / ``predict`` (+ ``predict_std`` / ``predict_var`` / ``predict_interval``). ``predict`` returns the
mean composite's clipped y-scale point prediction; the UQ methods add the predictive-variance layer on top.

Memory. No frame copy: only the narrow base column is pulled (via the mean composite's own extractor) and the small
``(n,)`` T-mean / T-sigma vectors are materialised; the feature frame flows straight into the inner estimators.

cProfile (fit + predict_std, n=4000 x 4 cols, HistGradientBoosting inner, two-model backend, ``_benchmarks/`` harness):
wall is dominated by the TWO inner boosting fits (mean + variance head), ~85% cumtime, already parallel inside sklearn.
The wrapper-side work -- one ``transform.forward``, three vectorised ``transform.inverse`` calls, the log/exp/calibration
reductions -- is <2 ms combined and does not register above cProfile's deep-stack attribution noise. No actionable
wrapper-side speedup; the cost is the inner model fits, tuned by the caller's choice of ``base_estimator``.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

from .estimator import CompositeTargetEstimator
from .transforms import get_transform

logger = logging.getLogger(__name__)

# Floor on squared residuals before the log so a zero/near-zero residual (a perfectly-fit or constant row) does not send
# the log-variance target to -inf; relative to the residual scale so it is unit-agnostic.
_SQ_RESID_REL_FLOOR: float = 1e-8


def _z_for_alpha(alpha: float) -> float:
    """Two-sided Normal quantile ``z`` such that a ``[mu - z*sigma, mu + z*sigma]`` band has coverage ``1 - alpha``."""
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha (nominal miscoverage) must be in (0, 1); got {alpha}")
    from scipy.stats import norm

    return float(norm.ppf(1.0 - alpha / 2.0))


class HeteroscedasticCompositeEstimator(BaseEstimator, RegressorMixin):
    """Predictive-variance composite: conditional mean AND conditional variance on the T-scale, inverted to y.

    Parameters
    ----------
    base_estimator
        Unfitted inner prototype for the MEAN model, cloned into a :class:`CompositeTargetEstimator` (shared transform
        machinery). Any sklearn regressor.
    variance_estimator
        Unfitted inner prototype for the two-model VARIANCE head (fit on ``log`` squared T-residuals). Defaults to a
        clone of ``base_estimator`` when None. Ignored when the ngboost backend is active.
    transform_name, base_column, base_columns, group_column, fallback_predict, drop_invalid_rows
        Forwarded to the inner :class:`CompositeTargetEstimator`. ``linear_residual`` is canonical.
    alpha
        Default nominal miscoverage for :meth:`predict_interval` / :meth:`predict_std` (0.1 = 90% interval).
    prefer_ngboost
        Use ``ngboost``'s native predictive scale when installed (default True); else the two-model fallback.
    residual_floor_rel
        Relative floor on squared residuals before the log (guards ``log(0)``).

    Attributes set at fit
    ---------------------
    mean_estimator_ : the fitted :class:`CompositeTargetEstimator` (mean + all transform state).
    variance_estimator_ : the fitted two-model variance head (None on the ngboost path).
    backend_ : ``"ngboost"`` or ``"two_model"``.
    sigma_calibration_ : the global scale factor applied to ``sigma_T``.
    feature_names_in_ : inherited from the mean composite (best effort).
    """

    def __init__(
        self,
        base_estimator: Any = None,
        variance_estimator: Any = None,
        transform_name: str = "linear_residual",
        base_column: str = "",
        alpha: float = 0.1,
        prefer_ngboost: bool = True,
        fallback_predict: str = "y_train_median",
        drop_invalid_rows: bool = True,
        base_columns: Sequence[str] | None = None,
        group_column: str | None = None,
        residual_floor_rel: float = _SQ_RESID_REL_FLOOR,
    ) -> None:
        self.base_estimator = base_estimator
        self.variance_estimator = variance_estimator
        self.transform_name = transform_name
        self.base_column = base_column
        self.alpha = alpha
        self.prefer_ngboost = prefer_ngboost
        self.fallback_predict = fallback_predict
        self.drop_invalid_rows = drop_invalid_rows
        self.base_columns = base_columns
        self.group_column = group_column
        self.residual_floor_rel = residual_floor_rel

    # ------------------------------------------------------------------
    def _make_ngboost_mean(self) -> Any:
        """An ``NGBRegressor`` (Normal dist) to serve as the mean-composite inner, or None when ngboost is absent."""
        if not self.prefer_ngboost:
            return None
        try:  # pragma: no cover - optional dep absent in CI
            from ngboost import NGBRegressor

            return NGBRegressor(verbose=False)
        except ImportError:
            return None

    def fit(self, X: Any, y: Any, sample_weight: np.ndarray | None = None, **fit_kwargs: Any) -> "HeteroscedasticCompositeEstimator":
        """Fit the mean composite plus the predictive-variance head on the T-scale.

        The mean is a :class:`CompositeTargetEstimator`; the variance is either ngboost's native scale head (when the
        inner is an ``NGBRegressor``) or a second regressor on the log squared T-residuals. A global calibration factor is
        then fit on the training residuals. Returns ``self``.
        """
        if self.base_estimator is None and not self.prefer_ngboost:
            raise ValueError("HeteroscedasticCompositeEstimator: base_estimator must not be None (or enable prefer_ngboost).")
        get_transform(self.transform_name)  # surface a typo'd name at fit

        ngb_inner = self._make_ngboost_mean()
        use_ngboost = ngb_inner is not None
        mean_inner = ngb_inner if use_ngboost else self.base_estimator
        if mean_inner is None:
            raise ValueError("HeteroscedasticCompositeEstimator: no mean inner available; pass base_estimator.")

        mean = CompositeTargetEstimator(
            base_estimator=mean_inner,
            transform_name=self.transform_name,
            base_column=self.base_column,
            base_columns=self.base_columns,
            group_column=self.group_column,
            fallback_predict=self.fallback_predict,
            drop_invalid_rows=self.drop_invalid_rows,
        )
        mean.fit(X, y, sample_weight=sample_weight, **fit_kwargs)
        self.mean_estimator_ = mean

        transform = get_transform(self.transform_name)
        params = mean.fitted_params_
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        base = self._base_of(mean, transform, X, n_rows=y_arr.shape[0])
        t_target = np.asarray(transform.forward(y_arr, base, params), dtype=np.float64).reshape(-1)
        t_hat_train = np.asarray(mean.estimator_.predict(X), dtype=np.float64).reshape(-1)

        finite = np.isfinite(t_target) & np.isfinite(t_hat_train)
        resid = t_target - t_hat_train

        self.variance_estimator_ = None
        if use_ngboost:
            self.backend_ = "ngboost"
            sigma_train = self._ngboost_sigma(mean.estimator_, X)
        else:
            self.backend_ = "two_model"
            var_proto = self.variance_estimator if self.variance_estimator is not None else self.base_estimator
            var_inner = clone(var_proto)
            floor = self.residual_floor_rel * max(float(np.var(resid[finite])) if finite.any() else 0.0, 1e-30)
            log_sq = np.log(np.maximum(resid[finite] ** 2, floor))
            X_valid = X if bool(finite.all()) else mean._subset_rows(X, finite)
            var_inner.fit(X_valid, log_sq)
            self.variance_estimator_ = var_inner
            sigma_train = np.exp(0.5 * np.asarray(var_inner.predict(X), dtype=np.float64).reshape(-1))

        self.sigma_calibration_ = self._fit_calibration(resid[finite], sigma_train[finite])

        ref_names = getattr(mean, "feature_names_in_", None)
        if ref_names is not None:
            self.feature_names_in_ = list(ref_names)
        n_feat = getattr(mean, "n_features_in_", None)
        if n_feat is not None:
            self.n_features_in_ = int(n_feat)
        return self

    @staticmethod
    def _base_of(mean: CompositeTargetEstimator, transform: Any, X: Any, n_rows: int) -> np.ndarray:
        """The per-row base vector for the transform forward/inverse (zeros for unary ``requires_base=False`` transforms)."""
        if not transform.requires_base:
            return np.zeros(n_rows, dtype=np.float64)
        base_cols = mean._resolve_base_columns()
        return np.asarray(mean._extract_base_for_transform(X, base_cols), dtype=np.float64)

    @staticmethod
    def _ngboost_sigma(ngb_model: Any, X: Any) -> np.ndarray:  # pragma: no cover - exercised only when ngboost installed
        """Native per-row predictive std of ``T`` from a fitted ``NGBRegressor`` (Normal ``scale`` head)."""
        dist = ngb_model.pred_dist(X)
        params = getattr(dist, "params", None)
        if isinstance(params, dict) and "scale" in params:
            return np.asarray(params["scale"], dtype=np.float64).reshape(-1)
        return np.asarray(getattr(dist, "scale"), dtype=np.float64).reshape(-1)

    @staticmethod
    def _fit_calibration(resid: np.ndarray, sigma: np.ndarray) -> float:
        """Global factor ``c`` so ``resid / (c*sigma)`` has unit variance -- corrects the log-chi-square / scale bias."""
        s = np.asarray(sigma, dtype=np.float64).reshape(-1)
        r = np.asarray(resid, dtype=np.float64).reshape(-1)
        ok = np.isfinite(s) & np.isfinite(r) & (s > 0)
        if not ok.any():
            return 1.0
        c = float(np.sqrt(np.mean((r[ok] / s[ok]) ** 2)))
        return c if np.isfinite(c) and c > 0 else 1.0

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not hasattr(self, "mean_estimator_"):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("HeteroscedasticCompositeEstimator: call fit before this method.")

    def _t_sigma(self, X: Any) -> np.ndarray:
        """Calibrated per-row predictive std of ``T`` at ``X`` (backend-dispatched, always strictly positive)."""
        mean = self.mean_estimator_
        if self.backend_ == "ngboost":  # pragma: no cover - optional dep path
            sigma = self._ngboost_sigma(mean.estimator_, X)
        else:
            sigma = np.exp(0.5 * np.asarray(self.variance_estimator_.predict(X), dtype=np.float64).reshape(-1))
        sigma = self.sigma_calibration_ * sigma
        return np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.finfo(np.float64).tiny)

    def predict(self, X: Any) -> np.ndarray:
        """y-scale point prediction (the mean composite's clipped inverse)."""
        self._check_fitted()
        return self.mean_estimator_.predict(X)

    def _t_bounds(self, X: Any, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(base, T_hat, T_lo, T_hi)`` for the central ``1-alpha`` band on the T-scale."""
        mean = self.mean_estimator_
        transform = get_transform(self.transform_name)
        z = _z_for_alpha(alpha)
        t_hat = np.asarray(mean.estimator_.predict(X), dtype=np.float64).reshape(-1)
        sigma = self._t_sigma(X)
        base = self._base_of(mean, transform, X, n_rows=t_hat.shape[0])
        return base, t_hat, t_hat - z * sigma, t_hat + z * sigma

    def predict_interval(self, X: Any, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Central predictive interval ``(lo, hi)`` on the y-scale at coverage ``1 - alpha`` (default ``self.alpha``).

        The T-scale band ``T_hat +/- z*sigma_T`` is inverted through the shared transform per row; the two inverted
        endpoints are sorted so ``lo <= hi`` even for a sign-flipping (decreasing) inverse.
        """
        self._check_fitted()
        a = self.alpha if alpha is None else alpha
        transform = get_transform(self.transform_name)
        params = self.mean_estimator_.fitted_params_
        base, _t_hat, t_lo, t_hi = self._t_bounds(X, a)
        y_lo = np.asarray(transform.inverse(t_lo, base, params), dtype=np.float64).reshape(-1)
        y_hi = np.asarray(transform.inverse(t_hi, base, params), dtype=np.float64).reshape(-1)
        lo = np.minimum(y_lo, y_hi)
        hi = np.maximum(y_lo, y_hi)
        return lo, hi

    def predict_std(self, X: Any, alpha: float | None = None) -> np.ndarray:
        """Per-row predictive std on the y-scale, read off the inverted band as ``(y_hi - y_lo) / (2 z)``.

        Additive-residual transforms (``linear_residual`` / ``diff``) invert as a pure per-row shift, so this equals the
        T-scale ``sigma_T``; multiplicative transforms (``ratio`` / ``logratio``) widen/narrow it with the base.
        """
        self._check_fitted()
        a = self.alpha if alpha is None else alpha
        z = _z_for_alpha(a)
        lo, hi = self.predict_interval(X, alpha=a)
        return np.asarray((hi - lo) / (2.0 * z))

    def predict_var(self, X: Any, alpha: float | None = None) -> np.ndarray:
        """Per-row predictive variance on the y-scale (square of :meth:`predict_std`)."""
        std = self.predict_std(X, alpha=alpha)
        return np.asarray(std * std)
