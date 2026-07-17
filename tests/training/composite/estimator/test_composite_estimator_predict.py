"""Regression sensors for the 2026-06-10 composite-estimator predict audit.

- E1 (P0): multi-base domain-violation fallback crashed -- np.where broadcast
  a (n,) row mask against the (n,K) base matrix (ValueError for K>=2).
- E3: predict_quantile crashed for every requires_base=False (unary)
  transform because it extracted a base column unconditionally.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.training.composite import CompositeTargetEstimator


class _ConstInner(BaseEstimator, RegressorMixin):
    """Minimal inner: predicts a fixed T-scale value; optional quantile head."""

    def __init__(self, t_value: float = 0.5, with_quantile: bool = False):
        self.t_value = t_value
        self.with_quantile = with_quantile

    def fit(self, X, y, **kw):
        """Fit."""
        self.n_features_in_ = X.shape[1]
        self._mean_t = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X):
        """Predict."""
        n = X.shape[0]
        return np.full(n, self._mean_t, dtype=np.float64)

    def predict_quantile(self, X, alpha=0.5):
        """Predict quantile."""
        n = X.shape[0]
        if np.isscalar(alpha):
            return np.full(n, self._mean_t, dtype=np.float64)
        return np.column_stack([np.full(n, self._mean_t + float(a) - 0.5) for a in alpha])


def _make_multibase_frame(n=200, seed=0):
    """Make multibase frame."""
    rng = np.random.default_rng(seed)
    b1 = rng.normal(0.0, 1.0, size=n)
    b2 = rng.normal(0.0, 5.0, size=n)
    feat = rng.normal(0.0, 1.0, size=n)
    y = 1.0 * b1 + 0.5 * b2 + 0.3 * feat + rng.normal(0.0, 0.1, size=n)
    X = pd.DataFrame({"b1": b1, "b2": b2, "feat": feat})
    return X, y


class TestMultiBaseDomainFallback:
    """Groups tests covering multi base domain fallback."""
    def test_predict_with_nonfinite_base_does_not_crash(self) -> None:
        """E1: a non-finite cell in a multi-base column at predict-time must
        route through the domain fallback, not raise a broadcast ValueError."""
        X, y = _make_multibase_frame()
        est = CompositeTargetEstimator(
            base_estimator=_ConstInner(),
            transform_name="linear_residual_multi",
            base_columns=["b1", "b2"],
            fallback_predict="y_train_median",
        )
        est.fit(X, y)
        X_pred = X.copy()
        X_pred.loc[X_pred.index[:5], "b2"] = np.inf  # force domain violation
        y_hat = est.predict(X_pred)  # pre-fix: ValueError (n,) vs (n,2)
        assert y_hat.shape == (len(X_pred),)
        assert np.all(np.isfinite(y_hat)), "fallback rows must be finite"

    def test_all_valid_multibase_predict_ok(self) -> None:
        """All valid multibase predict ok."""
        X, y = _make_multibase_frame()
        est = CompositeTargetEstimator(
            base_estimator=_ConstInner(),
            transform_name="linear_residual_multi",
            base_columns=["b1", "b2"],
        )
        est.fit(X, y)
        y_hat = est.predict(X)
        assert y_hat.shape == (len(X),)
        assert np.all(np.isfinite(y_hat))


class TestPredictQuantileUnary:
    """Groups tests covering predict quantile unary."""
    @pytest.mark.parametrize("alpha", [0.5, [0.1, 0.5, 0.9]])
    def test_predict_quantile_unary_transform(self, alpha) -> None:
        """E3: predict_quantile must work for a unary (requires_base=False)
        transform. Pre-fix it raised 'base_columns is empty'."""
        rng = np.random.default_rng(1)
        n = 300
        feat = rng.normal(0.0, 1.0, size=n)
        y = np.abs(rng.normal(5.0, 2.0, size=n)) + 0.5 * feat
        X = pd.DataFrame({"feat": feat})
        est = CompositeTargetEstimator(
            base_estimator=_ConstInner(with_quantile=True),
            transform_name="yeo_johnson_y",  # unary, requires_base=False
            base_column="",
        )
        est.fit(X, y)
        out = est.predict_quantile(X, alpha)  # pre-fix: ValueError
        if np.isscalar(alpha):
            assert out.shape == (n,)
        else:
            assert out.shape == (n, len(alpha))
        assert np.all(np.isfinite(out))
