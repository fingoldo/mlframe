"""Contract / unit tests for ``CompositeQuantileEstimator``.

Shape, non-crossing, NaN-safety, subset-quantile predict, fitted-quantile guard,
the inner-alpha wiring, and the unfitted / bad-config error paths. Uses a tiny
deterministic stub inner so the suite is fast and does not depend on LightGBM
being installed for the pure-contract checks (a separate LightGBM-gated test
covers the real pinball path).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import NotFittedError

from mlframe.training.composite import CompositeQuantileEstimator
from mlframe.training.composite.transforms import UnknownTransformError
from mlframe.training.composite.quantile import _set_inner_quantile_alpha


class _StubQuantileInner(BaseEstimator, RegressorMixin):
    """Deterministic pinball stub: predicts ``mean(T_train) + offset(alpha)``.

    ``alpha`` is exposed as a constructor param so sklearn ``clone`` /
    ``set_params`` round-trip it. The per-alpha offset (a monotone function of
    alpha) makes the heads emit ASCENDING quantiles, mimicking a real pinball
    learner without any training cost.
    """

    def __init__(self, alpha: float = 0.5, scale: float = 1.0) -> None:
        self.alpha = alpha
        self.scale = scale

    def fit(self, X, y, sample_weight=None):  # noqa: ARG002
        self._mean_ = float(np.mean(np.asarray(y, dtype=np.float64)))
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else len(X.columns)
        return self

    def predict(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        # Map alpha in (0,1) to a symmetric quantile offset via the logit-ish
        # scale; monotone increasing in alpha -> ascending heads.
        offset = self.scale * (float(self.alpha) - 0.5) * 4.0
        return np.full(n, self._mean_ + offset, dtype=np.float64)


def _make_xy(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, n)
    x1 = rng.normal(0.0, 1.0, n)
    y = 2.0 * base + 0.5 * x1 + rng.normal(0.0, 1.0, n)
    return pd.DataFrame({"base": base, "x1": x1}), y


def _fit_stub(quantiles=(0.1, 0.5, 0.9), transform_name="linear_residual", **kw):
    X, y = _make_xy()
    est = CompositeQuantileEstimator(
        base_estimator=_StubQuantileInner(),
        transform_name=transform_name,
        base_column="base",
        quantiles=quantiles,
        **kw,
    )
    est.fit(X, y)
    return est, X, y


# ----------------------------------------------------------------------
# Shape + non-crossing contract
# ----------------------------------------------------------------------

def test_predict_quantile_shape():
    est, X, _ = _fit_stub(quantiles=(0.1, 0.25, 0.5, 0.75, 0.9))
    Q = est.predict_quantile(X)
    assert Q.shape == (len(X), 5)


def test_non_crossing_rows_monotone():
    est, X, _ = _fit_stub(quantiles=(0.1, 0.25, 0.5, 0.75, 0.9))
    Q = est.predict_quantile(X)
    assert np.all(np.diff(Q, axis=1) >= -1e-9), "rows must be non-decreasing across quantiles"


def test_non_crossing_restores_order_when_heads_cross():
    """A stub whose offset DECREASES in alpha produces crossing heads; the
    per-row sort must restore ascending order."""
    class _Inverted(_StubQuantileInner):
        def predict(self, X):
            n = X.shape[0] if hasattr(X, "shape") else len(X)
            # DECREASING in alpha -> high quantile predicts LOWER -> crossing.
            offset = -self.scale * (float(self.alpha) - 0.5) * 4.0
            return np.full(n, self._mean_ + offset, dtype=np.float64)

    X, y = _make_xy()
    est = CompositeQuantileEstimator(
        base_estimator=_Inverted(), transform_name="linear_residual",
        base_column="base", quantiles=(0.1, 0.5, 0.9),
        enforce_non_crossing=False,
    )
    est.fit(X, y)
    Q_raw = est.predict_quantile(X)
    # Raw (un-sorted) heads cross: q0.1 column > q0.9 column.
    assert np.any(np.diff(Q_raw, axis=1) < 0)

    est_on = CompositeQuantileEstimator(
        base_estimator=_Inverted(), transform_name="linear_residual",
        base_column="base", quantiles=(0.1, 0.5, 0.9),
        enforce_non_crossing=True,
    )
    est_on.fit(X, y)
    Q_on = est_on.predict_quantile(X)
    assert np.all(np.diff(Q_on, axis=1) >= -1e-9)


# ----------------------------------------------------------------------
# Subset-quantile predict + fitted-quantile guard
# ----------------------------------------------------------------------

def test_predict_subset_of_fitted_quantiles():
    est, X, _ = _fit_stub(quantiles=(0.1, 0.25, 0.5, 0.75, 0.9))
    Q = est.predict_quantile(X, quantiles=(0.25, 0.75))
    assert Q.shape == (len(X), 2)


def test_predict_unfitted_quantile_raises():
    est, X, _ = _fit_stub(quantiles=(0.1, 0.5, 0.9))
    with pytest.raises(ValueError, match="was not fitted"):
        est.predict_quantile(X, quantiles=(0.42,))


def test_predict_median_point():
    est, X, _ = _fit_stub(quantiles=(0.1, 0.5, 0.9))
    p = est.predict(X)
    q = est.predict_quantile(X, quantiles=(0.5,))[:, 0]
    np.testing.assert_allclose(p, q)


# ----------------------------------------------------------------------
# NaN-safety
# ----------------------------------------------------------------------

def test_nan_safe_out_of_domain_base_routes_to_fallback():
    """A predict-time row with a non-finite base must NOT yield NaN under the
    default median fallback; every output is finite."""
    est, X, _ = _fit_stub(quantiles=(0.1, 0.5, 0.9), transform_name="ratio")
    Xp = X.copy()
    Xp.loc[0, "base"] = 0.0  # ratio: |base| == 0 is out of domain.
    Xp.loc[1, "base"] = np.inf
    Q = est.predict_quantile(Xp)
    assert np.all(np.isfinite(Q)), "default fallback must keep every quantile finite"


# ----------------------------------------------------------------------
# Inner-alpha wiring
# ----------------------------------------------------------------------

def test_set_inner_quantile_alpha_sklearn_quantile_regressor():
    from sklearn.linear_model import QuantileRegressor

    inner = _set_inner_quantile_alpha(QuantileRegressor(), 0.3)
    assert inner.get_params()["quantile"] == 0.3


def test_set_inner_quantile_alpha_gbr_loss_quantile():
    from sklearn.ensemble import GradientBoostingRegressor

    inner = _set_inner_quantile_alpha(GradientBoostingRegressor(), 0.7)
    p = inner.get_params()
    assert p["loss"] == "quantile" and p["alpha"] == 0.7


def test_set_inner_quantile_alpha_rejects_non_quantile_estimator():
    from sklearn.linear_model import LinearRegression

    with pytest.raises(ValueError, match="no recognised quantile knob"):
        _set_inner_quantile_alpha(LinearRegression(), 0.5)


def test_each_head_has_distinct_alpha():
    est, _, _ = _fit_stub(quantiles=(0.1, 0.5, 0.9))
    alphas = {q: head.base_estimator.alpha for q, head in est.estimators_.items()}
    assert alphas == {0.1: 0.1, 0.5: 0.5, 0.9: 0.9}


# ----------------------------------------------------------------------
# Config / lifecycle error paths
# ----------------------------------------------------------------------

def test_predict_before_fit_raises():
    est = CompositeQuantileEstimator(base_estimator=_StubQuantileInner())
    with pytest.raises(NotFittedError):
        est.predict_quantile(pd.DataFrame({"base": [1.0]}))


def test_none_base_estimator_raises():
    est = CompositeQuantileEstimator(base_estimator=None, base_column="base")
    with pytest.raises(ValueError, match="base_estimator must not be None"):
        est.fit(*_make_xy())


def test_bad_quantiles_raise():
    X, y = _make_xy()
    with pytest.raises(ValueError, match="strictly in"):
        CompositeQuantileEstimator(
            base_estimator=_StubQuantileInner(), base_column="base",
            quantiles=(0.0, 0.5),
        ).fit(X, y)
    with pytest.raises(ValueError, match="unique"):
        CompositeQuantileEstimator(
            base_estimator=_StubQuantileInner(), base_column="base",
            quantiles=(0.5, 0.5),
        ).fit(X, y)


def test_unknown_transform_raises_at_fit():
    X, y = _make_xy()
    with pytest.raises(UnknownTransformError):
        CompositeQuantileEstimator(
            base_estimator=_StubQuantileInner(), base_column="base",
            transform_name="not_a_transform",
        ).fit(X, y)


def test_default_quantiles_five_level_grid():
    X, y = _make_xy()
    est = CompositeQuantileEstimator(
        base_estimator=_StubQuantileInner(), base_column="base",
    )
    est.fit(X, y)
    np.testing.assert_allclose(est.quantiles_, [0.1, 0.25, 0.5, 0.75, 0.9])


def test_sklearn_clone_roundtrip():
    """The estimator must clone cleanly (params-only) before fit."""
    est = CompositeQuantileEstimator(
        base_estimator=_StubQuantileInner(), base_column="base",
        quantiles=(0.2, 0.8),
    )
    cloned = clone(est)
    assert cloned.quantiles == (0.2, 0.8)
    assert cloned.base_column == "base"
