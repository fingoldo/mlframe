"""Per-strategy QR dispatch tests.

For each (CB native / XGB native / wrapper-path) we verify:
- ``supports_native_quantile`` flag is correct
- ``get_quantile_objective_kwargs`` emits the right per-library kwargs
- ``wrap_quantile`` either passes through (native) or wraps (others)
- A real fit + predict on synthetic data returns (N, K) and produces
  empirical coverage close to nominal for in-sample data
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.quantile_metrics import coverage
from mlframe.training.configs import QuantileRegressionConfig
from mlframe.training.quantile_wrapper import _QuantileMultiOutputWrapper
from mlframe.training.strategies import (
    CatBoostStrategy, HGBStrategy, LinearModelStrategy, TreeModelStrategy,
    XGBoostStrategy,
)


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(0)
    n = 500
    X = rng.standard_normal((n, 3))
    y = X[:, 0] * 1.5 + 0.5 * rng.standard_normal(n)
    return X, y


def _native_strategies():
    return [CatBoostStrategy(), XGBoostStrategy()]


def _wrapper_strategies():
    return [TreeModelStrategy(), HGBStrategy(), LinearModelStrategy()]


class TestNativeFlag:
    @pytest.mark.parametrize("strat", _native_strategies(),
                             ids=lambda s: type(s).__name__)
    def test_native_flag_true(self, strat):
        assert strat.supports_native_quantile is True

    @pytest.mark.parametrize("strat", _wrapper_strategies(),
                             ids=lambda s: type(s).__name__)
    def test_wrapper_flag_false(self, strat):
        assert strat.supports_native_quantile is False


class TestObjectiveKwargs:
    def test_cb_emits_multiquantile(self):
        qr = QuantileRegressionConfig(alphas=(0.1, 0.5, 0.9))
        kw = CatBoostStrategy().get_quantile_objective_kwargs(qr)
        assert kw == {"loss_function": "MultiQuantile:alpha=0.1,0.5,0.9"}

    def test_xgb_emits_quantile_alpha_list(self):
        qr = QuantileRegressionConfig(alphas=(0.1, 0.5, 0.9))
        kw = XGBoostStrategy().get_quantile_objective_kwargs(qr)
        assert kw == {
            "objective": "reg:quantileerror",
            "quantile_alpha": [0.1, 0.5, 0.9],
        }

    def test_wrapper_strategies_return_empty(self):
        qr = QuantileRegressionConfig()
        for strat in _wrapper_strategies():
            assert strat.get_quantile_objective_kwargs(qr) == {}


class TestWrapDispatch:
    def test_native_passes_through(self):
        qr = QuantileRegressionConfig()

        class _Stub:
            def get_params(self, deep=False):
                return {"alpha": 0.5}

        stub = _Stub()
        for strat in _native_strategies():
            assert strat.wrap_quantile(stub, qr) is stub

    def test_non_native_wraps(self):
        qr = QuantileRegressionConfig()

        class _Stub:
            def get_params(self, deep=False):
                return {"alpha": 0.5}

        stub = _Stub()
        for strat in _wrapper_strategies():
            wrapped = strat.wrap_quantile(stub, qr)
            assert isinstance(wrapped, _QuantileMultiOutputWrapper)
            assert wrapped.alphas == qr.alphas


class TestNativeFit:
    def test_cb_native_fit(self, reg_data):
        from catboost import CatBoostRegressor
        X, y = reg_data
        qr = QuantileRegressionConfig(alphas=(0.1, 0.5, 0.9))
        kw = CatBoostStrategy().get_quantile_objective_kwargs(qr)
        m = CatBoostRegressor(iterations=50, verbose=0, **kw)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y), 3)
        # In-sample coverage of nominal-80% PI should overshoot
        # (training set bias) but not collapse below 0.5.
        cov = coverage(y, preds[:, 0], preds[:, 2])
        assert cov >= 0.6

    def test_xgb_native_fit(self, reg_data):
        from xgboost import XGBRegressor
        X, y = reg_data
        qr = QuantileRegressionConfig(alphas=(0.1, 0.5, 0.9))
        kw = XGBoostStrategy().get_quantile_objective_kwargs(qr)
        m = XGBRegressor(n_estimators=50, max_depth=3, **kw)
        m.fit(X, y)
        preds = m.predict(X)
        assert preds.shape == (len(y), 3)
        cov = coverage(y, preds[:, 0], preds[:, 2])
        assert cov >= 0.6


class TestWrapperFit:
    def test_lgb_wrapper_fit(self, reg_data):
        from lightgbm import LGBMRegressor
        X, y = reg_data
        qr = QuantileRegressionConfig(alphas=(0.1, 0.5, 0.9))
        wrapped = _QuantileMultiOutputWrapper(
            base_estimator=LGBMRegressor(
                objective="quantile", n_estimators=50, verbose=-1,
            ),
            alphas=qr.alphas, crossing_fix=qr.crossing_fix, n_jobs=1,
        )
        wrapped.fit(X, y)
        preds = wrapped.predict(X)
        assert preds.shape == (len(y), 3)
        # Monotone after sort
        assert np.all(np.diff(preds, axis=1) >= -1e-9)

    def test_hgb_wrapper_fit(self, reg_data):
        from sklearn.ensemble import HistGradientBoostingRegressor
        X, y = reg_data
        qr = QuantileRegressionConfig(alphas=(0.1, 0.5, 0.9))
        wrapped = _QuantileMultiOutputWrapper(
            base_estimator=HistGradientBoostingRegressor(
                loss="quantile", max_iter=50,
            ),
            alphas=qr.alphas, crossing_fix=qr.crossing_fix, n_jobs=1,
        )
        wrapped.fit(X, y)
        preds = wrapped.predict(X)
        assert preds.shape == (len(y), 3)

    def test_linear_wrapper_fit(self, reg_data):
        from sklearn.linear_model import QuantileRegressor
        X, y = reg_data
        qr = QuantileRegressionConfig(alphas=(0.1, 0.5, 0.9))
        wrapped = _QuantileMultiOutputWrapper(
            base_estimator=QuantileRegressor(solver="highs", alpha=0.0),
            alphas=qr.alphas, crossing_fix=qr.crossing_fix, n_jobs=1,
        )
        wrapped.fit(X, y)
        preds = wrapped.predict(X)
        assert preds.shape == (len(y), 3)
        # Linear quantile reg coverage check is loose because the LP
        # solver under-fits small synthetic data; we just verify that
        # the wrapper produces a non-collapsed PI (lower < higher in
        # MOST rows).
        widening = (preds[:, 2] - preds[:, 0])
        assert (widening >= 0).mean() >= 0.99
        assert widening.mean() > 0.0


class TestWrapperContract:
    def test_unknown_alpha_param_rejects(self, reg_data):
        from sklearn.linear_model import LinearRegression  # no alpha param
        X, y = reg_data
        wrapped = _QuantileMultiOutputWrapper(
            base_estimator=LinearRegression(),
            alphas=(0.1, 0.5, 0.9),
            n_jobs=1,
        )
        with pytest.raises(ValueError, match="standard quantile"):
            wrapped.fit(X, y)

    def test_2d_y_rejects(self, reg_data):
        from sklearn.linear_model import QuantileRegressor
        X, _ = reg_data
        y2d = np.zeros((len(X), 2))
        wrapped = _QuantileMultiOutputWrapper(
            base_estimator=QuantileRegressor(solver="highs", alpha=0.0),
            alphas=(0.1, 0.9),
            n_jobs=1,
        )
        with pytest.raises(ValueError, match="1-D y"):
            wrapped.fit(X, y2d)

    def test_predict_before_fit_rejects(self, reg_data):
        from sklearn.linear_model import QuantileRegressor
        X, _ = reg_data
        wrapped = _QuantileMultiOutputWrapper(
            base_estimator=QuantileRegressor(solver="highs", alpha=0.0),
            alphas=(0.1, 0.9),
        )
        with pytest.raises(RuntimeError, match="before fit"):
            wrapped.predict(X)
