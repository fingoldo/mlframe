"""F-34 smoke tests: per-strategy native multi-target regression on
small synthetic (N, K) data. Verifies the get_multi_target_objective_kwargs
+ wrap_multi_target dispatch produces a working regressor end-to-end.

Each test constructs the regressor manually using the strategy's
canonical kwargs (the suite-integration that auto-wires this at
fit-time is a separate phase; this test pins the per-strategy
building blocks).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _make_mtr_data(n=120, d=5, k=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    coefs = rng.normal(size=(d, k)).astype(np.float32)
    y = (X @ coefs + 0.05 * rng.normal(size=(n, k))).astype(np.float32)
    return X, y


def test_sklearn_linear_native_multi_target_works():
    """LinearModelStrategy declares native MTR; sklearn LinearRegression
    accepts (N, K) y out-of-the-box."""
    from sklearn.linear_model import LinearRegression

    from mlframe.training.strategies import LinearModelStrategy
    X, y = _make_mtr_data()
    strat = LinearModelStrategy()
    assert strat.supports_native_multi_target is True
    assert strat.get_multi_target_objective_kwargs() == {}

    # Native means wrap_multi_target is a no-op; estimator handles (N, K).
    est = strat.wrap_multi_target(LinearRegression())
    assert isinstance(est, LinearRegression)
    est.fit(X, y)
    preds = est.predict(X)
    assert preds.shape == (X.shape[0], y.shape[1])


def test_catboost_multirmse_native_multi_target_works():
    """CatBoost MultiRMSE: single ensemble outputs (N, K)."""
    catboost = pytest.importorskip("catboost")
    from mlframe.training.strategies import CatBoostStrategy

    X, y = _make_mtr_data()
    strat = CatBoostStrategy()
    kwargs = strat.get_multi_target_objective_kwargs()
    assert kwargs == {"loss_function": "MultiRMSE"}

    # CatBoost MultiRMSE requires the loss_function kwarg; 100 iterations
    # is enough to converge on a clean linear synthetic target (tree
    # ensemble needs more depth than LinearRegression on linear data).
    est = catboost.CatBoostRegressor(
        iterations=100, learning_rate=0.1, verbose=False, **kwargs,
    )
    est = strat.wrap_multi_target(est)  # native -> identity
    est.fit(X, y)
    preds = est.predict(X)
    assert preds.shape == (X.shape[0], y.shape[1])
    # MultiRMSE should reach a non-trivial fit on this clean target.
    from sklearn.metrics import r2_score
    r2 = r2_score(y, preds, multioutput="uniform_average")
    assert r2 > 0.5, f"CatBoost MultiRMSE R^2={r2:+.4f}; expected >0.5"


def test_lightgbm_wrap_multi_target_works():
    """LightGBM has no native MTR; the wrapper routes through
    sklearn.multioutput.MultiOutputRegressor (K independent fits)."""
    lgb = pytest.importorskip("lightgbm")
    from sklearn.multioutput import MultiOutputRegressor

    from mlframe.training.strategies import TreeModelStrategy

    X, y = _make_mtr_data()
    strat = TreeModelStrategy()  # the LGB strategy base — no native MTR
    assert strat.supports_native_multi_target is False
    assert strat.get_multi_target_objective_kwargs() == {}

    base = lgb.LGBMRegressor(n_estimators=20, verbose=-1)
    wrapped = strat.wrap_multi_target(base)
    assert isinstance(wrapped, MultiOutputRegressor)

    wrapped.fit(X, y)
    preds = wrapped.predict(X)
    assert preds.shape == (X.shape[0], y.shape[1])


def test_xgboost_multi_output_tree_native_works():
    """XGBoost native multi-output trees: single ensemble outputs (N, K)."""
    xgboost = pytest.importorskip("xgboost")
    # XGBoost multi_output_tree requires >=2.0
    from packaging.version import Version
    if Version(xgboost.__version__) < Version("2.0"):
        pytest.skip(f"XGBoost {xgboost.__version__} < 2.0; native MTR unavailable")

    from mlframe.training.strategies import XGBoostStrategy

    X, y = _make_mtr_data()
    strat = XGBoostStrategy()
    kwargs = strat.get_multi_target_objective_kwargs()
    assert kwargs == {"multi_strategy": "multi_output_tree", "tree_method": "hist"}

    est = xgboost.XGBRegressor(n_estimators=20, verbosity=0, **kwargs)
    est = strat.wrap_multi_target(est)  # native -> identity
    est.fit(X, y)
    preds = est.predict(X)
    assert preds.shape == (X.shape[0], y.shape[1])


def test_mlframe_mlp_via_strategy_metadata_only():
    """MLP doesn't need any extra kwargs at the strategy level —
    PytorchLightningRegressor auto-detects (N, K) y at fit-time
    (F-24 commit 2d300944). Strategy just declares native support
    so the suite skips the MultiOutputRegressor wrap."""
    from mlframe.training.strategies import NeuralNetStrategy

    strat = NeuralNetStrategy()
    assert strat.supports_native_multi_target is True
    # No extra kwargs needed; the MLP estimator handles K-output internally.
    assert strat.get_multi_target_objective_kwargs() == {}
