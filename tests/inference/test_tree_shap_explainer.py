"""mlframe's TreeExplainer: our TreeSHAP answers wherever it applies (same values as the public shap library), the public library answers the rest and the fallback is logged."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

shap = pytest.importorskip("shap")
xgb = pytest.importorskip("xgboost")
lgb = pytest.importorskip("lightgbm")

from mlframe.inference import tree_shap as ts


def _data(n: int = 400, f: int = 8, seed: int = 0):
    """A frame and a numeric target with an interaction."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, f)).astype(np.float32), columns=[f"c{i}" for i in range(f)])
    signal = 2 * X["c0"] + X["c1"] * X["c2"] - np.sin(X["c3"])
    return X, signal.to_numpy() + 0.1 * rng.normal(size=n), (signal.to_numpy() + 0.3 * rng.normal(size=n) > 0).astype(int)


def _public(model, X):
    """The public library's values and base for the same model."""
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    return np.asarray(explainer.shap_values(X, check_additivity=False)), explainer.expected_value


@pytest.mark.parametrize("family", ["xgb_reg", "xgb_bin", "lgb_reg", "lgb_bin"])
def test_values_match_the_public_library(family):
    """Same values (1e-9 for LightGBM, 1e-5 for XGBoost), same base, same feature rank as ``shap.TreeExplainer``; our backend answered."""
    X, y_reg, y_bin = _data()
    models = {
        "xgb_reg": lambda: xgb.XGBRegressor(n_estimators=40, max_depth=4, tree_method="hist", verbosity=0).fit(X, y_reg),
        "xgb_bin": lambda: xgb.XGBClassifier(n_estimators=40, max_depth=4, tree_method="hist", verbosity=0).fit(X, y_bin),
        "lgb_reg": lambda: lgb.LGBMRegressor(n_estimators=40, num_leaves=16, verbose=-1, n_jobs=2).fit(X, y_reg),
        "lgb_bin": lambda: lgb.LGBMClassifier(n_estimators=40, num_leaves=16, verbose=-1, n_jobs=2).fit(X, y_bin),
    }
    model = models[family]()
    explainer = ts.TreeExplainer(model)
    got = explainer.shap_values(X)
    ref, ref_base = _public(model, X)
    assert explainer.backend_ in ("treeshap_numba", "treeshap_gpu")
    # XGBoost keeps leaf values and covers in float32, which the public loader and our extraction read slightly differently (~1e-6); LightGBM agrees to machine precision on the CPU
    atol = (1e-5 if family.startswith("xgb") else 1e-9) if explainer.backend_ == "treeshap_numba" else 1e-4
    np.testing.assert_allclose(got, ref, rtol=0, atol=atol)
    assert float(np.asarray(explainer.expected_value).ravel()[0]) == pytest.approx(float(np.asarray(ref_base).ravel()[0]), abs=1e-6)
    assert list(np.argsort(-np.abs(got).mean(0))[:3]) == list(np.argsort(-np.abs(ref).mean(0))[:3])


def test_forcing_the_public_backend_returns_the_public_values():
    """backend="shap" is honoured."""
    X, y_reg, _ = _data()
    model = lgb.LGBMRegressor(n_estimators=20, num_leaves=8, verbose=-1, n_jobs=2).fit(X, y_reg)
    explainer = ts.TreeExplainer(model, backend="shap")
    got = explainer.shap_values(X)
    assert explainer.backend_ == "shap"
    np.testing.assert_allclose(got, _public(model, X)[0], rtol=0, atol=1e-12)


def test_multiclass_is_answered_by_the_public_library_and_logged(caplog):
    """A multiclass model is out of scope for our single-output kernels: the public library answers with its own structure, and the fallback is logged once."""
    X, _, y_bin = _data()
    y3 = y_bin + (X["c4"].to_numpy() > 0.5)
    model = lgb.LGBMClassifier(n_estimators=15, num_leaves=8, verbose=-1, n_jobs=2).fit(X, y3)
    ts._FALLBACK_LOGGED.clear()
    with caplog.at_level(logging.WARNING, logger=ts.logger.name):
        explainer = ts.TreeExplainer(model)
        values = explainer.shap_values(X.iloc[:50])
    assert explainer.backend_ == "shap"
    assert (isinstance(values, list) and len(values) == 3) or np.asarray(values).ndim == 3
    assert any("answered by the public shap library" in r.getMessage() for r in caplog.records)


def test_a_sklearn_forest_is_answered_by_the_public_library(caplog):
    """A family we do not implement falls back and says so."""
    from sklearn.ensemble import RandomForestRegressor

    X, y_reg, _ = _data(n=200)
    model = RandomForestRegressor(n_estimators=8, max_depth=4, random_state=0).fit(X, y_reg)
    ts._FALLBACK_LOGGED.clear()
    with caplog.at_level(logging.WARNING, logger=ts.logger.name):
        explainer = ts.TreeExplainer(model)
        values = explainer.shap_values(X.iloc[:30])
    assert explainer.backend_ == "shap" and np.asarray(values).shape == (30, X.shape[1])
    assert any("RandomForestRegressor" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("family", ["lgb", "xgb"])
def test_interaction_values_match_the_public_library(family):
    """Interactions: our numba kernel (LightGBM) or XGBoost's native predict (XGBoost) against ``shap_interaction_values``; symmetric, and the row sums are the SHAP values."""
    X, y_reg, _ = _data(n=150, f=6)
    if family == "lgb":
        model = lgb.LGBMRegressor(n_estimators=20, num_leaves=8, verbose=-1, n_jobs=2).fit(X, y_reg)
    else:
        model = xgb.XGBRegressor(n_estimators=20, max_depth=3, tree_method="hist", verbosity=0).fit(X, y_reg)
    explainer = ts.TreeExplainer(model)
    got = explainer.shap_interaction_values(X)
    ref = np.asarray(shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_interaction_values(X))
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-4)
    np.testing.assert_allclose(got, np.swapaxes(got, 1, 2), rtol=0, atol=1e-5 if family == "xgb" else 1e-9)  # XGBoost's native tensor is symmetric to its float32 precision
    np.testing.assert_allclose(got.sum(axis=2), ts.TreeExplainer(model).shap_values(X), rtol=0, atol=1e-4)


def test_pipeline_and_target_transform_wrappers_are_unwrapped():
    """A fitted TransformedTargetRegressor around a booster is explained through the booster."""
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.preprocessing import StandardScaler

    X, y_reg, _ = _data(n=200)
    wrapped = TransformedTargetRegressor(regressor=lgb.LGBMRegressor(n_estimators=15, num_leaves=8, verbose=-1, n_jobs=2), transformer=StandardScaler()).fit(X, y_reg)
    explainer = ts.TreeExplainer(wrapped)
    got = explainer.shap_values(X.iloc[:40])
    assert explainer.backend_ in ("treeshap_numba", "treeshap_gpu") and got.shape == (40, X.shape[1])


def test_to_explanation_carries_values_data_and_names():
    """The public plotting functions get an Explanation built from our values."""
    X, y_reg, _ = _data(n=100)
    model = lgb.LGBMRegressor(n_estimators=10, num_leaves=8, verbose=-1, n_jobs=2).fit(X, y_reg)
    explainer = ts.TreeExplainer(model)
    values = explainer.shap_values(X)
    exp = ts.to_explanation(values, X, base_values=explainer.expected_value)
    assert exp.values.shape == values.shape and list(exp.feature_names) == list(X.columns) and exp.data.shape == X.shape
