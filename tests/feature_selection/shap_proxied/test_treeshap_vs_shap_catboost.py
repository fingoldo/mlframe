"""CatBoost through mlframe's TreeExplainer against the public shap library and CatBoost's own native SHAP: values, base, additivity, categorical features, tree growth policies, interactions, fallbacks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

shap = pytest.importorskip("shap")
cb = pytest.importorskip("catboost")

from mlframe.inference.tree_shap import TreeExplainer


def _data(n: int = 300, f: int = 6, seed: int = 0, nan_share: float = 0.0):
    """A numeric frame (optionally with NaN), a numeric target with an interaction and a binary target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, f)).astype(np.float32), columns=[f"c{i}" for i in range(f)])
    signal = 2 * X["c0"] + X["c1"] * X["c2"] - np.sin(X["c3"])
    y_reg = signal.to_numpy() + 0.1 * rng.normal(size=n)
    y_bin = (signal.to_numpy() + 0.3 * rng.normal(size=n) > 0).astype(int)
    if nan_share:
        X = X.mask(rng.random(X.shape) < nan_share)
    return X, y_reg, y_bin


def _fit(kind: str, X, y, **params):
    """A small fitted CatBoost model."""
    base = {"iterations": 40, "depth": 4, "verbose": 0, "random_seed": 0, "thread_count": 2, "allow_writing_files": False}
    base.update(params)
    cls = cb.CatBoostRegressor if kind == "reg" else cb.CatBoostClassifier
    return cls(**base).fit(X, y)


def _native(model, X, cat_features=None):
    """CatBoost's own SHAP matrix and bias."""
    out = np.asarray(model.get_feature_importance(cb.Pool(X, cat_features=cat_features), type="ShapValues"), dtype=np.float64)
    return out[:, :-1], float(out[0, -1])


@pytest.mark.parametrize("kind", ["reg", "bin"])
def test_values_equal_catboost_native_and_the_public_library(kind):
    """The explainer returns CatBoost's native values (so also the public library's), answers with the catboost backend, and the values add up to the raw prediction."""
    X, y_reg, y_bin = _data()
    model = _fit(kind, X, y_reg if kind == "reg" else y_bin)
    explainer = TreeExplainer(model)
    got = explainer.shap_values(X)
    ref, base = _native(model, X)
    assert explainer.backend_ == "catboost"
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)
    assert float(np.asarray(explainer.expected_value).ravel()[0]) == pytest.approx(base, abs=1e-9)
    public = np.asarray(shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(X, check_additivity=False))
    np.testing.assert_allclose(got, public, rtol=0, atol=1e-6)
    raw = model.predict(X, prediction_type="RawFormulaVal")
    np.testing.assert_allclose(base + got.sum(axis=1), raw, rtol=0, atol=1e-6)
    assert list(np.argsort(-np.abs(got).mean(0))[:3]) == list(np.argsort(-np.abs(public).mean(0))[:3])


@pytest.mark.parametrize("params", [{"depth": 1}, {"depth": 8, "iterations": 20}, {"boosting_type": "Ordered", "iterations": 20}])
def test_symmetric_tree_shapes(params):
    """Stumps, deep trees and ordered boosting (symmetric trees) keep native agreement and additivity."""
    X, y_reg, _ = _data(seed=1)
    model = _fit("reg", X, y_reg, **params)
    got = TreeExplainer(model).shap_values(X)
    ref, base = _native(model, X)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)
    np.testing.assert_allclose(base + got.sum(axis=1), model.predict(X), rtol=0, atol=1e-5)


@pytest.mark.parametrize("params", [{"grow_policy": "Depthwise", "depth": 5}, {"grow_policy": "Lossguide", "max_leaves": 12}, {"grow_policy": "Depthwise", "depth": 3, "iterations": 5}])
def test_nonsymmetric_trees_are_additive(params):
    """CatBoost's native SHAP of Depthwise / Lossguide trees does not add up to the prediction (0.9 / 2.5 off on single rows); our TreeSHAP over the JSON export does, and keeps the native base value."""
    X, y_reg, _ = _data(seed=1)
    model = _fit("reg", X, y_reg, **params)
    explainer = TreeExplainer(model)
    got = explainer.shap_values(X)
    assert explainer.backend_ == "treeshap_numba"
    base = float(np.asarray(explainer.expected_value).ravel()[0])
    np.testing.assert_allclose(base + got.sum(axis=1), model.predict(X), rtol=0, atol=1e-9)
    ref, _ = _native(model, X)
    assert np.corrcoef(got.ravel(), ref.ravel())[0, 1] > 0.9


def test_nonsymmetric_binary_and_nan():
    """A Lossguide binary classifier with NaN inputs adds up to its raw margin."""
    X, _, y_bin = _data(seed=8, nan_share=0.15)
    model = _fit("bin", X, y_bin, grow_policy="Lossguide", max_leaves=10)
    explainer = TreeExplainer(model)
    got = explainer.shap_values(X)
    base = float(np.asarray(explainer.expected_value).ravel()[0])
    np.testing.assert_allclose(base + got.sum(axis=1), model.predict(X, prediction_type="RawFormulaVal"), rtol=0, atol=1e-8)


def test_categorical_features_are_passed_to_the_pool():
    """A string column declared in cat_features: the explainer builds the right Pool and matches CatBoost's native values."""
    X, y_reg, _ = _data(seed=2)
    X["colour"] = np.where(X["c4"] > 0, "red", "blue")
    X["shape"] = np.where(X["c5"] > 0.5, "square", np.where(X["c5"] < -0.5, "circle", "other"))
    y = y_reg + np.where(X["colour"] == "red", 1.0, 0.0)
    cat = ["colour", "shape"]
    model = _fit("reg", X, y, cat_features=cat)
    explainer = TreeExplainer(model)
    got = explainer.shap_values(X)
    ref, base = _native(model, X, cat_features=cat)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)
    np.testing.assert_allclose(base + got.sum(axis=1), model.predict(X), rtol=0, atol=1e-5)


def test_missing_values():
    """NaN in the numeric features follows CatBoost's own missing-value handling in both."""
    X, y_reg, _ = _data(seed=3, nan_share=0.2)
    model = _fit("reg", X, y_reg, nan_mode="Min")
    got = TreeExplainer(model).shap_values(X)
    ref, _ = _native(model, X)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-9)
    assert np.isfinite(got).all()


def test_one_row_and_polars_input():
    """A single row and a polars frame both work."""
    pl = pytest.importorskip("polars")
    X, y_reg, _ = _data(seed=4)
    model = _fit("reg", X, y_reg)
    explainer = TreeExplainer(model)
    assert explainer.shap_values(X.iloc[:1]).shape == (1, X.shape[1])
    np.testing.assert_allclose(explainer.shap_values(pl.from_pandas(X)), explainer.shap_values(X), rtol=0, atol=1e-12)


def test_multiclass_is_answered_by_the_public_library():
    """CatBoost multiclass has several outputs per row: out of scope for the single-output path, the public library answers and the structure is its own."""
    X, _, y_bin = _data(seed=5)
    y3 = y_bin + (X["c4"].to_numpy() > 0.5)
    model = _fit("bin", X, y3, loss_function="MultiClass", iterations=15)
    explainer = TreeExplainer(model)
    values = explainer.shap_values(X.iloc[:40])
    assert explainer.backend_ == "shap"
    assert (isinstance(values, list) and len(values) == 3) or np.asarray(values).ndim == 3


def test_interaction_values_equal_catboost_native():
    """Interaction tensor from CatBoost's native kernel: symmetric, row sums equal the SHAP values, shape (n, f, f)."""
    X, y_reg, _ = _data(n=120, f=5, seed=6)
    model = _fit("reg", X, y_reg, iterations=25)
    explainer = TreeExplainer(model)
    Phi = explainer.shap_interaction_values(X)
    assert explainer.backend_ == "catboost" and Phi.shape == (120, 5, 5)
    ref = np.asarray(model.get_feature_importance(cb.Pool(X), type="ShapInteractionValues"), dtype=np.float64)[:, :-1, :-1]
    np.testing.assert_allclose(Phi, ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(Phi, np.swapaxes(Phi, 1, 2), rtol=0, atol=1e-9)
    np.testing.assert_allclose(Phi.sum(axis=2), TreeExplainer(model).shap_values(X), rtol=0, atol=1e-6)


def test_the_explanation_wrapper_carries_catboost_values():
    """``explain`` returns a plot-ready Explanation with the same values and a per-row base."""
    X, y_reg, _ = _data(seed=7)
    model = _fit("reg", X, y_reg)
    exp = TreeExplainer(model).explain(X)
    ref, base = _native(model, X)
    np.testing.assert_allclose(exp.values, ref, rtol=0, atol=1e-9)
    np.testing.assert_allclose(exp.base_values, base, rtol=0, atol=1e-9)
