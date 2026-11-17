"""Regression tests of the three disagreements found by the test-coverage sweep against the public shap library: XGBoost DART weights, LightGBM linear trees, LightGBM values on the zero-bin boundary."""

from __future__ import annotations

import numpy as np
import pytest

shap = pytest.importorskip("shap")
xgb = pytest.importorskip("xgboost")
lgb = pytest.importorskip("lightgbm")

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import (
    _LIGHTGBM_ZERO_THRESHOLD,
    extract_ensemble,
    is_supported_lightgbm,
    treeshap_phi_base_numba,
    treeshap_phi_base_path,
)


def _frame(n: int = 300, f: int = 6, seed: int = 0):
    """Features and a numeric target with an interaction."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f)).astype(np.float32)
    return X, 2 * X[:, 0] + X[:, 1] * X[:, 2] - np.sin(X[:, 3]) + 0.2 * rng.normal(size=n)


def test_xgboost_dart_weights_are_applied():
    """A DART booster's prediction is the weight_drop-weighted sum of its trees; the SHAP values must add up to that margin and equal the public library's."""
    X, y = _frame()
    model = xgb.XGBRegressor(n_estimators=15, max_depth=4, booster="dart", rate_drop=0.3, verbosity=0, n_jobs=1).fit(X, y)
    margin = model.get_booster().predict(xgb.DMatrix(X), output_margin=True)
    for fn in (treeshap_phi_base_numba, treeshap_phi_base_path):
        phi, base = fn(extract_ensemble(model), X)
        assert float(np.max(np.abs(base + phi.sum(axis=1) - margin))) < 1e-4, fn.__name__
    ref = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent").shap_values(X, check_additivity=False)
    np.testing.assert_allclose(treeshap_phi_base_numba(extract_ensemble(model), X)[0], ref, rtol=0, atol=1e-4)


def test_a_plain_xgboost_booster_has_no_tree_weights():
    """gbtree boosters keep unit weights: the DART branch must not touch them."""
    X, y = _frame()
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3, verbosity=0, n_jobs=1).fit(X, y)
    margin = model.get_booster().predict(xgb.DMatrix(X), output_margin=True)
    phi, base = treeshap_phi_base_numba(extract_ensemble(model), X)
    assert float(np.max(np.abs(base + phi.sum(axis=1) - margin))) < 1e-4


def test_lightgbm_linear_trees_are_declined():
    """A linear_tree leaf predicts const + coef . x; the kernels add only the constant, so the support check must say no (the public library raises for linear trees too)."""
    X, y = _frame()
    plain = lgb.LGBMRegressor(n_estimators=5, num_leaves=8, verbose=-1, n_jobs=1).fit(X, y)
    linear = lgb.LGBMRegressor(n_estimators=5, num_leaves=8, linear_tree=True, verbose=-1, n_jobs=1).fit(X, y)
    assert is_supported_lightgbm(plain) is True
    assert is_supported_lightgbm(linear) is False


def test_lightgbm_values_on_the_zero_bin_boundary_follow_lightgbm():
    """LightGBM reads |x| <= kZeroThreshold (1e-35f) as 0.0, so a value exactly on the +-kZeroThreshold split goes where zero goes. Our attributions equal LightGBM's native pred_contrib there."""
    X, y = _frame(seed=0)
    model = lgb.LGBMRegressor(n_estimators=5, num_leaves=15, learning_rate=0.2, min_child_samples=5, verbose=-1, n_jobs=1, random_state=0).fit(X.astype(np.float64), y)
    ensemble = extract_ensemble(model)
    assert ensemble.zero_threshold == _LIGHTGBM_ZERO_THRESHOLD
    boundary = np.float32(_LIGHTGBM_ZERO_THRESHOLD)
    for value in (-boundary, boundary, np.nextafter(boundary, np.float32(0.0)), np.float32(0.0)):
        for feature in range(X.shape[1]):
            row = np.median(X, axis=0).astype(np.float32)[None]
            row[0, feature] = value
            native = model.booster_.predict(row, pred_contrib=True)[0][:-1]
            for fn in (treeshap_phi_base_path, treeshap_phi_base_numba):
                np.testing.assert_allclose(fn(ensemble, row)[0][0], native, rtol=0, atol=1e-9, err_msg=f"{fn.__name__} feature {feature} value {float(value)!r}")


def test_xgboost_has_no_zero_threshold_rewrite():
    """Only LightGBM carries the sparse-row convention; an XGBoost ensemble leaves tiny values alone."""
    X, y = _frame()
    model = xgb.XGBRegressor(n_estimators=3, max_depth=2, verbosity=0, n_jobs=1).fit(X, y)
    ensemble = extract_ensemble(model)
    tiny = np.full((1, X.shape[1]), 1e-36, dtype=np.float32)
    assert ensemble.zero_threshold == 0.0 and np.array_equal(ensemble.as_float32(tiny), tiny)
