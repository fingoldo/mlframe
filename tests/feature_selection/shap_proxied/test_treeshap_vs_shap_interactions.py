"""SHAP interaction values against the official shap: symmetry, row sums, GPU against CPU."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions import interaction_tensor_numba
from tests.feature_selection.shap_proxied.treeshap_vs_shap_common import (
    fit_lgb,
    fit_xgb,
    gpu_available,
    make_data,
    margin,
    shap_interactions_ref,
    ts,
    xgb_interactions_ref64,
)

CASES = [("xgb", "reg"), ("xgb", "bin"), ("lgb", "reg"), ("lgb", "bin")]


def _case(name, kind="reg", f=5, n=80, **kw):
    """``(model, X)`` of a small model of the given family."""
    X, y = make_data(n=n, f=f, kind=kind)
    fit = fit_xgb if name == "xgb" else fit_lgb
    return fit(X, y, kind, n_estimators=8, **kw), X


def _nan_trained_case(name):
    """``(model, X)`` trained with 20 percent NaN, so the splits learned a NaN direction."""
    X, y = make_data(n=80, f=5)
    Xn = np.where(np.random.default_rng(2).random(X.shape) < 0.2, np.nan, X).astype(np.float32)
    fit = fit_xgb if name == "xgb" else fit_lgb
    return fit(Xn, y, n_estimators=8), X


def _ref(model, X, name):
    """Float64 reference interactions: shap's C++ kernel (xgboost through the dict form, lightgbm directly)."""
    return xgb_interactions_ref64(model, X) if name == "xgb" else shap_interactions_ref(model, X)


@pytest.mark.parametrize("name,kind", CASES)
def test_matches_shap_float64_kernel(name, kind):
    """Phi equals shap's interaction values to 1e-9; the main effects are the row sums."""
    m, X = _case(name, kind)
    Phi, phi, _ = interaction_tensor_numba(ts.extract_ensemble(m), X)
    np.testing.assert_allclose(Phi, _ref(m, X, name), rtol=0, atol=1e-9)
    np.testing.assert_allclose(Phi.sum(2), phi, rtol=0, atol=1e-9)


@pytest.mark.parametrize("name,kind", CASES)
def test_symmetry_and_row_sum_and_additivity(name, kind):
    """Phi_ij == Phi_ji, sum_j Phi_ij == phi_i, and base + sum Phi equals the margin."""
    m, X = _case(name, kind)
    Phi, phi, base = interaction_tensor_numba(ts.extract_ensemble(m), X)
    assert np.array_equal(Phi, Phi.transpose(0, 2, 1))
    np.testing.assert_allclose(Phi.sum(2), phi, rtol=0, atol=1e-9)
    tol = 5e-6 if name == "xgb" else 1e-9
    np.testing.assert_allclose(base + Phi.sum((1, 2)), margin(m, X), rtol=0, atol=tol)


def test_xgboost_matches_shap_native_float32():
    """Against shap's direct XGBoost path (native float32 pred_interactions) the difference is below 1e-5."""
    m, X = _case("xgb")
    Phi = interaction_tensor_numba(ts.extract_ensemble(m), X)[0]
    np.testing.assert_allclose(Phi, shap_interactions_ref(m, X), rtol=0, atol=1e-5)


def test_main_effect_rank_agrees_with_shap():
    """The mean|row sum| of the interaction matrix ranks the features as shap's."""
    m, X = _case("lgb")
    Phi = interaction_tensor_numba(ts.extract_ensemble(m), X)[0]
    ref = shap_interactions_ref(m, X)
    assert list(np.argsort(-np.abs(Phi.sum(2)).mean(0))) == list(np.argsort(-np.abs(ref.sum(2)).mean(0)))


@pytest.mark.parametrize("name", ["xgb", "lgb"])
def test_nan_and_single_row(name):
    """NaN rows and n=1."""
    m, X = _nan_trained_case(name)
    Xs = X.copy()
    Xs[::3, 1] = np.nan
    ens = ts.extract_ensemble(m)
    np.testing.assert_allclose(interaction_tensor_numba(ens, Xs)[0], _ref(m, Xs, name), rtol=0, atol=1e-9)
    np.testing.assert_allclose(interaction_tensor_numba(ens, X[:1])[0], _ref(m, X[:1], name), rtol=0, atol=1e-9)


@pytest.mark.parametrize("name", ["xgb", "lgb"])
def test_stumps_and_constant_target(name):
    """Stumps have no off-diagonal; a single-leaf model gives all zeros."""
    X, y = make_data(n=80, f=4)
    fit = fit_xgb if name == "xgb" else fit_lgb
    stump = fit(X, y, n_estimators=6, **(dict(max_depth=1) if name == "xgb" else dict(num_leaves=2)))
    Phi = interaction_tensor_numba(ts.extract_ensemble(stump), X)[0]
    np.testing.assert_allclose(Phi, _ref(stump, X, name), rtol=0, atol=1e-9)
    assert np.all(Phi[:, ~np.eye(4, dtype=bool)] == 0)
    const = fit(X, np.ones(len(X)), n_estimators=3)
    assert np.all(interaction_tensor_numba(ts.extract_ensemble(const), X)[0] == 0)


def test_repeated_feature_and_duplicated_columns():
    """Few features with deep trees repeat features on a path; a duplicated column splits credit like shap."""
    X, y = make_data(n=200, f=3)
    X[:, 2] = X[:, 0]
    m = fit_lgb(X, y, n_estimators=6, num_leaves=31)
    Phi = interaction_tensor_numba(ts.extract_ensemble(m), X)[0]
    np.testing.assert_allclose(Phi, shap_interactions_ref(m, X), rtol=0, atol=1e-9)


def test_lightgbm_nan_without_training_missing_is_self_consistent():
    """LightGBM trained without NaN scores NaN as 0.0; our interactions keep row sums equal to the main effects there (the official shap interactions do not, see the findings notes)."""
    m, X = _case("lgb")
    X[::3, 1] = np.nan
    ens = ts.extract_ensemble(m)
    Phi = interaction_tensor_numba(ens, X)[0]
    np.testing.assert_allclose(Phi.sum(2), m.booster_.predict(X, pred_contrib=True)[:, :-1], rtol=0, atol=1e-9)


def test_unsupported_models_have_no_ensemble():
    """Multiclass cannot enter the interaction path: extract_ensemble is None."""
    X, y = make_data(kind="multi")
    assert ts.extract_ensemble(fit_xgb(X, y, "bin")) is None
    assert ts.extract_ensemble(fit_lgb(X, y, "bin")) is None


@pytest.mark.skipif(not gpu_available(), reason="no CUDA device / cupy unavailable")
@pytest.mark.parametrize("name,kind", CASES)
def test_gpu_interactions_equal_cpu_and_shap(name, kind):
    """The device interaction tensor equals the numba one and shap, with symmetry and row sums."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_gpu import interaction_tensor_gpu

    m, X = _case(name, kind)
    ens = ts.extract_ensemble(m)
    Phi, phi, _ = interaction_tensor_gpu(ens, X)
    cpu = interaction_tensor_numba(ens, X)[0]
    np.testing.assert_allclose(Phi, cpu, rtol=0, atol=1e-9)
    np.testing.assert_allclose(Phi, _ref(m, X, name), rtol=0, atol=1e-9)
    assert np.array_equal(Phi, Phi.transpose(0, 2, 1))
    np.testing.assert_allclose(Phi.sum(2), phi, rtol=0, atol=1e-9)


@pytest.mark.skipif(not gpu_available(), reason="no CUDA device / cupy unavailable")
def test_gpu_interactions_with_nan_rows():
    """NaN rows on the device."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_gpu import interaction_tensor_gpu

    m, X = _nan_trained_case("lgb")
    X[::3, 1] = np.nan
    ens = ts.extract_ensemble(m)
    np.testing.assert_allclose(interaction_tensor_gpu(ens, X)[0], shap_interactions_ref(m, X), rtol=0, atol=1e-9)
