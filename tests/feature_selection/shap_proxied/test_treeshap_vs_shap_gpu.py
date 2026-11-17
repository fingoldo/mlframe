"""The device TreeSHAP kernels against the official shap: float32 default (1e-4 and the same rank), float64 reference (1e-9), table and path forms, depth cap."""

from __future__ import annotations

import numpy as np
import pytest

from tests.feature_selection.shap_proxied.treeshap_vs_shap_common import (
    assert_matches_shap,
    fit_lgb,
    fit_xgb,
    make_data,
    ours,
    requires_gpu,
    shap_ref,
    ts,
    xgb_ref64,
)

pytestmark = requires_gpu


@pytest.fixture(params=["lgb", "xgb"])
def fitter(request):
    """The model fitting function of each family."""
    return fit_lgb if request.param == "lgb" else fit_xgb


@pytest.mark.parametrize("which", ["gpu", "gpu_path"])
@pytest.mark.parametrize("kind", ["reg", "bin"])
def test_float32_default(fitter, kind, which, monkeypatch):
    """Default float32 kernels stay within 1e-4 of shap with the same mean|phi| rank and additivity."""
    monkeypatch.delenv("MLFRAME_SHAP_TREESHAP_REAL", raising=False)
    X, y = make_data(kind=kind)
    assert_matches_shap(fitter(X, y, kind), X, which, atol=1e-4)


@pytest.mark.parametrize("which", ["gpu", "gpu_path"])
def test_float64_reference(fitter, which, monkeypatch):
    """MLFRAME_SHAP_TREESHAP_REAL=float64 reproduces shap to 1e-9."""
    monkeypatch.setenv("MLFRAME_SHAP_TREESHAP_REAL", "float64")
    X, y = make_data()
    assert_matches_shap(fitter(X, y), X, which, atol=1e-9)


@pytest.mark.parametrize("which", ["gpu", "gpu_path"])
def test_float64_equals_numba(fitter, which, monkeypatch):
    """Between our own backends: device float64 equals the numba kernel to 1e-9."""
    monkeypatch.setenv("MLFRAME_SHAP_TREESHAP_REAL", "float64")
    X, y = make_data()
    m = fitter(X, y)
    np.testing.assert_allclose(ours(m, X, which)[0], ours(m, X, "numba")[0], rtol=0, atol=1e-9)


def test_gpu_missing_values_and_infinities(monkeypatch):
    """NaN and +-inf rows on a LightGBM model."""
    monkeypatch.setenv("MLFRAME_SHAP_TREESHAP_REAL", "float64")
    X, y = make_data()
    Xs = X.copy()
    Xs[::5, 0] = np.nan
    Xs[::7, 1] = np.inf
    Xs[::9, 2] = -np.inf
    for which in ("gpu", "gpu_path"):
        assert_matches_shap(fit_lgb(X, y), Xs, which, atol=1e-9)


def test_gpu_single_row_and_fortran_input(fitter):
    """n=1 and a Fortran-ordered matrix."""
    X, y = make_data()
    m = fitter(X, y)
    assert_matches_shap(m, X[:1], "gpu", atol=1e-4)
    assert_matches_shap(m, np.asfortranarray(X), "gpu", atol=1e-4)


def test_gpu_large_magnitudes_float32():
    """Features of 1e30."""
    X, y = make_data()
    assert_matches_shap(fit_lgb(X * 1e30, y), X * 1e30, "gpu", atol=1e-4)


def test_gpu_deep_tree_within_cap(monkeypatch):
    """Depth 14 is within the device cap and matches shap in float64."""
    monkeypatch.setenv("MLFRAME_SHAP_TREESHAP_REAL", "float64")
    X, y = make_data(n=3000)
    m = fit_xgb(X, y, max_depth=14, n_estimators=3, min_child_weight=0, reg_lambda=0)
    for which in ("gpu", "gpu_path"):
        assert_matches_shap(m, X, which, atol=1e-9)


def test_gpu_depth_beyond_cap_raises_clearly():
    """A tree deeper than the device cap raises ValueError naming the numba backend; numba still answers."""
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_treeshap_gpu as g

    X, y = make_data(n=400, f=3)
    ens = ts.extract_ensemble(fit_lgb(X, y, n_estimators=2))
    ens.max_depth = g._MAX_SUPPORTED_DEPTH + 1
    with pytest.raises(ValueError, match="numba"):
        g.treeshap_phi_base_gpu(ens, X)
    assert ts.treeshap_phi_base_numba(ens, X[:5])[0].shape == (5, 3)


def test_gpu_feature_width_1000_float32():
    """1000 columns, 5 trees."""
    X = np.random.default_rng(3).normal(size=(100, 1000)).astype(np.float32)
    m = fit_xgb(X, X[:, 0] + X[:, 5] * X[:, 7], n_estimators=5)
    assert_matches_shap(m, X, "gpu", atol=1e-4)


def test_gpu_repeated_feature_on_path(monkeypatch):
    """Two features, deep trees: repeated features on one path, float64."""
    monkeypatch.setenv("MLFRAME_SHAP_TREESHAP_REAL", "float64")
    X, y = make_data(n=1000, f=2)
    m = fit_xgb(X, y, max_depth=9, n_estimators=8)
    for which in ("gpu", "gpu_path"):
        assert_matches_shap(m, X, which, atol=1e-9)


def test_gpu_float32_close_to_native_shap_xgboost(monkeypatch):
    """Against shap on the XGBoost model directly, and its float64 kernel, the float32 device values stay within 1e-4."""
    monkeypatch.delenv("MLFRAME_SHAP_TREESHAP_REAL", raising=False)
    X, y = make_data()
    m = fit_xgb(X, y)
    ref, _ = shap_ref(m, X)
    phi, _ = ours(m, X, "gpu")
    np.testing.assert_allclose(phi, ref, rtol=0, atol=1e-4)
    np.testing.assert_allclose(phi, xgb_ref64(m, X), rtol=0, atol=1e-4)
