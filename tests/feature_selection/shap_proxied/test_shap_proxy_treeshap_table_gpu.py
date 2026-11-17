"""The device table TreeSHAP equals the numba path kernel (float64 exactly, float32 within its rounding), handles missing values, and the dispatcher falls back to the path kernel where the table does not apply."""

from __future__ import annotations

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")


def _gpu_available() -> bool:
    """Whether a CUDA device usable for the kernel is present."""
    try:
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_gpu import gpu_treeshap_available

        return gpu_treeshap_available()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _gpu_available(), reason="no CUDA device / cupy unavailable")


def _ensemble(n_features: int, depth: int, trees: int, classification: bool = False, seed: int = 0):
    """``(ensemble, X)`` of a small XGBoost model; few features and a deep tree make features repeat on a path."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(300, n_features)).astype(np.float32)
    signal = X[:, 0] * 2 + X[:, 1] * X[:, 2 % n_features] - np.sin(X[:, 3 % n_features])
    if classification:
        y = (signal + 0.3 * rng.normal(size=300) > 0).astype(int)
        model = xgb.XGBClassifier(n_estimators=trees, max_depth=depth, tree_method="hist", verbosity=0).fit(X, y)
    else:
        y = signal + 0.1 * rng.normal(size=300)
        model = xgb.XGBRegressor(n_estimators=trees, max_depth=depth, tree_method="hist", verbosity=0).fit(X, y)
    return extract_ensemble(model), X


@pytest.mark.parametrize("n_features,depth,classification", [(8, 4, False), (4, 6, False), (5, 5, True), (3, 7, False)])
def test_float64_table_kernel_equals_the_numba_path_kernel(n_features, depth, classification, monkeypatch):
    """Exact reference type: the device table matches the numba path kernel to machine eps, repeated features on a path included."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import treeshap_phi_base_path
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_table_gpu import treeshap_phi_base_table_gpu

    monkeypatch.setenv("MLFRAME_SHAP_TREESHAP_REAL", "float64")
    ens, X = _ensemble(n_features, depth, 25, classification)
    got = treeshap_phi_base_table_gpu(ens, X)
    assert got is not None
    ref, base = treeshap_phi_base_path(ens, X)
    np.testing.assert_allclose(got[0], ref, rtol=0, atol=1e-9)
    assert got[1] == base


def test_float32_default_is_close_and_keeps_the_rank(monkeypatch):
    """The default type (float32) stays within 1e-4 and ranks the features alike; the phi contract stays float64."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import treeshap_phi_base_path
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_table_gpu import treeshap_phi_base_table_gpu

    monkeypatch.delenv("MLFRAME_SHAP_TREESHAP_REAL", raising=False)
    ens, X = _ensemble(8, 5, 25)
    got = treeshap_phi_base_table_gpu(ens, X)
    ref, _ = treeshap_phi_base_path(ens, X)
    assert got[0].dtype == np.float64
    np.testing.assert_allclose(got[0], ref, rtol=0, atol=1e-4)
    assert list(np.argsort(-np.abs(ref).mean(0))[:3]) == list(np.argsort(-np.abs(got[0]).mean(0))[:3])


def test_missing_values_route_to_the_default_child(monkeypatch):
    """A NaN follows the default child on the device exactly as on the host."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import treeshap_phi_base_path
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_table_gpu import treeshap_phi_base_table_gpu

    monkeypatch.setenv("MLFRAME_SHAP_TREESHAP_REAL", "float64")
    ens, X = _ensemble(6, 5, 30)
    Xn = X.copy()
    Xn[np.random.default_rng(1).random(Xn.shape) < 0.2] = np.nan
    np.testing.assert_allclose(treeshap_phi_base_table_gpu(ens, Xn)[0], treeshap_phi_base_path(ens, Xn)[0], rtol=0, atol=1e-9)


def test_the_dispatcher_falls_back_to_the_path_kernel_when_the_table_does_not_fit(monkeypatch):
    """Over the table budget the table form answers None and ``treeshap_phi_base_gpu`` still returns the path kernel's values."""
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_treeshap_table as table_mod
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import treeshap_phi_base_path
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_gpu import treeshap_phi_base_gpu
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_table_gpu import treeshap_phi_base_table_gpu

    monkeypatch.setenv("MLFRAME_SHAP_TREESHAP_REAL", "float64")
    monkeypatch.setattr(table_mod, "MAX_TABLE_ENTRIES", 10)
    ens, X = _ensemble(8, 5, 25)
    assert treeshap_phi_base_table_gpu(ens, X) is None
    np.testing.assert_allclose(treeshap_phi_base_gpu(ens, X)[0], treeshap_phi_base_path(ens, X)[0], rtol=0, atol=1e-9)


def test_use_table_false_runs_the_device_path_kernel(monkeypatch):
    """The path kernel stays reachable as the reference of the table form."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import treeshap_phi_base_path
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_gpu import treeshap_phi_base_gpu

    monkeypatch.setenv("MLFRAME_SHAP_TREESHAP_REAL", "float64")
    ens, X = _ensemble(6, 4, 20)
    np.testing.assert_allclose(treeshap_phi_base_gpu(ens, X, use_table=False)[0], treeshap_phi_base_path(ens, X)[0], rtol=0, atol=1e-9)


def test_preorder_kernel_equals_stack_kernel_with_nan(monkeypatch):
    """The static-preorder walk (default) gives the stack walk's phi bit for bit, NaN rows included."""
    from mlframe.feature_selection._benchmarks.profiling.treeshap_gpu_profile import build_ensemble
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_table_gpu import treeshap_phi_base_table_gpu

    ens, X = build_ensemble(3000, 20, 30, 5)
    X = X.copy()
    X[::7, 2] = np.nan
    X[::5, 9] = np.nan
    monkeypatch.setenv("MLFRAME_SHAP_TREESHAP_KERNEL", "stack")
    ref = treeshap_phi_base_table_gpu(ens, X)
    monkeypatch.setenv("MLFRAME_SHAP_TREESHAP_KERNEL", "preorder")
    got = treeshap_phi_base_table_gpu(ens, X)
    np.testing.assert_array_equal(got[0], ref[0])
