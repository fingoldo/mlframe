"""Unit + biz_val for the custom path-dependent TreeSHAP backend.

The custom numba TreeSHAP replaces the ``shap`` library on the wide-data OOF-SHAP hotspot. These tests
are the mandatory correctness gate:
  * EXACT additivity: ``base + phi.sum(1) == model margin`` (the tree_path_dependent invariant),
  * PARITY: phi matches ``shap.TreeExplainer(feature_perturbation="tree_path_dependent")`` to ~1e-4,
plus a biz_value test that quantifies the speedup vs ``shap`` on a wide (2000-feature) matrix, and a
dispatcher routing test (auto picks the custom path on wide xgboost, ``shap`` on narrow / unsupported).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("xgboost")
pytest.importorskip("numba")


def _fit_xgb(X, y, *, classification, n_estimators=40, max_depth=5, seed=0):
    from xgboost import XGBClassifier, XGBRegressor

    params = dict(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.2,
                  random_state=seed, tree_method="hist")
    m = XGBClassifier(**params, eval_metric="logloss") if classification else XGBRegressor(**params)
    m.fit(X, y)
    return m


def _shap_reference(model, X):
    """phi (n,f) and scalar base from the shap library, normalised to the positive class for binary."""
    import shap

    ex = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    phi = np.asarray(ex.shap_values(X, check_additivity=False), dtype=np.float64)
    base = ex.expected_value
    if phi.ndim == 3:  # (n, f, classes) -> positive class
        phi = phi[:, :, -1]
    base = float(np.ravel(base)[-1]) if np.ndim(base) > 0 else float(base)
    return phi, base


@pytest.mark.parametrize("classification", [False, True])
def test_treeshap_numba_additivity(classification):
    """base + phi.sum(1) must reconstruct the model margin (the tree_path_dependent invariant)."""
    from mlframe.feature_selection._shap_proxy_treeshap import extract_ensemble, treeshap_phi_base_numba

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 8)), columns=[f"f{i}" for i in range(8)])
    signal = 2.0 * X["f0"] + X["f1"] - 0.5 * X["f2"]
    y = (signal + 0.3 * rng.normal(size=200) > 0).astype(int) if classification else \
        (signal + 0.1 * rng.normal(size=200)).to_numpy()
    model = _fit_xgb(X, y, classification=classification)

    ens = extract_ensemble(model)
    assert ens is not None
    phi, base = treeshap_phi_base_numba(ens, X.values)
    margin = model.predict(X, output_margin=True)
    recon = base + phi.sum(axis=1)
    np.testing.assert_allclose(recon, margin, rtol=0, atol=1e-4)


@pytest.mark.parametrize("classification", [False, True])
def test_treeshap_numba_parity_vs_shap(classification):
    """phi (and base) must match the shap library TreeExplainer to tight tolerance."""
    from mlframe.feature_selection._shap_proxy_treeshap import extract_ensemble, treeshap_phi_base_numba

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(250, 10)), columns=[f"f{i}" for i in range(10)])
    signal = X["f0"] + 0.5 * X["f1"] - X["f3"]
    y = (signal + 0.3 * rng.normal(size=250) > 0).astype(int) if classification else \
        (signal + 0.1 * rng.normal(size=250)).to_numpy()
    model = _fit_xgb(X, y, classification=classification, n_estimators=50, max_depth=6)

    ens = extract_ensemble(model)
    phi, base = treeshap_phi_base_numba(ens, X.values)
    phi_ref, base_ref = _shap_reference(model, X)

    # rtol ~1e-4 (scaled by attribution magnitude) is the mandated parity gate.
    np.testing.assert_allclose(phi, phi_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(base, base_ref, rtol=1e-4, atol=1e-4)


def test_treeshap_handles_missing_values():
    """NaN features must route via the node's ``missing`` child and still satisfy additivity."""
    from mlframe.feature_selection._shap_proxy_treeshap import extract_ensemble, treeshap_phi_base_numba

    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(180, 6)), columns=[f"f{i}" for i in range(6)])
    y = (2.0 * X["f0"] + X["f1"] + 0.1 * rng.normal(size=180)).to_numpy()
    model = _fit_xgb(X, y, classification=False)
    Xn = X.copy()
    Xn.iloc[::7, 0] = np.nan
    Xn.iloc[::5, 3] = np.nan

    ens = extract_ensemble(model)
    phi, base = treeshap_phi_base_numba(ens, Xn.values)
    margin = model.predict(Xn, output_margin=True)
    np.testing.assert_allclose(base + phi.sum(axis=1), margin, rtol=0, atol=1e-4)
    phi_ref, _ = _shap_reference(model, Xn)
    np.testing.assert_allclose(phi, phi_ref, rtol=1e-4, atol=1e-4)


def test_dispatcher_routes_by_width_and_model():
    """auto picks the custom numba path on WIDE xgboost, falls back to shap on narrow / unsupported."""
    from sklearn.linear_model import LogisticRegression

    from mlframe.feature_selection._shap_proxy_explain import _pick_backend, _treeshap_numba_min_features

    rng = np.random.default_rng(3)
    narrow = pd.DataFrame(rng.normal(size=(50, 8)), columns=[f"f{i}" for i in range(8)])
    wide_f = _treeshap_numba_min_features() + 16
    wide = pd.DataFrame(rng.normal(size=(50, wide_f)), columns=[f"f{i}" for i in range(wide_f)])
    y = (narrow["f0"] > 0).astype(int)

    xgb_wide = _fit_xgb(wide, y, classification=True, n_estimators=10)
    xgb_narrow = _fit_xgb(narrow, y, classification=True, n_estimators=10)
    assert _pick_backend(xgb_wide, wide, "auto") == "treeshap_numba"
    assert _pick_backend(xgb_narrow, narrow, "auto") == "shap"  # narrow -> shap C-ext already fast

    # Unsupported model type always falls back to shap.
    lr = LogisticRegression(max_iter=200).fit(wide, y)
    assert _pick_backend(lr, wide, "auto") == "shap"
    # Explicit override is honoured.
    assert _pick_backend(xgb_narrow, narrow, "treeshap_numba") == "treeshap_numba"
    assert _pick_backend(xgb_wide, wide, "shap") == "shap"


def test_dispatcher_phi_parity_through_compute_shap_matrix():
    """compute_shap_matrix(out_of_fold=False) with the custom backend matches the shap backend."""
    from mlframe.feature_selection._shap_proxy_explain import compute_shap_matrix, make_default_estimator

    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    X, y, _ = make_regime_dataset(n_samples=400, n_informative=5, n_noise=90, task="regression", seed=4)
    tmpl = make_default_estimator(False, n_estimators=60)

    phi_c, base_c, _ = compute_shap_matrix(
        tmpl, X, y, classification=False, out_of_fold=False,
        rng=np.random.default_rng(0), shap_backend="treeshap_numba")
    phi_s, base_s, _ = compute_shap_matrix(
        tmpl, X, y, classification=False, out_of_fold=False,
        rng=np.random.default_rng(0), shap_backend="shap")

    np.testing.assert_allclose(phi_c, phi_s, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(base_c, base_s, rtol=1e-4, atol=1e-4)


def _gpu_treeshap_available():
    try:
        from mlframe.feature_selection._shap_proxy_treeshap_gpu import gpu_treeshap_available

        return gpu_treeshap_available()
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_treeshap_available(), reason="no CUDA device / cupy unavailable")
@pytest.mark.parametrize("classification", [False, True])
def test_treeshap_gpu_matches_numba(classification):
    """The cupy kernel must match the numba kernel (the always-available fallback) to ~machine eps."""
    from mlframe.feature_selection._shap_proxy_treeshap import extract_ensemble, treeshap_phi_base_numba
    from mlframe.feature_selection._shap_proxy_treeshap_gpu import treeshap_phi_base_gpu

    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(220, 9)), columns=[f"f{i}" for i in range(9)])
    signal = 2.0 * X["f0"] + X["f1"] - 0.5 * X["f2"]
    y = (signal + 0.3 * rng.normal(size=220) > 0).astype(int) if classification else \
        (signal + 0.1 * rng.normal(size=220)).to_numpy()
    model = _fit_xgb(X, y, classification=classification, n_estimators=40, max_depth=6)
    ens = extract_ensemble(model)

    phi_n, base_n = treeshap_phi_base_numba(ens, X.values)
    phi_g, base_g = treeshap_phi_base_gpu(ens, X.values)
    np.testing.assert_allclose(phi_g, phi_n, rtol=0, atol=1e-9)
    np.testing.assert_allclose(base_g, base_n, rtol=0, atol=1e-12)


@pytest.mark.slow
def test_biz_val_treeshap_faster_than_shap_on_wide_data():
    """biz_value: the custom numba TreeSHAP beats the shap library on a wide (2000-feature) matrix.

    Quantifies the win on the profiled OOF-SHAP hotspot. The numba kernel cost is ~flat in width while
    shap scales with it, so the speedup grows with the column count. We assert a conservative >=1.3x.
    """
    from mlframe.feature_selection._shap_proxy_treeshap import extract_ensemble, treeshap_phi_base_numba

    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    X, y, _ = make_regime_dataset(n_samples=1500, n_informative=8, n_noise=1992, task="regression", seed=5)
    model = _fit_xgb(X, y, classification=False, n_estimators=200, max_depth=4)
    ens = extract_ensemble(model)

    treeshap_phi_base_numba(ens, X.values[:16])  # JIT warmup (excluded from timing)
    t0 = time.perf_counter()
    phi_n, _ = treeshap_phi_base_numba(ens, X.values)
    t_numba = time.perf_counter() - t0

    phi_ref, _ = _shap_reference(model, X)  # also warms shap's first-call setup
    t0 = time.perf_counter()
    import shap
    ex = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    _ = ex.shap_values(X, check_additivity=False)
    t_shap = time.perf_counter() - t0

    speedup = t_shap / max(t_numba, 1e-9)
    # Correctness still holds at scale.
    np.testing.assert_allclose(phi_n, phi_ref, rtol=1e-4, atol=1e-4)
    assert speedup >= 1.3, f"expected >=1.3x speedup on 2000 features, got {speedup:.2f}x " \
                           f"(numba {t_numba:.3f}s vs shap {t_shap:.3f}s)"
