"""Unit tests for lightgbm support in the custom path-dependent TreeSHAP backend.

``extract_ensemble`` parses a lightgbm booster into the SAME flat node tensors as xgboost, so the
shared numba kernels (main-effect AND interaction) apply unchanged. The two lightgbm-specific
normalisations -- ``x <= threshold`` routing converted to the kernel's ``x < threshold`` via
``nextafter``, and sample-count cover -- are validated here by:
  * EXACT additivity ``base + phi.sum(1) == raw-score margin``,
  * PARITY with ``shap.TreeExplainer`` main-effect values to ~1e-4,
  * interaction symmetry + row-sum + parity vs ``shap.shap_interaction_values``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shap")
pytest.importorskip("lightgbm")
pytest.importorskip("numba")


def _fit_lgb(X, y, *, classification, n_estimators=40, max_depth=4, num_leaves=15, seed=0):
    from lightgbm import LGBMClassifier, LGBMRegressor

    params = dict(n_estimators=n_estimators, max_depth=max_depth, num_leaves=num_leaves, learning_rate=0.2, random_state=seed, verbose=-1)
    m = LGBMClassifier(**params) if classification else LGBMRegressor(**params)
    m.fit(X, y)
    return m


def _shap_main_reference(model, X):
    import shap

    ex = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    phi = ex.shap_values(X, check_additivity=False)
    if isinstance(phi, list):  # binary -> positive class
        phi = phi[1] if len(phi) == 2 else phi[0]
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim == 3:
        phi = phi[:, :, -1]
    base = ex.expected_value
    base = float(np.ravel(base)[-1]) if np.ndim(base) > 0 else float(base)
    return phi, base


@pytest.mark.parametrize("classification", [False, True])
def test_lightgbm_is_supported_and_extracts(classification):
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble, is_supported_lightgbm

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(150, 6)), columns=[f"f{i}" for i in range(6)])
    signal = 2.0 * X["f0"] + X["f1"] - 0.5 * X["f2"]
    y = (signal + 0.3 * rng.normal(size=150) > 0).astype(int) if classification else (signal + 0.1 * rng.normal(size=150)).to_numpy()
    model = _fit_lgb(X, y, classification=classification)
    assert is_supported_lightgbm(model)
    ens = extract_ensemble(model)
    assert ens is not None and ens.n_features == 6


@pytest.mark.parametrize("classification", [False, True])
def test_lightgbm_main_effect_additivity_and_parity(classification):
    """base + phi.sum(1) reconstructs the raw-score margin, and phi matches shap to ~1e-4."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble, treeshap_phi_base_numba

    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(250, 8)), columns=[f"f{i}" for i in range(8)])
    signal = X["f0"] + 0.5 * X["f1"] - X["f3"]
    y = (signal + 0.3 * rng.normal(size=250) > 0).astype(int) if classification else (signal + 0.1 * rng.normal(size=250)).to_numpy()
    model = _fit_lgb(X, y, classification=classification, n_estimators=50, max_depth=5, num_leaves=31)

    ens = extract_ensemble(model)
    phi, base = treeshap_phi_base_numba(ens, X.values)
    margin = model.predict(X, raw_score=True)
    np.testing.assert_allclose(base + phi.sum(axis=1), margin, rtol=0, atol=1e-4)

    phi_ref, base_ref = _shap_main_reference(model, X)
    np.testing.assert_allclose(phi, phi_ref, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(base, base_ref, rtol=1e-4, atol=1e-4)


def test_lightgbm_handles_missing_values():
    """NaN features route via the node default child (lightgbm default_left) and stay additive."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble, treeshap_phi_base_numba

    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(180, 6)), columns=[f"f{i}" for i in range(6)])
    y = (2.0 * X["f0"] + X["f1"] + 0.1 * rng.normal(size=180)).to_numpy()
    model = _fit_lgb(X, y, classification=False)
    Xn = X.copy()
    Xn.iloc[::7, 0] = np.nan
    Xn.iloc[::5, 3] = np.nan

    ens = extract_ensemble(model)
    phi, base = treeshap_phi_base_numba(ens, Xn.values)
    margin = model.predict(Xn, raw_score=True)
    np.testing.assert_allclose(base + phi.sum(axis=1), margin, rtol=0, atol=1e-4)
    phi_ref, _ = _shap_main_reference(model, Xn)
    np.testing.assert_allclose(phi, phi_ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("classification", [False, True])
def test_lightgbm_interaction_parity(classification):
    """Interaction symmetry + row-sum identity + parity vs shap + additivity, for lightgbm."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions import interaction_tensor_numba

    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(180, 7)), columns=[f"f{i}" for i in range(7)])
    signal = X["f0"] * X["f1"] + 0.5 * X["f2"] - X["f3"]
    y = (signal + 0.3 * rng.normal(size=180) > 0).astype(int) if classification else (signal + 0.1 * rng.normal(size=180)).to_numpy()
    model = _fit_lgb(X, y, classification=classification, n_estimators=50, max_depth=5, num_leaves=31)

    ens = extract_ensemble(model)
    Phi, phi, base = interaction_tensor_numba(ens, X.values)

    # symmetry + row-sum identity.
    np.testing.assert_allclose(Phi, np.transpose(Phi, (0, 2, 1)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(Phi.sum(axis=2), phi, rtol=0, atol=1e-10)

    import shap

    ex = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    Phi_ref = np.asarray(ex.shap_interaction_values(X), dtype=np.float64)
    if Phi_ref.ndim == 4:
        Phi_ref = Phi_ref[:, :, :, -1]
    np.testing.assert_allclose(Phi, Phi_ref, rtol=1e-4, atol=1e-4)

    margin = model.predict(X, raw_score=True)
    np.testing.assert_allclose(base + Phi.sum(axis=(1, 2)), margin, rtol=0, atol=1e-4)
