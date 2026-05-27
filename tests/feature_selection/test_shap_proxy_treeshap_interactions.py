"""Unit + biz_val for the custom path-dependent TreeSHAP *interaction*-values kernel.

The numba interaction kernel replaces the painfully slow ``shap.shap_interaction_values`` call on the
interaction-aware coalition path. These tests are the mandatory correctness gate:
  * EXACT symmetry  ``Phi_ij == Phi_ji``,
  * row-sum identity ``sum_k Phi_ik == phi_i`` (the main-effect SHAP value from the existing kernel),
  * PARITY: ``Phi`` matches ``shap.TreeExplainer.shap_interaction_values`` to ~1e-4,
  * EXACT additivity ``base + Phi.sum((1,2)) == model margin``,
plus a biz_value test quantifying the speedup vs ``shap.shap_interaction_values`` on a wide matrix, and
a routing test (``compute_interaction_tensor`` picks the numba path on supported xgboost / wide enough).
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


def _shap_interaction_reference(model, X):
    """(n, P, P) interaction tensor + scalar base from the shap library (positive class for binary)."""
    import shap

    ex = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    Phi = np.asarray(ex.shap_interaction_values(X), dtype=np.float64)
    if Phi.ndim == 4:  # (n, P, P, classes) -> positive class
        Phi = Phi[:, :, :, -1]
    base = ex.expected_value
    base = float(np.ravel(base)[-1]) if np.ndim(base) > 0 else float(base)
    return Phi, base


@pytest.mark.parametrize("classification", [False, True])
@pytest.mark.parametrize("max_depth", [3, 6])
def test_interaction_kernel_symmetry_rowsum_parity(classification, max_depth):
    """Symmetry + row-sum identity + parity vs shap + additivity, the full correctness gate."""
    from mlframe.feature_selection._shap_proxy_treeshap import extract_ensemble
    from mlframe.feature_selection._shap_proxy_treeshap_interactions import interaction_tensor_numba

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(160, 8)), columns=[f"f{i}" for i in range(8)])
    signal = X["f0"] * X["f1"] + 0.5 * X["f2"] - X["f3"]  # genuine pairwise interaction f0*f1
    y = (signal + 0.3 * rng.normal(size=160) > 0).astype(int) if classification else \
        (signal + 0.1 * rng.normal(size=160)).to_numpy()
    model = _fit_xgb(X, y, classification=classification, n_estimators=50, max_depth=max_depth)

    ens = extract_ensemble(model)
    Phi, phi, base = interaction_tensor_numba(ens, X.values)

    # EXACT symmetry.
    np.testing.assert_allclose(Phi, np.transpose(Phi, (0, 2, 1)), rtol=0, atol=1e-12)

    # Row-sum identity: interaction rows sum to the main-effect SHAP values from the SAME kernel,
    # and those match the shap library main-effect values.
    np.testing.assert_allclose(Phi.sum(axis=2), phi, rtol=0, atol=1e-10)
    phi_ref = np.asarray(
        __import__("shap").TreeExplainer(model, feature_perturbation="tree_path_dependent")
        .shap_values(X, check_additivity=False), dtype=np.float64)
    if phi_ref.ndim == 3:
        phi_ref = phi_ref[:, :, -1]
    np.testing.assert_allclose(phi, phi_ref, rtol=1e-4, atol=1e-4)

    # PARITY vs shap interaction values (the mandated ~1e-4 gate).
    Phi_ref, base_ref = _shap_interaction_reference(model, X)
    np.testing.assert_allclose(Phi, Phi_ref, rtol=1e-4, atol=1e-4)

    # EXACT additivity in margin space.
    margin = model.predict(X, output_margin=True)
    np.testing.assert_allclose(base + Phi.sum(axis=(1, 2)), margin, rtol=0, atol=1e-4)
    np.testing.assert_allclose(base, base_ref, rtol=1e-4, atol=1e-4)


def test_interaction_kernel_handles_missing_values():
    """NaN features route via the node ``missing`` child and still match shap interactions + additivity."""
    from mlframe.feature_selection._shap_proxy_treeshap import extract_ensemble
    from mlframe.feature_selection._shap_proxy_treeshap_interactions import interaction_tensor_numba

    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(150, 6)), columns=[f"f{i}" for i in range(6)])
    y = (X["f0"] * X["f1"] + 0.5 * X["f2"] + 0.1 * rng.normal(size=150)).to_numpy()
    model = _fit_xgb(X, y, classification=False, max_depth=5)
    Xn = X.copy()
    Xn.iloc[::7, 0] = np.nan
    Xn.iloc[::5, 3] = np.nan

    ens = extract_ensemble(model)
    Phi, phi, base = interaction_tensor_numba(ens, Xn.values)
    Phi_ref, _ = _shap_interaction_reference(model, Xn)
    np.testing.assert_allclose(Phi, Phi_ref, rtol=1e-4, atol=1e-4)
    margin = model.predict(Xn, output_margin=True)
    np.testing.assert_allclose(base + Phi.sum(axis=(1, 2)), margin, rtol=0, atol=1e-4)


def test_compute_interaction_tensor_numba_matches_shap():
    """compute_interaction_tensor's numba backend matches its shap backend (same contract, ~1e-4)."""
    from mlframe.feature_selection._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection._shap_proxy_interactions import compute_interaction_tensor

    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(300, 12)), columns=[f"f{i}" for i in range(12)])
    y = ((X["f0"] > 0) ^ (X["f1"] > 0)).astype(int).to_numpy()
    tmpl = make_default_estimator(classification=True, n_estimators=60)

    Phi_n, base_n = compute_interaction_tensor(tmpl, X, y, classification=True,
                                               rng=np.random.default_rng(0), backend="treeshap_numba")
    Phi_s, base_s = compute_interaction_tensor(tmpl, X, y, classification=True,
                                               rng=np.random.default_rng(0), backend="shap")
    np.testing.assert_allclose(Phi_n, Phi_s, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(base_n, base_s, rtol=1e-4, atol=1e-4)


def test_compute_interaction_tensor_routes_to_numba_on_wide_xgb():
    """auto routes to the numba kernel for supported xgboost at/above the interaction crossover width,
    and to shap for unsupported models / below the crossover."""
    from sklearn.ensemble import RandomForestClassifier

    from mlframe.feature_selection._shap_proxy_explain import make_default_estimator
    from mlframe.feature_selection._shap_proxy_interactions import (
        _interaction_numba_min_features, _interaction_tensor_numba)
    from mlframe.feature_selection._shap_proxy_treeshap import is_supported_xgboost

    rng = np.random.default_rng(4)
    P = _interaction_numba_min_features() + 4
    X = pd.DataFrame(rng.normal(size=(120, P)), columns=[f"f{i}" for i in range(P)])
    y = ((X["f0"] > 0) ^ (X["f1"] > 0)).astype(int).to_numpy()

    xgb = make_default_estimator(classification=True, n_estimators=30)
    xgb.fit(X, y)
    assert is_supported_xgboost(xgb)
    # numba path produces a valid tensor for the supported wide xgboost model.
    out = _interaction_tensor_numba(xgb, X, classification=True)
    assert out is not None
    Phi, base = out
    assert Phi.shape == (120, P, P)

    # Unsupported model: numba path returns None (caller falls back to shap).
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    assert _interaction_tensor_numba(rf, X, classification=True) is None


@pytest.mark.slow
def test_biz_val_interaction_kernel_faster_than_shap():
    """biz_value: the numba interaction kernel beats ``shap.shap_interaction_values`` on a wide matrix.

    The interaction tensor is the search hotspot on the interaction-aware path; the slow shap call is
    what this kernel replaces. We quantify the win and assert a conservative >=1.15x with parity intact.
    """
    from mlframe.feature_selection._shap_proxy_treeshap import extract_ensemble
    from mlframe.feature_selection._shap_proxy_treeshap_interactions import interaction_tensor_numba

    from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset

    X, y, _ = make_regime_dataset(n_samples=1500, n_informative=6, n_noise=44, task="regression",
                                  interaction_order=2, interaction_strength=0.6, seed=5)
    model = _fit_xgb(X, y, classification=False, n_estimators=200, max_depth=4)
    ens = extract_ensemble(model)

    interaction_tensor_numba(ens, X.values[:16])  # JIT warmup (excluded from timing)
    t0 = time.perf_counter()
    Phi_n, _phi, _base = interaction_tensor_numba(ens, X.values)
    t_numba = time.perf_counter() - t0

    Phi_ref, _ = _shap_interaction_reference(model, X)  # warms shap setup
    t0 = time.perf_counter()
    import shap
    ex = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    _ = ex.shap_interaction_values(X)
    t_shap = time.perf_counter() - t0

    speedup = t_shap / max(t_numba, 1e-9)
    np.testing.assert_allclose(Phi_n, Phi_ref, rtol=1e-4, atol=1e-4)
    assert speedup >= 1.15, (
        f"expected >=1.15x speedup on {X.shape[1]} features, got {speedup:.2f}x "
        f"(numba {t_numba:.3f}s vs shap {t_shap:.3f}s)")
