"""The table form of the TreeSHAP interaction tensor equals the conditioned-pass reference kernel (also with NaN and repeated path features) and falls back to it where the tables do not apply."""

from __future__ import annotations

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_treeshap_interactions_table as itab
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions import interaction_tensor_numba, interaction_tensor_reference
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_table import build_interaction_tables, interaction_tensor_table


def _ensemble(n_features: int, depth: int, trees: int, classification: bool = False, seed: int = 0):
    """``(ensemble, X)`` of a small XGBoost model; few features and a deep tree make features repeat on a path."""
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
def test_interaction_table_equals_the_reference(n_features, depth, classification):
    """Including the deep, few-feature cases where one feature repeats on a path (its splits merge into one element)."""
    ens, X = _ensemble(n_features, depth, 15, classification)
    Phi_ref, phi_ref, base_ref = interaction_tensor_reference(ens, X)
    got = interaction_tensor_table(ens, X)
    assert got is not None
    Phi, phi, base = got
    assert base == base_ref
    np.testing.assert_allclose(Phi, Phi_ref, rtol=0, atol=1e-10)
    np.testing.assert_allclose(phi, phi_ref, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(Phi, Phi.transpose(0, 2, 1))
    np.testing.assert_allclose(Phi.sum(axis=2), phi, rtol=0, atol=1e-10)


def test_interaction_table_routes_missing_values_like_the_reference():
    """NaN rows follow the default child in both kernels (a fast-math flag would fold the NaN test)."""
    ens, X = _ensemble(6, 5, 15)
    X = X.copy()
    X[np.random.default_rng(3).random(X.shape) < 0.2] = np.nan
    Phi_ref, phi_ref, _ = interaction_tensor_reference(ens, X)
    Phi, phi, _ = interaction_tensor_table(ens, X)
    assert np.isfinite(Phi).all()
    np.testing.assert_allclose(Phi, Phi_ref, rtol=0, atol=1e-10)
    np.testing.assert_allclose(phi, phi_ref, rtol=0, atol=1e-10)


def test_dispatcher_answers_with_the_table_and_the_reference_agrees():
    """``interaction_tensor_numba`` takes the table branch on a fitting ensemble."""
    ens, X = _ensemble(8, 4, 15)
    Phi_ref = interaction_tensor_reference(ens, X)[0]
    np.testing.assert_allclose(interaction_tensor_numba(ens, X)[0], Phi_ref, rtol=0, atol=1e-10)


def test_entry_budget_falls_back_to_the_reference(monkeypatch):
    """Over the interaction-table budget the table form returns None and the dispatcher answers with the reference."""
    ens, X = _ensemble(8, 5, 15)
    monkeypatch.setattr(itab, "MAX_ITABLE_ENTRIES", 10)
    assert build_interaction_tables(ens) is None
    assert interaction_tensor_table(ens, X) is None
    np.testing.assert_allclose(interaction_tensor_numba(ens, X)[0], interaction_tensor_reference(ens, X)[0], rtol=0, atol=1e-12)


def test_too_few_rows_to_repay_the_build_use_the_reference(monkeypatch):
    """With the payoff share forced to zero the tables are never worth building."""
    ens, X = _ensemble(8, 5, 15)
    monkeypatch.setattr(itab, "PAYOFF_SHARE", 0.0)
    assert build_interaction_tables(ens, n_rows=1) is None
    assert interaction_tensor_table(ens, X[:1]) is None
    np.testing.assert_allclose(interaction_tensor_numba(ens, X[:1])[0], interaction_tensor_reference(ens, X[:1])[0], rtol=0, atol=1e-12)
