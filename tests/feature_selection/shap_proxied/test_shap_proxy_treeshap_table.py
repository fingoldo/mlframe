"""The table form of the path-dependent TreeSHAP equals the path kernel on every shape of input, and the dispatcher falls back to the path kernel where the table does not apply."""

from __future__ import annotations

import numpy as np
import pytest

xgb = pytest.importorskip("xgboost")

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_treeshap_table as table_mod
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble, treeshap_phi_base_numba, treeshap_phi_base_path
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_table import build_tables, treeshap_phi_base_table


def _ensemble(n_features: int, depth: int, trees: int, classification: bool = False, seed: int = 0):
    """``(ensemble, X)`` of a small XGBoost model; few features and a deep tree make features repeat on a path."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(400, n_features)).astype(np.float32)
    signal = X[:, 0] * 2 + X[:, 1] * X[:, 2 % n_features] - np.sin(X[:, 3 % n_features])
    if classification:
        y = (signal + 0.3 * rng.normal(size=400) > 0).astype(int)
        model = xgb.XGBClassifier(n_estimators=trees, max_depth=depth, tree_method="hist", verbosity=0).fit(X, y)
    else:
        y = signal + 0.1 * rng.normal(size=400)
        model = xgb.XGBRegressor(n_estimators=trees, max_depth=depth, tree_method="hist", verbosity=0).fit(X, y)
    return extract_ensemble(model), X


@pytest.mark.parametrize("n_features,depth,classification", [(8, 4, False), (4, 6, False), (5, 5, True), (3, 7, False)])
def test_table_equals_the_path_kernel(n_features, depth, classification):
    """Including the deep, few-feature cases where one feature repeats on a path (its splits are merged into one element)."""
    ens, X = _ensemble(n_features, depth, 25, classification)
    phi_path, base_path = treeshap_phi_base_path(ens, X)
    got = treeshap_phi_base_table(ens, X)
    assert got is not None
    np.testing.assert_allclose(got[0], phi_path, rtol=0, atol=1e-10)
    assert got[1] == base_path


def test_table_equals_the_path_kernel_with_missing_values():
    """A NaN routes to the default child in both forms: the table kernel must not fold the NaN test away."""
    ens, X = _ensemble(6, 5, 30)
    Xn = X.copy()
    Xn[np.random.default_rng(1).random(Xn.shape) < 0.2] = np.nan
    np.testing.assert_allclose(treeshap_phi_base_table(ens, Xn)[0], treeshap_phi_base_path(ens, Xn)[0], rtol=0, atol=1e-10)


def test_the_phi_sums_to_the_margin():
    """Efficiency property: base + sum of phi is the model margin for every row (the table did not change the contract)."""
    ens, X = _ensemble(6, 4, 20)
    phi, base = treeshap_phi_base_numba(ens, X)
    ref_phi, ref_base = treeshap_phi_base_path(ens, X)
    np.testing.assert_allclose(base + phi.sum(axis=1), ref_base + ref_phi.sum(axis=1), rtol=0, atol=1e-9)


def test_a_table_over_the_entry_budget_falls_back_to_the_path_kernel(monkeypatch):
    """An ensemble whose table would exceed the budget returns None from the table form; the dispatcher then answers with the path kernel."""
    ens, X = _ensemble(8, 5, 25)
    monkeypatch.setattr(table_mod, "MAX_TABLE_ENTRIES", 10)
    assert build_tables(ens) is None
    assert treeshap_phi_base_table(ens, X) is None
    np.testing.assert_allclose(treeshap_phi_base_numba(ens, X)[0], treeshap_phi_base_path(ens, X)[0], rtol=0, atol=1e-12)


def test_too_few_rows_to_repay_the_build_use_the_path_kernel(monkeypatch):
    """With the payoff share forced to zero the table is never worth building and the answer still comes from the path kernel."""
    ens, X = _ensemble(8, 5, 25)
    monkeypatch.setattr(table_mod, "PAYOFF_SHARE", 0.0)
    assert build_tables(ens, n_rows=1) is None
    assert treeshap_phi_base_table(ens, X[:1]) is None
    np.testing.assert_allclose(treeshap_phi_base_numba(ens, X[:1])[0], treeshap_phi_base_path(ens, X[:1])[0], rtol=0, atol=1e-12)


def test_unique_feature_cap_falls_back(monkeypatch):
    """A leaf with more unique features than ``MAX_UNIQUE`` keeps the path kernel."""
    ens, _ = _ensemble(8, 5, 25)
    monkeypatch.setattr(table_mod, "MAX_UNIQUE", 1)
    assert build_tables(ens) is None
