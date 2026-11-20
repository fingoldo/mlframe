"""Regression tests: log-link xgboost base offset and the parallel-kernel guard on the interaction entry point."""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

pytest.importorskip("xgboost")
pytest.importorskip("numba")


def _data(n=400, p=5, seed=0):
    """Small positive-count regression frame."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = rng.poisson(np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1] + 1.0)).astype(float)
    return X, y


@pytest.mark.parametrize("objective", ["count:poisson", "reg:gamma"])
def test_log_link_objective_base_is_log_of_base_score(objective):
    """base + phi.sum(1) must equal the xgboost margin for a log-link objective (base_score is stored in output space)."""
    from xgboost import XGBRegressor

    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble, treeshap_phi_base_numba

    X, y = _data()
    if objective == "reg:gamma":
        y = y + 1.0
    m = XGBRegressor(objective=objective, n_estimators=8, max_depth=3, learning_rate=0.3, tree_method="hist", base_score=2.5)
    m.fit(X, y)
    ens = extract_ensemble(m)
    phi, base = treeshap_phi_base_numba(ens, X)
    margin = m.get_booster().predict(__import__("xgboost").DMatrix(X), output_margin=True)
    assert np.allclose(base + phi.sum(1), margin, atol=1e-4)


def test_interaction_tensor_enters_parallel_kernel_guard(monkeypatch):
    """interaction_tensor_numba must hold the process-wide parallel-kernel guard like treeshap_phi_base_numba does."""
    from xgboost import XGBRegressor

    import mlframe._numba_parallel_guard as guard
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions import interaction_tensor_numba

    X, y = _data()
    m = XGBRegressor(n_estimators=3, max_depth=2, tree_method="hist").fit(X, y)
    ens = extract_ensemble(m)
    entered = []

    @contextlib.contextmanager
    def _spy():
        """Record each entry."""
        entered.append(1)
        yield

    monkeypatch.setattr(guard, "parallel_kernel_entry", _spy)
    interaction_tensor_numba(ens, X[:20])
    assert entered == [1]
