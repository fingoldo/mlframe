"""Cluster 09 regression: the batched finite-difference gradient equals the per-column loop (audit 2026-11-20, finding 09-F30)."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

import mlframe.feature_engineering.transformer.gradient_direction_agreement as gda


def _loop_gradient(model, X, is_binary, eps):
    """The pre-change implementation: one predict per perturbed column."""
    n, d = X.shape
    p_base = gda._predict(model, X, is_binary)
    grad = np.zeros((n, d), dtype=np.float32)
    for j in range(d):
        Xp = X.copy()
        Xp[:, j] += eps
        grad[:, j] = (gda._predict(model, Xp, is_binary) - p_base) / eps
    return grad


def test_batched_gradient_equals_the_loop_for_a_tree_and_a_linear_model(monkeypatch):
    """Both member kinds give the same gradient from the stacked predict and from d separate predicts, and the input matrix is left unchanged."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5)).astype(np.float32)
    y = (X[:, 0] * 2 - X[:, 1] + 0.1 * rng.normal(size=120)).astype(np.float32)
    for model in (RandomForestRegressor(n_estimators=10, max_depth=4, random_state=0).fit(X, y), Ridge().fit(X, y)):
        X_before = X.copy()
        np.testing.assert_allclose(gda._gradient(model, X, False, 0.05), _loop_gradient(model, X, False, 0.05), rtol=1e-4, atol=1e-5)
        np.testing.assert_array_equal(X, X_before)
    monkeypatch.setattr(gda, "_MAX_STACK_ELEMS", 1)  # forces the per-column fallback path
    np.testing.assert_allclose(gda._gradient(model, X, False, 0.05), _loop_gradient(model, X, False, 0.05), rtol=1e-4, atol=1e-5)
