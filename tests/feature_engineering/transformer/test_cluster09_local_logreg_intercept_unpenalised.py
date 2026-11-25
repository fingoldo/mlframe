"""Cluster 09 regression: the local logistic solver does not shrink its intercept toward zero (audit 2026-11-20, finding 09-F44).

With every local label positive and a strong ridge, the unpenalised intercept carries the probability toward one. Under the old penalty the
intercept was pulled toward zero and the query probability stayed near one half.
"""

import numpy as np

from mlframe.feature_engineering.transformer.local_classifier import _solve_weighted_logreg


def test_intercept_is_not_pulled_to_zero_by_the_ridge():
    """All-positive neighbourhood with a strong ridge: the query probability is near one, not near one half (09-F44)."""
    n = 20
    X_local = np.zeros((n, 1), dtype=np.float32)
    y_local = np.ones(n, dtype=np.float32)
    w = np.ones(n, dtype=np.float32)
    x_query = np.zeros(1, dtype=np.float32)
    proba, *_ = _solve_weighted_logreg(X_local, y_local, w, x_query, n_iter=30, ridge=10.0)
    assert proba > 0.9


def test_slope_is_still_shrunk_by_the_ridge():
    """Control: a strong ridge still shrinks the slope toward zero on a separable-looking feature (09-F44)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 1)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.float32)
    w = np.ones(60, dtype=np.float32)
    weak, *_ = _solve_weighted_logreg(x, y, w, np.array([3.0], dtype=np.float32), n_iter=30, ridge=0.01)
    strong, *_ = _solve_weighted_logreg(x, y, w, np.array([3.0], dtype=np.float32), n_iter=30, ridge=500.0)
    assert strong < weak
