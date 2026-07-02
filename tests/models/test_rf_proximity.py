"""Unit + biz_value tests for RF proximity matrix + Breiman outlier measure (PZAD rf)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.models.rf_proximity import (
    proximity_to_distance,
    rf_outlier_measure,
    rf_proximity_matrix,
)


# ---------------------------------------------------------------- unit
def test_proximity_from_leaf_matrix():
    # 3 rows, 4 trees; rows 0 and 1 share all leaves, row 2 shares none
    leaves = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [9, 8, 7, 6]])
    P = rf_proximity_matrix(leaves)
    assert P.shape == (3, 3)
    assert np.allclose(np.diag(P), 1.0)
    assert P[0, 1] == 1.0 and P[1, 0] == 1.0
    assert P[0, 2] == 0.0
    assert np.allclose(P, P.T)


def test_partial_proximity():
    leaves = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])  # share 2 of 4 trees
    P = rf_proximity_matrix(leaves)
    assert P[0, 1] == 0.5


def test_proximity_to_distance():
    P = np.array([[1.0, 0.75], [0.75, 1.0]])
    D = proximity_to_distance(P)
    assert np.allclose(np.diag(D), 0.0)
    assert abs(D[0, 1] - np.sqrt(0.25)) < 1e-12


def test_max_n_guard():
    leaves = np.zeros((50, 3), dtype=int)
    with pytest.raises(ValueError):
        rf_proximity_matrix(leaves, max_n=10)


def test_outlier_shape_and_mismatch():
    P = np.eye(5) + 0.1
    np.fill_diagonal(P, 1.0)
    scores = rf_outlier_measure(P)
    assert scores.shape == (5,)
    with pytest.raises(ValueError):
        rf_outlier_measure(P, y=np.zeros(3))


def test_accepts_fitted_forest():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 4))
    y = (X[:, 0] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    P = rf_proximity_matrix(rf, X)
    assert P.shape == (40, 40)
    assert np.allclose(np.diag(P), 1.0)


# ---------------------------------------------------------------- biz_value
def test_biz_val_proximity_separates_classes():
    """RF proximity is a LEARNED metric: rows of the same class land in the same leaves far more often than rows of
    different classes, so mean within-class proximity >> mean between-class proximity (the basis for RF clustering)."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(1)
    n = 200
    X = np.vstack([rng.normal(-2, 0.5, size=(n // 2, 5)), rng.normal(2, 0.5, size=(n // 2, 5))])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    rf = RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y)
    P = rf_proximity_matrix(rf, X)
    same = (y[:, None] == y[None, :])
    np.fill_diagonal(same, False)
    within = P[same].mean()
    between = P[~same & ~np.eye(n, dtype=bool)].mean()
    assert within >= between + 0.3, f"within-class prox {within:.3f} should exceed between-class {between:.3f} by >=0.3"


def test_biz_val_outlier_measure_flags_injected_outlier():
    """Breiman's proximity outlier score: a row planted far from its class in feature space is weakly connected to its
    class in forest space, so it scores far above the class median (the RF anomaly-detection use of slide 2)."""
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(2)
    n = 150
    X = np.vstack([rng.normal(0, 0.5, size=(n, 4)), rng.normal(10, 0.5, size=(n, 4))])
    y = np.array([0] * n + [1] * n)
    # inject an outlier: labeled class 0 but sitting in class-1 territory
    X_out = np.array([[10.0, 10.0, 10.0, 10.0]])
    X_all = np.vstack([X, X_out])
    y_all = np.concatenate([y, [0]])
    rf = RandomForestClassifier(n_estimators=150, random_state=0).fit(X_all, y_all)
    P = rf_proximity_matrix(rf, X_all)
    scores = rf_outlier_measure(P, y_all)
    outlier_idx = len(y_all) - 1
    class0 = np.where(y_all == 0)[0]
    assert scores[outlier_idx] == scores[class0].max(), "injected outlier must have the top within-class outlier score"
    assert scores[outlier_idx] > np.median(scores[class0]) + 3.0
