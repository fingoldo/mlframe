"""Cluster 09 regression: the pure-positive and distance SMOTE synthesisers are draw-then-vectorise and bit-identical to their row loops (audit 2026-11-20, finding 09-F28)."""

import numpy as np

from mlframe.feature_engineering.transformer.pure_pos_smote import _pure_pos_smote_synthesize
from mlframe.feature_engineering.transformer.smote_distance import _smote_synthesize


def _knn_ids(X, k_used):
    """The same kNN table the synthesisers build."""
    from sklearn.neighbors import NearestNeighbors

    return NearestNeighbors(n_neighbors=k_used).fit(X).kneighbors(X)[1]


def _old_smote_distance(X, n_syn, k, seed):
    """The row loop this replaced."""
    n_min = X.shape[0]
    k_used = min(k + 1, n_min)
    ids = _knn_ids(X, k_used)
    rng = np.random.default_rng(seed)
    out = np.zeros((n_syn, X.shape[1]), dtype=np.float32)
    for i in range(n_syn):
        s = rng.integers(0, n_min)
        row = ids[s]
        cand = row[row != s][: k_used - 1]
        if cand.size == 0:
            out[i] = X[s]
            continue
        nbr = cand[rng.integers(0, cand.size)]
        a = rng.random()
        out[i] = X[s] + a * (X[nbr] - X[s])
    return out.astype(np.float32)


def _old_pure_pos(X, centroid, n_syn, k, seed):
    """The row loop this replaced."""
    n_min = X.shape[0]
    d = np.linalg.norm(X - centroid, axis=1) + 1e-9
    w = d / d.sum()
    k_used = min(k + 1, n_min)
    ids = _knn_ids(X, k_used)
    rng = np.random.default_rng(seed)
    src = rng.choice(n_min, size=n_syn, p=w)
    out = np.zeros((n_syn, X.shape[1]), dtype=np.float32)
    for i in range(n_syn):
        s = src[i]
        row = ids[s]
        cand = row[row != s][: k_used - 1]
        if cand.size == 0:
            out[i] = X[s]
            continue
        nbr = cand[rng.integers(0, cand.size)]
        a = rng.random()
        out[i] = X[s] + a * (X[nbr] - X[s])
    return out


def test_smote_distance_matches_its_old_row_loop_exactly():
    """Same seed, same float32 minority rows (including a duplicated point): identical output."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4)).astype(np.float32)
    X[5] = X[4]
    np.testing.assert_array_equal(_smote_synthesize(X, 200, 5, 3), _old_smote_distance(X, 200, 5, 3))


def test_pure_pos_smote_matches_its_old_row_loop_exactly():
    """Same seed and centroid: identical output."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3)).astype(np.float32)
    centroid = np.zeros(3, dtype=np.float32) - 1.0
    np.testing.assert_array_equal(_pure_pos_smote_synthesize(X, centroid, 150, 4, 9), _old_pure_pos(X, centroid, 150, 4, 9))
