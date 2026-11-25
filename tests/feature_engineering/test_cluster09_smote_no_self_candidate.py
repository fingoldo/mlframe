"""Cluster 09 regression: SMOTE never interpolates a point with itself (audit 2026-11-20, finding 09-F40).

The candidates used to be taken as neighbour columns 1..k, which assumes the point itself sits in column 0. With duplicate points it does not.
This test makes self the SECOND neighbour of every row. The old slice then includes self, and interpolating a point with itself returns the point
exactly. The fixed code excludes self by index, so no synthetic row equals its source.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from mlframe.feature_engineering.transformer.pure_pos_smote import _pure_pos_smote_synthesize
from mlframe.feature_engineering.transformer.smote_distance import _smote_synthesize


def _self_second_kneighbors(self, X=None, n_neighbors=None, return_distance=True):
    """Return neighbour lists where each row's own index is in position 1 and its predecessor in position 0."""
    n = X.shape[0] if X is not None else self._fit_X.shape[0]
    k = self.n_neighbors
    ids = np.empty((n, k), dtype=np.intp)
    for i in range(n):
        order = [(i - 1) % n, i] + [j for j in range(n) if j not in ((i - 1) % n, i)]
        ids[i] = order[:k]
    dists = np.ones((n, k), dtype=np.float64)
    return (dists, ids) if return_distance else ids


def _minority(seed: int = 0):
    """Distinct points, so an exact copy of a training row can only come from interpolating a point with itself."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(12, 2))


def _has_exact_copy_of_a_source(out: np.ndarray, X: np.ndarray) -> bool:
    """True when any synthetic row equals a training row at float32 precision."""
    Xf = X.astype(np.float32)
    return any(np.any(np.all(np.isclose(Xf, row, rtol=0, atol=1e-7), axis=1)) for row in out)


def test_smote_distance_never_returns_a_source_copy(monkeypatch):
    """With self in the second neighbour slot, synthetic rows are not copies of their source (09-F40)."""
    monkeypatch.setattr(NearestNeighbors, "kneighbors", _self_second_kneighbors)
    X = _minority()
    out = _smote_synthesize(X, 400, 3, 0)
    assert not _has_exact_copy_of_a_source(out, X)


def test_pure_pos_smote_never_returns_a_source_copy(monkeypatch):
    """Same property for the pure-positive synthesiser (09-F40)."""
    monkeypatch.setattr(NearestNeighbors, "kneighbors", _self_second_kneighbors)
    X = _minority()
    out = _pure_pos_smote_synthesize(X, np.zeros(2), 400, 3, 0)
    assert not _has_exact_copy_of_a_source(out, X)
