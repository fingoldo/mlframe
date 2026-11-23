"""Cluster 09 regression: SMOTE never takes a row as its own neighbour, even when self is not column 0 (audit 2026-11-20, finding 09-F40).

The neighbour search is stubbed so that self sits at position 1, the situation duplicates create. A synthetic row that picked self would
equal its source row exactly; the fixed code excludes self by index, so no synthetic row equals a minority row.
"""

import numpy as np

from mlframe.feature_engineering.transformer import density_weighted_smote as dw


class _SelfSecond:
    """Stand-in for NearestNeighbors that places each row's own index at column 1."""

    def __init__(self, n_neighbors):
        """Record the requested neighbour count."""
        self.n_neighbors = n_neighbors

    def fit(self, X):
        """Store the training matrix size."""
        self.n_ = X.shape[0]
        return self

    def kneighbors(self, X):
        """Return distances and ids with self at column 1 and the others after it."""
        n = self.n_
        ids = np.zeros((n, self.n_neighbors), dtype=np.int64)
        dists = np.zeros((n, self.n_neighbors), dtype=np.float64)
        for r in range(n):
            others = [c for c in range(n) if c != r][: self.n_neighbors - 1]
            ids[r] = [others[0], r, *others[1:]]
            dists[r] = np.arange(self.n_neighbors, dtype=np.float64) + 0.5
        return dists, ids


def test_no_synthetic_row_equals_a_minority_row(monkeypatch):
    """With self at column 1, no synthetic point collapses onto its source (09-F40)."""
    monkeypatch.setattr(dw, "NearestNeighbors", _SelfSecond, raising=False)
    import sklearn.neighbors as sk_nb

    monkeypatch.setattr(sk_nb, "NearestNeighbors", _SelfSecond)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, 3)).astype(np.float64)
    out = dw._density_weighted_smote_synthesize(X, n_synthetic=40, k_neighbors=3, seed=0)
    for row in out:
        assert not np.any(np.all(np.isclose(X, row), axis=1))
