"""Cluster 25 regression: the prefilter keeps the smallest qualifying size even when candidate_sizes is unsorted (audit 2026-11-20, finding 25-F57)."""

import numpy as np

from mlframe.feature_selection import ridge_forward_prefilter as rfp


def test_smallest_qualifying_size_wins_for_unsorted_input(monkeypatch):
    """With sizes given as [10, 2] and size 2 clearing the floor, two features are kept, not ten (25-F57)."""
    monkeypatch.setattr(rfp, "nested_pool_cv_scores", lambda *a, **k: {10: 0.91, 2: 0.90})
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 10))
    y = X[:, 0] + rng.normal(scale=0.1, size=60)
    names = [f"f{i}" for i in range(10)]
    kept = rfp.ridge_coefficient_prefilter(X, y, names, candidate_sizes=[10, 2], tol=0.05)
    assert len(kept) == 2
