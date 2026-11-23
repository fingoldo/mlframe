"""Cluster 26 regression: DBSCAN noise is not scored as a cluster in the silhouette (audit 2026-11-20, finding 26-F15)."""

import numpy as np
from sklearn import metrics

from mlframe.preprocessing.cluster import _silhouette_without_noise


def test_noise_points_are_excluded_from_silhouette():
    """The silhouette equals the one over the clustered points only (26-F15)."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (20, 2)), rng.normal(5, 0.1, (20, 2)), rng.normal(-5, 4.0, (10, 2))])
    labels = np.array([0] * 20 + [1] * 20 + [-1] * 10)
    keep = labels != -1
    assert np.isclose(_silhouette_without_noise(X, labels), metrics.silhouette_score(X[keep], labels[keep]))


def test_single_real_cluster_is_undefined():
    """With one real cluster plus noise the silhouette is undefined, returned as None (26-F15)."""
    X = np.arange(12, dtype=float).reshape(6, 2)
    assert _silhouette_without_noise(X, np.array([0, 0, 0, -1, -1, -1])) is None
