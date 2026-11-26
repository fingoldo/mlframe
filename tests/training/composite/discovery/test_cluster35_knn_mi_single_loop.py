"""Audit 2026-11-20 finding 35-F38: the kNN MI aggregate equals the per-feature vector's aggregate, from one shared loop.

_mi_to_target's kNN branch now calls _mi_per_feature_knn, so the pair mask lives in one place. The result must match the aggregate of the
per-feature vector exactly, including columns with NaN.
"""

import numpy as np

from mlframe.training.composite.discovery.screening import _mi_per_feature_knn, _mi_to_target


def test_knn_aggregate_matches_per_feature_vector_including_nan():
    """Mean aggregate of the per-feature kNN vector equals _mi_to_target's kNN result, with a NaN-bearing column."""
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 4))
    X[::7, 2] = np.nan
    y = X[:, 0] + 0.5 * rng.normal(size=n)
    vec = _mi_per_feature_knn(X, y, n_neighbors=3, random_state=0)
    agg = _mi_to_target(X, y, n_neighbors=3, random_state=0, estimator="knn", aggregation="mean")
    assert agg == float(np.mean(vec))
