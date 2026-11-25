"""Cluster 09 regression: the quantile-neighbour graph uses the Euclidean metric the docstring now names (audit 2026-11-20, finding 09-F35).

The module docstring used to say cosine while the code used sklearn's default (Euclidean). The docstring is corrected; this test pins the
metric the code actually requests, so the two cannot drift apart again.
"""

import numpy as np
import sklearn.neighbors as skn

from mlframe.feature_engineering.transformer import quantile_neighbours as qn


def test_neighbour_graph_requests_the_default_euclidean_metric(monkeypatch):
    """The NearestNeighbors built by compute_quantile_neighbours does not override the metric, so it is Euclidean (09-F35)."""
    captured = []
    original = skn.NearestNeighbors

    def _spy(*args, **kwargs):
        """Record the constructor kwargs, then build the real estimator."""
        captured.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(skn, "NearestNeighbors", _spy)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = rng.normal(size=80)
    qn.compute_quantile_neighbours(X[20:], y[20:], X[:20], None, seed=0, k=5)
    assert captured, "the kNN graph was not built through sklearn NearestNeighbors"
    assert all("metric" not in kw for kw in captured)
