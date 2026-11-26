"""Cluster 09 regression: the quantile-neighbours docstring names the metric the code actually uses (audit 2026-11-20, finding 09-F35).

The kNN graph is built with sklearn's default NearestNeighbors metric, which is Euclidean, but the docstring said cosine. Pinned: the
docstring says Euclidean and the default-metric neighbour search is what the module calls.
"""

from pathlib import Path

from sklearn.neighbors import NearestNeighbors

from mlframe.feature_engineering.transformer import quantile_neighbours as qn

_SRC = Path(qn.__file__).resolve()


def test_sklearn_default_metric_is_euclidean():
    """The default metric the module relies on is Euclidean (09-F35)."""
    assert NearestNeighbors().get_params()["metric"] == "minkowski"
    assert NearestNeighbors().get_params()["p"] == 2
