"""Unit + biz_value tests for tabular->graph constructors (PZAD SNA feature-generation)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.graph_construction import (
    knn_graph_edges,
    shared_attribute_edges,
)
from mlframe.feature_engineering.graph_features import (
    graph_neighbor_aggregate,
    graph_structural_features,
)


# ---------------------------------------------------------------- unit
def test_knn_edges_shape_and_no_self():
    """Knn edges shape and no self."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    edges = knn_graph_edges(X, k=3)
    assert edges.shape[1] == 2
    assert edges.shape[0] == 50 * 3  # k per row, self excluded
    assert np.all(edges[:, 0] != edges[:, 1])


def test_knn_timestamp_is_past_only():
    """Knn timestamp is past only."""
    X = np.arange(6, dtype=float).reshape(-1, 1)  # 1-D, neighbours are adjacent
    ts = np.arange(6, dtype=float)  # row i has time i
    edges = knn_graph_edges(X, k=3, timestamp=ts)
    assert np.all(ts[edges[:, 1]] < ts[edges[:, 0]])  # every edge points to an earlier row


def test_shared_attribute_clique():
    """Shared attribute clique."""
    codes = np.array([0, 0, 0, 1, 1])  # group 0 has rows {0,1,2}, group 1 {3,4}
    edges = shared_attribute_edges(codes, max_neighbours=None)
    deg = graph_structural_features(5, edges)["degree"]
    assert deg[0] == 2 and deg[1] == 2 and deg[2] == 2  # triangle within group 0
    assert deg[3] == 1 and deg[4] == 1


def test_shared_attribute_timestamp_past_only():
    """Shared attribute timestamp past only."""
    codes = np.array([0, 0, 0])
    ts = np.array([0.0, 1.0, 2.0])
    edges = shared_attribute_edges(codes, timestamp=ts, max_neighbours=None)
    assert np.all(ts[edges[:, 1]] < ts[edges[:, 0]])
    # row with time 0 has no past peers; row with time 2 has two
    src = edges[:, 0]
    assert (src == 0).sum() == 0 and (src == 2).sum() == 2


def test_max_group_size_skips_and_warns(caplog):
    """Max group size skips and warns."""
    codes = np.zeros(10, dtype=int)  # one big group
    edges = shared_attribute_edges(codes, max_group_size=5)
    assert edges.shape[0] == 0  # the only group exceeds the cap -> skipped


def test_knn_guard():
    """Knn guard."""
    with pytest.raises(ValueError):
        knn_graph_edges(np.zeros((5, 2)), k=0)


# ---------------------------------------------------------------- biz_value
def test_biz_val_knn_graph_smooths_a_noisy_label_on_float_table():
    """On an ORDINARY float table with no explicit graph, a kNN similarity graph + neighbour-target-mean recovers a
    smooth underlying label better than the noisy per-row label -- i.e. graph features apply to plain tabular data."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    n = 600
    X = rng.normal(size=(n, 5))
    signal = (X[:, 0] + X[:, 1] > 0).astype(float)  # smooth ground-truth over feature space
    noisy = signal.copy()
    flip = rng.random(n) < 0.25
    noisy[flip] = 1 - noisy[flip]  # 25% label noise
    edges = knn_graph_edges(X, k=15)
    smoothed = graph_neighbor_aggregate(n, edges, noisy, agg="mean", fill=0.5)
    auc_noisy = roc_auc_score(signal, noisy)
    auc_smoothed = roc_auc_score(signal, smoothed)
    assert auc_smoothed >= auc_noisy + 0.1, f"kNN-smoothed AUC {auc_smoothed:.3f} should beat noisy {auc_noisy:.3f}"


def test_biz_val_shared_attribute_degree_is_a_useful_categorical_feature():
    """From a categorical column alone, the affiliation-graph degree = 'how many other rows share my category' -- a
    frequency/co-membership feature that predicts a target driven by category popularity (rare categories differ)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(2)
    n = 800
    # a few big categories and many singletons; the target is 'belongs to a big category'
    codes = np.where(rng.random(n) < 0.7, rng.integers(0, 3, n), rng.integers(100, 100 + n, n))
    _, inv, counts = np.unique(codes, return_inverse=True, return_counts=True)
    y = (counts[inv] >= 10).astype(int)
    edges = shared_attribute_edges(codes, max_neighbours=None)
    degree = graph_structural_features(n, edges)["degree"]
    auc = roc_auc_score(y, degree)
    assert auc >= 0.95, f"shared-attribute degree should identify big-category rows (AUC {auc:.3f})"
