"""Unit + biz_value tests for graph->tabular feature generation (PZAD SNA)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.graph_features import (
    graph_neighbor_aggregate,
    graph_structural_features,
)


# ---------------------------------------------------------------- unit
def test_triangle_clustering_is_one():
    """Triangle clustering is one."""
    edges = np.array([[0, 1], [1, 2], [0, 2]])  # a triangle
    f = graph_structural_features(3, edges)
    assert np.allclose(f["clustering"], 1.0)
    assert np.allclose(f["triangles"], 1.0)
    assert np.allclose(f["degree"], 2.0)


def test_star_center_clustering_is_zero():
    """Star center clustering is zero."""
    edges = np.array([[0, 1], [0, 2], [0, 3]])  # star, center 0
    f = graph_structural_features(4, edges)
    assert f["clustering"][0] == 0.0  # center's neighbours are not connected
    assert f["degree"][0] == 3.0
    assert f["triangles"][0] == 0.0


def test_neighbor_count_equals_degree():
    """Neighbor count equals degree."""
    edges = np.array([[0, 1], [0, 2], [1, 2]])
    cnt = graph_neighbor_aggregate(3, edges, np.zeros(3), agg="count")
    deg = graph_structural_features(3, edges)["degree"]
    assert np.array_equal(cnt, deg)


def test_neighbor_mean_label():
    # node 0 connected to 1 (label 1) and 2 (label 0) -> mean 0.5
    """Neighbor mean label."""
    edges = np.array([[0, 1], [0, 2]])
    labels = np.array([9.0, 1.0, 0.0])  # node 0's own label irrelevant (only neighbours aggregated)
    m = graph_neighbor_aggregate(3, edges, labels, agg="mean")
    assert m[0] == 0.5


def test_weighted_mean_uses_edge_weights():
    """Weighted mean uses edge weights."""
    edges = np.array([[0, 1], [0, 2]])
    w = np.array([3.0, 1.0])  # neighbour 1 weighted 3x
    vals = np.array([0.0, 1.0, 0.0])
    wm = graph_neighbor_aggregate(3, edges, vals, agg="wmean", weights=w)
    assert abs(wm[0] - (3 * 1 + 1 * 0) / 4.0) < 1e-12


def test_isolated_node_gets_fill_and_guards():
    """Isolated node gets fill and guards."""
    edges = np.array([[0, 1]])
    m = graph_neighbor_aggregate(3, edges, np.array([1.0, 1.0, 1.0]), agg="mean", fill=-1.0)
    assert m[2] == -1.0  # node 2 isolated
    with pytest.raises(ValueError):
        graph_neighbor_aggregate(3, edges, np.zeros(3), agg="nope")
    with pytest.raises(ValueError):
        graph_neighbor_aggregate(3, edges, np.zeros(2))  # length mismatch
    with pytest.raises(ValueError):
        graph_structural_features(1, np.array([[0, 5]]))  # endpoint out of range


# ---------------------------------------------------------------- biz_value
def test_biz_val_homophily_neighbor_label_predicts_own_label():
    """Homophily / social-influence feature (slides 60-62): in a graph where the label is community-correlated, the
    fraction of a node's neighbours that carry the label is a strong LEAKAGE-SAFE predictor of the node's own label
    (it aggregates only OTHER nodes' labels) -- far better than degree, which is label-agnostic here."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    n = 400
    half = n // 2
    y = np.array([1] * half + [0] * half)
    # dense intra-community edges, sparse inter-community -> label spreads along edges (homophily)
    edges = []
    for _ in range(3000):
        if rng.random() < 0.9:  # intra-community edge
            c = rng.integers(0, 2)
            lo, hi = (0, half) if c == 1 else (half, n)
            a, b = rng.integers(lo, hi), rng.integers(lo, hi)
        else:  # inter-community edge
            a, b = rng.integers(0, half), rng.integers(half, n)
        if a != b:
            edges.append((a, b))
    edges = np.array(edges)
    homophily = graph_neighbor_aggregate(n, edges, y.astype(float), agg="mean", fill=0.5)
    degree = graph_structural_features(n, edges)["degree"]
    auc_homophily = roc_auc_score(y, homophily)
    auc_degree = roc_auc_score(y, degree)
    assert auc_homophily >= 0.9, f"neighbour-label homophily AUC {auc_homophily:.3f} should be >=0.9"
    assert auc_homophily >= auc_degree + 0.25, f"homophily {auc_homophily:.3f} must beat degree {auc_degree:.3f}"


def test_biz_val_clustering_separates_clique_from_bridge():
    """The clustering coefficient is a feature (slide 46): clique members score ~1 while a bridge node linking two
    cliques scores low -- so it separates 'embedded in a dense group' from 'connector', which degree alone cannot."""
    # two triangles {0,1,2} and {3,4,5}, bridged by node 6 connected to 0 and 3
    edges = np.array([[0, 1], [1, 2], [0, 2], [3, 4], [4, 5], [3, 5], [6, 0], [6, 3]])
    f = graph_structural_features(7, edges)
    clique_members = f["clustering"][[1, 2, 4, 5]]  # pure triangle members
    bridge = f["clustering"][6]
    assert np.all(clique_members >= 0.99), "pure clique members should have clustering ~1"
    assert bridge <= 0.01, f"bridge node clustering {bridge:.3f} should be ~0"
    # node 6 has the same degree (2) as the clique members but a very different clustering -> distinct signal
    assert f["degree"][6] == f["degree"][1]


# ---------------------------------------------------------------- sum/max/min agg + directed (previously untested)


def test_neighbor_aggregate_sum():
    """agg='sum': raw sum of neighbour values, not averaged."""
    edges = np.array([[0, 1], [0, 2]])
    vals = np.array([0.0, 3.0, 4.0])
    s = graph_neighbor_aggregate(3, edges, vals, agg="sum")
    assert s[0] == 7.0


def test_neighbor_aggregate_max_and_min():
    """agg='max'/'min': the largest/smallest neighbour value."""
    edges = np.array([[0, 1], [0, 2]])
    vals = np.array([0.0, 3.0, 4.0])
    assert graph_neighbor_aggregate(3, edges, vals, agg="max")[0] == 4.0
    assert graph_neighbor_aggregate(3, edges, vals, agg="min")[0] == 3.0


def test_neighbor_aggregate_max_min_isolated_node_gets_fill():
    """An isolated node under agg='max'/'min' gets the fill value, not a spurious 0/-inf."""
    edges = np.array([[0, 1]])
    vals = np.array([1.0, 1.0, 1.0])
    assert graph_neighbor_aggregate(3, edges, vals, agg="max", fill=-99.0)[2] == -99.0
    assert graph_neighbor_aggregate(3, edges, vals, agg="min", fill=-99.0)[2] == -99.0


def test_neighbor_aggregate_directed_is_asymmetric():
    """directed=True: node aggregates are computed only over OUTGOING edges, not symmetrized."""
    # 0 -> 1 only: node 0 has an outgoing neighbour (1), node 1 has none.
    edges = np.array([[0, 1]])
    vals = np.array([0.0, 5.0])
    directed = graph_neighbor_aggregate(2, edges, vals, agg="mean", directed=True, fill=-1.0)
    assert directed[0] == 5.0  # 0's only outgoing neighbour is 1
    assert directed[1] == -1.0  # 1 has no outgoing edges -> fill
    # Undirected: the same edge list treats the connection as symmetric -> both nodes see each other.
    undirected = graph_neighbor_aggregate(2, edges, vals, agg="mean", directed=False, fill=-1.0)
    assert undirected[0] == 5.0
    assert undirected[1] == 0.0


def test_structural_features_strength_is_weighted_degree():
    """graph_structural_features' 'strength' is the sum of incident edge weights, distinct from plain degree."""
    edges = np.array([[0, 1], [0, 2]])
    weights = np.array([2.0, 5.0])
    f = graph_structural_features(3, edges, weights=weights)
    assert f["degree"][0] == 2.0
    assert f["strength"][0] == 7.0  # 2.0 + 5.0
    assert f["strength"][1] == 2.0
    assert f["strength"][2] == 5.0


def test_structural_features_directed_uses_symmetric_view_for_clustering():
    """directed=True: degree/strength come from the directed CSR, but clustering/triangles still use the
    symmetrized (undirected) view -- per the function's own docstring."""
    # A directed 3-cycle: 0->1->2->0. Each node has out-degree 1 (directed), but the undirected view
    # (used for clustering) sees a genuine triangle.
    edges = np.array([[0, 1], [1, 2], [2, 0]])
    f = graph_structural_features(3, edges, directed=True)
    assert np.all(f["degree"] == 1.0)  # directed out-degree
    assert np.allclose(f["clustering"], 1.0)  # undirected view: a full triangle
    assert np.allclose(f["triangles"], 1.0)
