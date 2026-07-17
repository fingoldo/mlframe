"""Unit + biz_value tests for pairwise link-prediction features (PZAD LPP)."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.graph_features import link_prediction_features


# ---------------------------------------------------------------- unit
def test_common_neighbors_and_jaccard():
    # x=0, y=1 share neighbours {2,3}; 0~{2,3}, 1~{2,3} -> cn=2, union=2, jaccard=1
    edges = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])
    f = link_prediction_features(4, edges, np.array([[0, 1]]))
    assert f["common_neighbors"][0] == 2.0
    assert f["jaccard"][0] == 1.0
    assert f["preferential_attachment"][0] == 4.0  # deg0=2 * deg1=2


def test_no_common_neighbors():
    edges = np.array([[0, 2], [1, 3]])
    f = link_prediction_features(4, edges, np.array([[0, 1]]))
    assert f["common_neighbors"][0] == 0.0
    assert f["jaccard"][0] == 0.0
    assert f["adamic_adar"][0] == 0.0
    assert f["resource_allocation"][0] == 0.0


def test_adamic_adar_and_resource_allocation_known():
    # one common neighbour z=2 with degree 2 -> AA = 1/log(2), RA = 1/2
    edges = np.array([[0, 2], [1, 2]])
    f = link_prediction_features(3, edges, np.array([[0, 1]]))
    assert abs(f["adamic_adar"][0] - 1.0 / np.log(2)) < 1e-12
    assert abs(f["resource_allocation"][0] - 0.5) < 1e-12


def test_rare_common_neighbor_weighted_higher_by_adamic_adar():
    # pair A shares a LOW-degree hub, pair B shares a HIGH-degree hub -> A's Adamic/Adar > B's
    edges = np.array(
        [
            [0, 10],
            [1, 10],  # 10 is a degree-2 hub shared by (0,1)
            [2, 11],
            [3, 11],
            [4, 11],
            [5, 11],
            [6, 11],
            [7, 11],
        ],  # 11 is a degree-6 hub shared by (2,3)
    )
    f = link_prediction_features(12, edges, np.array([[0, 1], [2, 3]]))
    assert f["common_neighbors"][0] == f["common_neighbors"][1] == 1.0
    assert f["adamic_adar"][0] > f["adamic_adar"][1], "sharing a rarer hub should weigh more"


def test_guard_out_of_range():
    with pytest.raises(ValueError):
        link_prediction_features(3, np.array([[0, 1]]), np.array([[0, 5]]))


# ---------------------------------------------------------------- biz_value
def test_biz_val_adamic_adar_ranks_held_out_edges_above_non_edges():
    """The canonical link-prediction validation: hide a set of real edges and sample random non-edges; the Adamic/Adar
    score (computed on the reduced graph) ranks the hidden true edges well above the random non-edges (AUC high)."""
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    n = 300
    half = n // 2
    # two communities: dense intra-community links -> many common neighbours within a community
    all_edges = set()
    for _ in range(4000):
        c = rng.integers(0, 2)
        lo, hi = (0, half) if c == 0 else (half, n)
        a, b = int(rng.integers(lo, hi)), int(rng.integers(lo, hi))
        if a != b:
            all_edges.add((min(a, b), max(a, b)))
    all_edges = list(all_edges)
    rng.shuffle(all_edges)
    n_hold = 300
    held_out = all_edges[:n_hold]  # positives: real edges removed from the graph
    train_edges = np.array(all_edges[n_hold:])
    # negatives: random non-edges
    edge_set = set(all_edges)
    negatives = []
    while len(negatives) < n_hold:
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b and (min(a, b), max(a, b)) not in edge_set:
            negatives.append((a, b))
    pairs = np.array(held_out + negatives)
    y = np.array([1] * n_hold + [0] * n_hold)
    feats = link_prediction_features(n, train_edges, pairs)
    auc_aa = roc_auc_score(y, feats["adamic_adar"])
    auc_pa = roc_auc_score(y, feats["preferential_attachment"])
    # measured ~0.73 on this random-within-community graph; floor 0.68 (a genuine ranking signal, not perfect)
    assert auc_aa >= 0.68, f"Adamic/Adar link-prediction AUC {auc_aa:.3f} should be >=0.68"
    # the neighbour-overlap features carry the real link signal; degree-only preferential-attachment is near-random here
    assert auc_aa >= auc_pa + 0.15, f"Adamic/Adar {auc_aa:.3f} must beat preferential-attachment {auc_pa:.3f}"
