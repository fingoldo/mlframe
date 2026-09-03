"""FE_TRANSFORMER_B-3: geodesic_kgraph's kNN graph construction queried a NearestNeighbors index fit on
Xt_s with Xt_s itself using n_neighbors=k_graph (not k_graph+1), so each row's self-match occupied one of
its k_graph slots, silently weakening graph connectivity. At k_graph=1 this is total: every row's sole
"neighbor" is itself (distance 0), so the kNN graph carries essentially no real edges and multi-source
Dijkstra can't traverse between clusters at all -- every row falls back to the "far" sentinel regardless
of true proximity to the target set."""

from __future__ import annotations

import numpy as np

from mlframe.feature_engineering.transformer.geodesic_kgraph import compute_geodesic_kgraph_features


def test_k_graph_1_still_distinguishes_near_from_far_query_rows():
    """With a real self-excluded 1-NN graph on a single connected blob, a query row near the target
    subset (rows closest to the origin) must get a smaller mean geodesic distance than one far from it.

    At k_graph=1, the PRE-FIX bug leaves every train row with ZERO real graph edges (its sole
    "neighbor" is itself at distance 0) -- the kNN graph is literally empty of off-diagonal edges, so
    Dijkstra can never reach a non-target row, every row's distance collapses to the same fallback
    value, and near/far queries become indistinguishable (mean_geo identically 0.0 for all rows)."""
    rng = np.random.default_rng(0)
    # A single well-connected blob (not two separated clusters -- at k_graph=1 two far-apart clusters
    # would stay graph-disconnected regardless of the self-match fix, which would falsely look like
    # the same failure mode this test targets).
    X_train = rng.uniform(-10.0, 10.0, size=(150, 2)).astype(np.float32)
    dist_from_origin = np.linalg.norm(X_train, axis=1)
    # Target/positive class: the 20 rows closest to the origin.
    target_mask = dist_from_origin <= np.quantile(dist_from_origin, 20.0 / 150.0)
    y_train = target_mask.astype(np.float32)

    near_query = rng.uniform(-0.3, 0.3, size=(5, 2)).astype(np.float32)
    far_query = np.array([[9.5, 9.5]] * 5, dtype=np.float32) + rng.uniform(-0.3, 0.3, size=(5, 2)).astype(np.float32)
    X_query = np.concatenate([near_query, far_query], axis=0)

    out = compute_geodesic_kgraph_features(X_train, y_train, X_query, seed=0, task="binary", k_graph=1, k_query=5, standardize=False)
    mean_geo = out["geo_mean"].to_numpy()

    near_mean = mean_geo[:5].mean()
    far_mean = mean_geo[5:].mean()
    assert far_mean > 0.0, "the graph must have real connectivity (a self-only graph collapses everything to the target rows' own 0-distance)"
    assert near_mean < far_mean, f"expected rows near the target subset to have smaller geodesic distance; got near={near_mean:.3f} far={far_mean:.3f}"
