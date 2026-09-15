"""Cluster stability selection groups a column with its tiny-scale rescaled duplicate.

The correlation step z-scored with ``std + 1e-12``. For a column whose spread is ~1e-13 the pad dominates the true std, the z-scores shrink
by ~10x, its correlations collapse, and the exact rescaled duplicate of a selected column survives as its own singleton cluster.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._stability_cluster import cluster_stability_selection


def _first_col_selector(X_sub, y_sub):
    """A trivial base selector: always pick column 0."""
    return np.array([0])


def test_stability_cluster_groups_rescaled_duplicate():
    """x and x * 1e-13 are perfectly correlated, so they must share a cluster; an independent column must not join them."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    X = np.column_stack([x, x * 1e-13, rng.normal(size=500)])
    y = (x > 0).astype(np.int64)
    _, _, info = cluster_stability_selection(X, y, _first_col_selector, n_bootstrap=4, return_clusters=True, rng_seed=0)
    cid = info["cluster_id"]
    assert cid[0] == cid[1], f"a column and its 1e-13-scaled duplicate were split into different clusters: {cid}"
    assert cid[2] != cid[0]


def test_constant_column_stays_a_singleton():
    """Control: a constant column has no correlation structure and must not merge with anything."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    X = np.column_stack([x, np.full(500, 3.0), x * 2.0])
    y = (x > 0).astype(np.int64)
    _, _, info = cluster_stability_selection(X, y, _first_col_selector, n_bootstrap=4, return_clusters=True, rng_seed=0)
    cid = info["cluster_id"]
    assert cid[0] == cid[2]
    assert cid[1] not in (cid[0], cid[2])
