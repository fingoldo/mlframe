"""FCBF-style chain rejection of ``_component_cluster`` (CLUSTERING_STABILITY-2).

Single-linkage components can chain A-B-C into one group through a bridge member B although A and C do not correlate directly. The ordered-relevance pruning keeps only the
members that correlate with the REPRESENTATIVE, so the chain loses its far end, while a cluster whose members all reflect one latent factor is kept whole.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._cluster_aggregate import _component_cluster


def _cluster(X: pd.DataFrame, relevance: dict, *, corr_threshold: float):
    """Run ``_component_cluster`` on the whole frame as one pool and one component, with minimum cluster size 2."""
    cols = list(X.columns)
    pool = list(range(len(cols)))
    corr = np.corrcoef(X.to_numpy(), rowvar=False)
    return corr, _component_cluster(
        pool, pool, relevance, corr, X, cols, set(),
        min_cluster_size=2, corr_threshold=corr_threshold, max_cluster_size=10, homogeneity_tau=0.5,
    )


def test_bridge_member_does_not_chain_two_uncorrelated_ends_into_one_cluster():
    """A and B correlate, B and C correlate, A and C do not: the representative A keeps B and drops C."""
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.standard_normal(n)
    c = rng.standard_normal(n)
    b = 0.7 * a + 0.7 * c + 0.1 * rng.standard_normal(n)
    X = pd.DataFrame({"a": a, "b": b, "c": c})
    corr, cluster = _cluster(X, {0: 0.9, 1: 0.5, 2: 0.4}, corr_threshold=0.6)
    assert abs(corr[0, 2]) < 0.1 and abs(corr[0, 1]) > 0.6 and abs(corr[1, 2]) > 0.6, "fixture: a-b and b-c correlate, a-c does not"
    assert cluster is not None
    assert cluster["rep"] == 0
    assert 2 not in cluster["members"], "C is only linked through the bridge B and must not join A's cluster"
    assert 1 in cluster["members"]


def test_reflections_of_one_latent_factor_stay_one_cluster():
    """Three noisy copies of one factor correlate pairwise, so none is pruned and the cluster is unidimensional."""
    rng = np.random.default_rng(1)
    n = 4000
    z = rng.standard_normal(n)
    X = pd.DataFrame({f"r{i}": z + 0.2 * rng.standard_normal(n) for i in range(3)})
    _corr, cluster = _cluster(X, {0: 0.9, 1: 0.8, 2: 0.7}, corr_threshold=0.8)
    assert cluster is not None
    assert sorted(cluster["members"]) == [0, 1, 2]
