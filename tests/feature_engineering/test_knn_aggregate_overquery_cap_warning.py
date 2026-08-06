"""FE_ROOT_B-20: knn_aggregate's group-filter overquery is capped heuristically at k*4+1. When a query's
own-group density exceeds this cap, its aggregates silently degrade to partial-k/NaN with no signal that
the CAP (not genuine data sparsity) is the cause. A warning must fire naming the affected row count.
"""

from __future__ import annotations

import logging

import numpy as np

from mlframe.feature_engineering.spatial import knn_aggregate


def test_knn_aggregate_warns_when_overquery_cap_exhausted_by_dense_same_group(caplog):
    """A ref pool where one group has far more than k*4+1 members clustered near a query row of that
    same group must trigger the overquery-cap warning, since same-group filtering can exhaust the whole
    q_k candidate window before finding k different-group neighbours."""
    rng = np.random.default_rng(0)
    k = 3
    n_dense_same_group = 50  # >> k*4+1=13, all tightly clustered near the query point
    n_other = 30

    dense_coords = rng.normal(loc=0.0, scale=0.01, size=(n_dense_same_group, 2))
    other_coords = rng.uniform(5.0, 10.0, size=(n_other, 2))
    ref_coords = np.vstack([dense_coords, other_coords])
    ref_labels = rng.uniform(0, 1, size=ref_coords.shape[0])
    ref_group_ids = np.array([0] * n_dense_same_group + [1] * n_other)

    q_coords = np.array([[0.0, 0.0]])
    q_group_ids = np.array([0])

    with caplog.at_level(logging.WARNING, logger="mlframe.feature_engineering.spatial"):
        knn_aggregate(q_coords, ref_coords, ref_labels, k=k, q_group_ids=q_group_ids, ref_group_ids=ref_group_ids)

    assert any("overquery cap" in r.message for r in caplog.records)
