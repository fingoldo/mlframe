"""FE_ROOT_B-19: knn_within_bucket_aggregate must support the same agg_fns as its sibling knn_aggregate
(min/max/p10/p90 in addition to median/mean/std/iqr) -- previously only the canonical four were supported,
an undocumented API-surface inconsistency between the two parallel functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.spatial import knn_aggregate, knn_within_bucket_aggregate


@pytest.mark.parametrize("agg", ["min", "max", "p10", "p90"])
def test_knn_within_bucket_aggregate_supports_min_max_p10_p90(agg):
    """Each of the previously-unsupported aggregators is now accepted and matches knn_aggregate on a
    single-bucket dataset (bucket restriction is a no-op when every row shares one bucket value)."""
    rng = np.random.default_rng(0)
    n_ref, n_q, d = 200, 50, 2
    ref_coords = rng.uniform(0, 10, size=(n_ref, d))
    ref_labels = rng.uniform(0, 1, size=n_ref)
    q_coords = rng.uniform(0, 10, size=(n_q, d))

    bucket = np.zeros(n_ref, dtype=np.int64)
    q_bucket = np.zeros(n_q, dtype=np.int64)

    result = knn_within_bucket_aggregate(q_coords, ref_coords, ref_labels, q_bucket=q_bucket, ref_bucket=bucket, k=5, agg_fns=[agg])
    reference = knn_aggregate(q_coords, ref_coords, ref_labels, k=5, agg_fns=[agg])

    assert agg in result
    np.testing.assert_allclose(result[agg], reference[agg])
