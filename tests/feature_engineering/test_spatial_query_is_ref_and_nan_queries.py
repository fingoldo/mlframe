"""Spatial kNN features exclude the query point itself when the query pool is the reference pool, and tolerate non-finite queries (audit 2026-11-20, findings 24-F9, 24-F10, 24-F11)."""

import numpy as np

from mlframe.feature_engineering.spatial import (
    inverse_distance_weighted_aggregate,
    knn_gradient_features,
    knn_within_bucket_aggregate,
)

# Five points on a line with distinct labels; a point's own label must never feed its own feature.
_X = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
_Y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])


def test_idw_self_is_excluded_when_query_is_reference():
    """With k=1 and query == reference, row i must see its nearest OTHER point's label, not its own."""
    out = inverse_distance_weighted_aggregate(_X, _X, _Y, k=1)
    idw = out["idw"]
    # Nearest other neighbour of point 0 is point 1 (label 20); of point 4 it is point 3 (label 40).
    assert idw[0] == 20.0
    assert idw[4] == 40.0


def test_idw_separate_query_pool_is_untouched():
    """A distinct query pool keeps the plain nearest-k behaviour: a query on top of a reference point does take its label."""
    q = np.array([[1.0, 0.0]])
    out = inverse_distance_weighted_aggregate(q, _X, _Y, k=1)
    assert out["idw"][0] == 20.0


def test_within_bucket_self_is_excluded_when_query_is_reference():
    """Single bucket, query == reference, k=1: the feature must come from a different point."""
    bucket = np.zeros(len(_X), dtype=np.int64)
    out = knn_within_bucket_aggregate(_X, _X, _Y, q_bucket=bucket, ref_bucket=bucket, k=1, agg_fns=("mean",))
    assert out["mean"][0] == 20.0
    assert out["mean"][4] == 40.0


def test_gradient_with_nonfinite_query_row_stays_nan_and_others_are_finite():
    """A NaN query coordinate must not poison the batched solve for the other rows."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(size=(30, 2))
    labels = pts[:, 0] + 2.0 * pts[:, 1]
    q = pts.copy()
    q[2] = np.nan
    out = knn_gradient_features(q, pts, labels, k=5)
    assert np.isnan(out["grad_axis_0"][2])
    finite_rows = [0, 1, 3, 4, 5]
    assert np.isfinite(out["grad_axis_0"][finite_rows]).all()
    np.testing.assert_allclose(out["grad_axis_0"][finite_rows], 1.0, atol=1e-6)
