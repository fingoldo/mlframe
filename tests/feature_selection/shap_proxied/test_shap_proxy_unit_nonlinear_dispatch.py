"""Regression guard for the ShapProxiedFS non-linear aggregator dispatch
(audit 2026-06-03: gap-04).

``build_unit_matrix`` collapses each cluster to one representative. For
non-linear combiners (median / median_z / signed_max_abs / signed_l2_sum) the
weight vector is None, and the old code hardcoded ``np.median`` for ALL of them
-- silently turning signed_max_abs and signed_l2_sum into a plain median,
diverging from the canonical _cluster_aggregate dispatch. The fix routes them
through the shared non-linear row-reducer.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster import build_unit_matrix
from mlframe.feature_selection.filters import (
    apply_cluster_aggregate_nonlinear,
    standardize_align_cluster,
)


def _two_member_cluster(seed=0, n=300):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n)
    X = np.column_stack(
        [
            base + 0.1 * rng.standard_normal(n),
            base + 0.1 * rng.standard_normal(n),
        ]
    ).astype(np.float64)
    labels = np.array([0, 0], dtype=np.int64)  # one cluster of two members
    return X, labels


def test_signed_l2_sum_not_collapsed_to_median():
    X, labels = _two_member_cluster()
    units_l2, _, _ = build_unit_matrix(X, labels, weighting="signed_l2_sum")
    units_med, _, _ = build_unit_matrix(X, labels, weighting="median")
    assert not np.allclose(units_l2[:, 0], units_med[:, 0]), (
        "signed_l2_sum aggregate is identical to median -- the non-linear dispatch is not being applied (gap-04 regression)."
    )


def test_signed_l2_sum_matches_canonical_reducer():
    X, labels = _two_member_cluster()
    units_l2, _, _ = build_unit_matrix(X, labels, weighting="signed_l2_sum")
    Z, *_ = standardize_align_cluster(X[:, [0, 1]], 0)
    expected = apply_cluster_aggregate_nonlinear(Z, "signed_l2_sum")
    assert np.allclose(units_l2[:, 0], expected), "build_unit_matrix signed_l2_sum aggregate must match the canonical _apply_method_nonlinear reducer."


def test_signed_max_abs_not_collapsed_to_median():
    X, labels = _two_member_cluster(seed=1)
    units_sma, _, _ = build_unit_matrix(X, labels, weighting="signed_max_abs")
    units_med, _, _ = build_unit_matrix(X, labels, weighting="median")
    assert not np.allclose(units_sma[:, 0], units_med[:, 0])
