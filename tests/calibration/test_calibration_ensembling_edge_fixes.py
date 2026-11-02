"""Calibration checks: legitimate disjoint binary splits pass, constant members do not hide redundancy, the fallback pool honours weights."""

from __future__ import annotations

import warnings

import numpy as np

from mlframe.calibration._independence_check import member_residual_correlation
from mlframe.calibration._post_train_calibrators import _check_target_identity
from mlframe.calibration.ensembling import odds_ratio_combine


def test_disjoint_binary_splits_with_equal_class_counts_are_not_an_overlap():
    """[0,1,0,1] against [1,0,1,0] have the same sorted values but are different rows."""
    _check_target_identity("m", np.array([0, 1, 0, 1]), np.array([1, 0, 1, 0]))


def test_identical_binary_targets_are_still_refused():
    """Exactly equal targets stay an overlap (negative control)."""
    import pytest

    from mlframe.calibration.post import _CalibTestOverlapError

    with pytest.raises(_CalibTestOverlapError):
        _check_target_identity("m", np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]))


def test_constant_member_does_not_dilute_the_redundancy_summary():
    """Two identical members plus a constant one: the redundancy summary is the identical pair's, not averaged down by a fake 0."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0.1, 0.9, size=500)
    members = np.column_stack([p, p, np.full(500, 0.5)])
    assert member_residual_correlation(members)["mean_abs_residual_correlation"] > 0.85


def test_correlation_fallback_keeps_a_zero_weight_member_out():
    """weights [1, 0, 1] with correlated members: the zero-weight member must not move the pooled result."""
    rng = np.random.default_rng(1)
    base = rng.uniform(0.1, 0.9, size=400)
    members = np.column_stack([base, rng.uniform(0.01, 0.99, size=400), base])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = odds_ratio_combine(members, weights=np.array([1.0, 0.0, 1.0]), check_independence=True, correlation_threshold=0.1, on_correlation_violation="fallback", prior=None)
    expected = odds_ratio_combine(members[:, [0, 2]], weights=np.array([0.5, 0.5]), prior=None)
    np.testing.assert_allclose(got, expected, atol=1e-9)
