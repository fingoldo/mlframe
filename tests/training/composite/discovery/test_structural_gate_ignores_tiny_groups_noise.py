"""The structural-fragility ratio is adjusted for group count: an i.i.d. base is not a per-group level however fine the grouping."""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery._yscale_holdout_gate import _between_group_variance_ratio


def test_iid_base_in_tiny_groups_has_near_zero_ratio():
    """6,000 i.i.d. values in 3,000 two-row groups: the plain between / total ratio is ~0.5-1; the adjusted one is ~0."""
    rng = np.random.default_rng(0)
    values = rng.normal(size=6000)
    inverse = np.repeat(np.arange(3000), 2)
    assert _between_group_variance_ratio(values, inverse, 3000) < 0.1


def test_group_level_base_keeps_a_high_ratio():
    """A genuine per-group level (large between-group spread, small noise) stays near 1."""
    rng = np.random.default_rng(1)
    levels = rng.uniform(0, 1000, 20)
    inverse = np.repeat(np.arange(20), 200)
    values = levels[inverse] + rng.normal(0, 5, inverse.size)
    assert _between_group_variance_ratio(values, inverse, 20) > 0.95


def test_constant_or_underdetermined_input_is_zero():
    """A constant column, or as many groups as rows, has no estimable between-group share."""
    assert _between_group_variance_ratio(np.ones(10), np.arange(10) % 3, 3) == 0.0
    assert _between_group_variance_ratio(np.arange(5.0), np.arange(5), 5) == 0.0
