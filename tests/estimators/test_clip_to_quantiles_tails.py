"""clip_to_quantiles compresses the tails inward, tolerates ties at the extremes, and keeps order for negative targets."""

from __future__ import annotations

import numpy as np

from mlframe.estimators.custom import clip_to_quantiles, clip_to_quantiles_winsor_quantile


def test_winsor_linear_pulls_the_maximum_in_instead_of_doubling_it():
    """Max 100 on 1..100 at q=0.01 lands just above the 99th percentile (~99), never near 200."""
    arr = np.arange(1, 101, dtype=float)
    out = clip_to_quantiles(arr, quantile=0.01, method="winsor_linear")
    assert out.max() < 106.0


def test_binary_target_does_not_raise():
    """Ties at both extremes leave nothing beyond either quantile; the data comes back unchanged."""
    arr = np.array([0.0] * 50 + [1.0] * 50)
    np.testing.assert_array_equal(clip_to_quantiles_winsor_quantile(arr), arr)


def test_negative_target_keeps_the_order_of_values_beyond_the_quantile():
    """On an all-negative target the values above the upper quantile stay above it and keep their rank order."""
    arr = -np.arange(1, 101, dtype=float)
    out = clip_to_quantiles(arr, quantile=0.05, method="winsor_quantile")
    order_in, order_out = np.argsort(arr, kind="stable"), np.argsort(out, kind="stable")
    np.testing.assert_array_equal(order_in, order_out)
