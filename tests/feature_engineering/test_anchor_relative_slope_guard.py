"""Regression: the anchor OLS slope guard must be relative, not an absolute 1e-12."""
import numpy as np

from mlframe.feature_engineering import anchor


def test_tiny_spread_anchor_positions_give_true_slope():
    """Positions spread ~1e-7 must yield the true slope (1e7) instead of a silent 0."""
    xs = np.array([0.0, 1e-7, 2e-7])
    ys = np.array([0.0, 1.0, 2.0])
    assert abs(anchor._window_ols_slope(xs, ys) - 1e7) < 1.0
