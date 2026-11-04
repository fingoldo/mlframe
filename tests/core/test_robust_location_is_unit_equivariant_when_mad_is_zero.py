"""The M-estimator location scales with the data even when more than half of the values tie at the median (MAD = 0)."""
import numpy as np

from mlframe.core.robust_location import robust_mean_mestimator


def test_estimate_does_not_depend_on_the_units():
    """With MAD = 0 the scale fell back to the constant 1.0: 0.5-sized values were barely down-weighted, 500-sized ones were rejected outright."""
    x = np.array([0.0] * 6 + [0.5] * 4)
    small = robust_mean_mestimator(x)
    large = robust_mean_mestimator(x * 1000.0) / 1000.0
    assert abs(small - large) < 1e-6, (small, large)
