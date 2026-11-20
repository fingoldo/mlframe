"""Regression tests for the meta FE recommender's column readers."""
import numpy as np

from mlframe.feature_selection.filters._meta_fe_recommender import _skew_kurt


def test_skew_kurt_is_scale_invariant_for_tiny_scales():
    """A heavy-tailed column at a 1e-7 scale reports the same skew/kurtosis as at unit scale."""
    x = np.random.default_rng(1).standard_t(3, 5000)
    big = _skew_kurt(x)
    small = _skew_kurt(x * 1e-7)
    assert small[0] == __import__("pytest").approx(big[0]) and small[1] == __import__("pytest").approx(big[1])
    assert abs(small[1]) > 0.0
