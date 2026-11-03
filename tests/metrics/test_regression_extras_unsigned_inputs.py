"""MBE, explained variance and wMAPE on unsigned integer inputs must not wrap around."""

from __future__ import annotations

import numpy as np

from mlframe.metrics.regression._regression_extras import fast_explained_variance, fast_mean_bias_error, fast_wmape


def test_mean_bias_error_of_unsigned_inputs_is_negative_one_not_2e19():
    """pred 0 vs true 1 under-predicts by 1."""
    assert fast_mean_bias_error(np.array([1], np.uint8).repeat(3), np.array([0], np.uint8).repeat(3)) == -1.0


def test_wmape_of_unsigned_inputs_matches_the_float_result():
    """|0 - 1| / 1 = 1 for every element."""
    assert abs(fast_wmape(np.array([1, 2, 3], np.uint8), np.array([0, 4, 3], np.uint8)) - (1 + 2 + 0) / 6) < 1e-12


def test_explained_variance_of_unsigned_inputs_matches_float_inputs():
    """The unsigned result equals the float64 one."""
    yt = np.array([3, 5, 2, 9, 1], np.uint8)
    yp = np.array([2, 6, 2, 7, 3], np.uint8)
    assert abs(fast_explained_variance(yt, yp) - fast_explained_variance(yt.astype(float), yp.astype(float))) < 1e-12
