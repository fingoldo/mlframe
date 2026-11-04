"""coverage and winkler_score return NaN when an input row is NaN instead of counting it as a miss or dropping its penalty."""
import numpy as np

from mlframe.metrics.quantile import coverage, winkler_score


def test_coverage_with_a_nan_target_is_nan():
    """coverage([nan, 1], [0, 0], [2, 2]) returned 0.5: the NaN row failed the interval test and counted as missed."""
    assert np.isnan(coverage([np.nan, 1.0], [0.0, 0.0], [2.0, 2.0]))


def test_winkler_with_a_nan_target_is_nan():
    """winkler_score([nan], [0], [1], 0.2) returned 1.0: both comparisons were False, so no penalty was added."""
    assert np.isnan(winkler_score([np.nan], [0.0], [1.0], 0.2))


def test_finite_inputs_are_unchanged():
    """No NaN, no change."""
    assert coverage([0.5, 3.0], [0.0, 0.0], [1.0, 1.0]) == 0.5
