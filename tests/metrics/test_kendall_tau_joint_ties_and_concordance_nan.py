"""Kendall tau-b counts joint ties in both denominators, and the C-index returns on NaN input (tracker 2026-11-02 wave 12, NJ12-2, NJ12-3)."""

import numpy as np
import pytest
from scipy.stats import kendalltau

from mlframe.metrics.regression._regression_corr import fast_concordance_index, fast_kendall_tau


def test_tau_b_matches_scipy_with_pairs_tied_in_both_variables():
    """x=[1,1,2], y=[1,1,3]: the joint tie belongs to both tie counts; the result must be 1.0 as scipy gives (it was 0.667)."""
    x = np.array([1.0, 1.0, 2.0])
    y = np.array([1.0, 1.0, 3.0])
    assert fast_kendall_tau(x, y) == pytest.approx(kendalltau(x, y).statistic)


def test_tau_b_matches_scipy_on_a_random_tied_sample():
    """Small integer data with many ties of every kind agrees with scipy."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 4, 120).astype(float)
    y = rng.integers(0, 4, 120).astype(float)
    assert fast_kendall_tau(x, y) == pytest.approx(kendalltau(x, y).statistic, abs=1e-12)


@pytest.mark.timeout(60)
def test_concordance_index_with_a_nan_returns_nan_instead_of_hanging():
    """A NaN in y_true stopped the tie-group scan from advancing; the call never returned."""
    assert np.isnan(fast_concordance_index(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0])))
