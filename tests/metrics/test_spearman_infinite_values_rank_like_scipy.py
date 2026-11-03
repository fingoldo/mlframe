"""Infinite values are ordinary extremes for a rank correlation: the numba paths agree with scipy instead of returning NaN."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from mlframe.metrics.rank_correlation import spearmanr_scalar_dispatch


def test_scalar_spearman_with_an_infinite_value_matches_scipy():
    """x = [1, 2, inf] against y = [1, 2, 3] is a perfect monotone relation: rho = 1.0 (the numba path gave NaN)."""
    x = np.array([1.0, 2.0, np.inf, 0.5, 4.0])
    y = np.array([1.0, 2.0, 3.0, 0.1, 2.5])
    got = spearmanr_scalar_dispatch(x, y)
    assert abs(got - spearmanr(x, y)[0]) < 1e-12


def test_nan_still_gives_nan():
    """A NaN in either input keeps returning NaN (negative control)."""
    x = np.array([1.0, np.nan, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isnan(spearmanr_scalar_dispatch(x, y))
