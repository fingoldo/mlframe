"""Audit 2026-11-20 finding 34-F15: the vectorised tie ranking and the compiled PAVA match the reference results.

The rank helper must give the same mean tie ranks as scipy's rankdata, and the PAVA kernel must match a plain isotonic fit.
"""

import numpy as np
from scipy.stats import rankdata

from mlframe.training.composite.discovery import _leakage


def test_rank_matches_scipy_average_ranks_with_ties():
    """Tied values share the mean of their rank positions, as rankdata(method='average') does."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 7, size=500).astype(np.float64)
    np.testing.assert_allclose(_leakage._rank(a), rankdata(a, method="average") - 1.0)  # the helper is 0-based


def test_rank_keeps_nan_entries_distinct():
    """A NaN does not tie with another NaN, so the NaN path still ranks every NaN on its own."""
    a = np.array([1.0, np.nan, 1.0, np.nan])
    r = _leakage._rank(a)
    assert r[0] == r[2] == 0.5
    assert r[1] != r[3]


def test_pava_is_monotone_and_least_squares():
    """The compiled PAVA output is non-decreasing and equals the block means of a monotone fit of a decreasing sequence."""
    y = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
    out = _leakage._isotonic_pava(y)
    assert np.all(np.diff(out) >= -1e-12)
    np.testing.assert_allclose(out, [2.0, 2.0, 2.0, 4.5, 4.5])
