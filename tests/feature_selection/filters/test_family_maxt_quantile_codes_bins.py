"""Quantile bin codes of a continuous column: the bins are the quantile edges, the top value is clipped into the last bin, and the count is the number of bins used.

For ``arange(100)`` with ``n_bins=4`` the edges are 0, 24.75, 49.5, 74.25, 99, so the bins hold 25 values each and the top value 99 is clipped into bin 3.
"""

from __future__ import annotations

import numpy as np

import mlframe.feature_selection.filters._family_maxt_gate as gate


def test_continuous_column_splits_into_equal_quantile_bins():
    """Each of the four quantile bins holds 25 of the 100 values, and the count of codes is four."""
    codes, n_codes = gate._quantile_codes(np.arange(100, dtype=np.float64), 4)
    assert n_codes == 4
    np.testing.assert_array_equal(np.bincount(codes, minlength=4), [25, 25, 25, 25])


def test_bin_boundaries_fall_where_the_quantile_edges_put_them():
    """A value just below an edge stays in the lower bin and a value at or above it moves up; the largest value is clipped into the last bin."""
    codes, _ = gate._quantile_codes(np.arange(100, dtype=np.float64), 4)
    assert codes[0] == 0 and codes[24] == 0
    assert codes[25] == 1 and codes[49] == 1
    assert codes[50] == 2
    assert codes[74] == 2
    assert codes[75] == 3
    assert codes[99] == 3


def test_codes_are_non_decreasing_in_the_value():
    """The bins are ordered: a larger value never receives a smaller code."""
    rng = np.random.default_rng(3)
    v = np.sort(rng.normal(size=500))
    codes, n_codes = gate._quantile_codes(v, 6)
    assert np.all(np.diff(codes) >= 0)
    assert codes.min() == 0 and codes.max() == n_codes - 1


def test_a_few_distinct_values_get_exact_codes_not_quantile_bins():
    """A column with at most ``n_bins`` distinct values keeps one code per value, so a binary column is not collapsed to one bin."""
    v = np.array([3.0, 7.0, 3.0, 7.0, 7.0])
    codes, n_codes = gate._quantile_codes(v, 4)
    assert n_codes == 2
    np.testing.assert_array_equal(codes, [0, 1, 0, 1, 1])
