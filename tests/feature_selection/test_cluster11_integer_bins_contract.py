"""Cluster 11 regression: the MI binning used by every MI-based arm keeps its documented contract (audit 2026-11-20, finding 11-F20).

No test previously referenced _integer_bins. Pinned here: constant columns give zeros, few-unique columns map to their rank, many-unique
columns fill the bins evenly, and missing values take bin 0 rather than the top bin.
"""

import numpy as np

from mlframe.feature_selection._benchmarks.fs_hybrid._arms import _integer_bins


def test_constant_column_is_all_zero():
    """A single unique value (or none finite) yields all-zero codes (11-F20)."""
    assert _integer_bins(np.full(6, 3.0), 4).tolist() == [0] * 6
    assert _integer_bins(np.full(4, np.nan), 4).tolist() == [0] * 4


def test_few_unique_values_map_to_their_rank():
    """With no more unique values than bins, each value gets its rank among the uniques (11-F20)."""
    codes = _integer_bins(np.array([10.0, 20.0, 10.0, 30.0]), 5)
    assert codes.tolist() == [0, 1, 0, 2]


def test_many_unique_values_fill_bins_evenly():
    """Ranks are spread over n_bins with codes in 0..n_bins-1 (11-F20)."""
    codes = _integer_bins(np.arange(100.0), 4)
    assert codes.min() == 0 and codes.max() == 3
    counts = np.bincount(codes, minlength=4)
    assert counts.tolist() == [25, 25, 25, 25]


def test_missing_values_take_bin_zero_not_the_top():
    """NaN rows are not ranked into the top bin (the argsort-puts-NaN-last bug) (11-F20)."""
    values = np.concatenate([np.arange(20.0), [np.nan, np.nan]])
    codes = _integer_bins(values, 4)
    assert codes[-1] == 0 and codes[-2] == 0
    assert codes[:20].max() == 3
