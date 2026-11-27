"""Cluster 24 regression: ``_direct_window_agg`` writes by row position, whatever the query's index labels are (audit 2026-11-20, 24-F38 sweep)."""

import numpy as np
import pandas as pd

from mlframe.feature_engineering.multi_window_aggregate import _direct_window_agg


def _history():
    """Two entities with three and two events."""
    return pd.DataFrame({"e": [1, 1, 1, 2, 2], "t": [1.0, 2.0, 3.0, 1.0, 2.0], "v": [10.0, 20.0, 30.0, 5.0, 7.0]})


def _agg(query, rows=None):
    """Trailing-5 sum of v at each query row's t."""
    return _direct_window_agg(_history(), "e", "t", query, "t", 5.0, "v", "sum", rows=rows)


def test_non_default_index_no_longer_raises_and_matches_the_reset_frame():
    """A query indexed 10, 20 gives the same array as the same rows with a fresh RangeIndex (it raised IndexError) (24-F38)."""
    q = pd.DataFrame({"e": [1, 2], "t": [3.5, 2.5]}, index=[10, 20])
    np.testing.assert_array_equal(_agg(q), _agg(q.reset_index(drop=True)))
    np.testing.assert_array_equal(_agg(q), [60.0, 12.0])


def test_permuted_index_labels_do_not_misplace_values():
    """Index labels that are a permutation of 0..n-1 used to write each value into the wrong row (24-F38)."""
    q = pd.DataFrame({"e": [1, 2, 1], "t": [3.5, 2.5, 2.5]}, index=[2, 0, 1])
    np.testing.assert_array_equal(_agg(q), _agg(q.reset_index(drop=True)))


def test_row_mask_selects_positions_and_leaves_the_rest_nan():
    """With a boolean mask only the masked rows are computed, at their own positions, under a non-default index (24-F38)."""
    q = pd.DataFrame({"e": [1, 2, 1], "t": [3.5, 2.5, 2.5]}, index=[7, 8, 9])
    out = _agg(q, rows=np.array([False, True, False]))
    assert np.isnan(out[0]) and np.isnan(out[2]) and out[1] == 12.0
