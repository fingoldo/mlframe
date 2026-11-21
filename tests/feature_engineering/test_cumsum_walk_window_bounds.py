"""The cumsum walks return a window whose elements are exactly the ones they summed (audit 2026-11-20, finding 24-F2/24-F3)."""

import numpy as np

from mlframe.feature_engineering.timeseries import (
    find_next_cumsum_left_index,
    find_next_cumsum_right_index,
)


def test_forward_walk_includes_the_crossing_element_in_the_window():
    """The element that crosses the threshold must be inside ``arr[left:end]``, not dropped after being summed."""
    arr = np.array([10.0, 1.0, 1.0, 1.0, 1.0])
    end, total = find_next_cumsum_right_index(arr, amount=3.0, left_index=0)
    assert end == 1
    assert total == 10.0
    assert arr[0:end].sum() == total


def test_forward_walk_window_sum_matches_total_on_mixed_values():
    """Window sum equals the walk total when values are mixed-sign."""
    arr = np.array([2.0, -1.0, 4.0, 0.5, 3.0, 1.0])
    end, total = find_next_cumsum_right_index(arr, amount=5.0, left_index=1)
    assert arr[1:end].sum() == total


def test_forward_walk_unreached_threshold_covers_the_whole_tail():
    """An unreachable threshold returns the exclusive end at array length with the full tail summed."""
    arr = np.array([1.0, 1.0, 1.0])
    end, total = find_next_cumsum_right_index(arr, amount=100.0, left_index=0)
    assert end == 3
    assert total == 3.0


def test_backward_walk_reaches_index_zero():
    """Index 0 was never accumulated, so a walk back to the array head undercounted."""
    arr = np.array([7.0, 0.0, 0.0])
    start, total = find_next_cumsum_left_index(arr, amount=5.0, right_index=1)
    assert start == 0
    assert total == 7.0
    assert arr[start:1].sum() == total


def test_backward_walk_window_sum_matches_total():
    """Backward window sum equals the walk total and reaches the threshold."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    start, total = find_next_cumsum_left_index(arr, amount=8.0, right_index=5)
    assert arr[start:5].sum() == total
    assert total >= 8.0
