"""Audit 2026-11-20 finding 34-F5: negative train/test positions are counted from the end before the leakage check.

Before the fix only ``train_idx.max()`` was bounds-checked and the overlap test compared raw integers, so ``-1`` in train and
``n-1`` in test passed as disjoint and the last row was fitted on while also being held out.
"""

import numpy as np
import pytest

from mlframe.training.composite.discovery._fit import _fit_normalise_indices


def test_negative_train_index_overlapping_test_is_leakage():
    """train ends with -1 (row 4 of 5) and test holds row 4: the overlap is refused."""
    df = np.zeros((5, 1))
    with pytest.raises(ValueError, match="overlaps test_idx"):
        _fit_normalise_indices(df, np.array([0, 1, -1]), None, np.array([4]))


def test_negative_positions_are_normalised_to_row_positions():
    """-5 on a five-row frame is row 0, and the returned train positions are non-negative."""
    df = np.zeros((5, 1))
    train, _val, _test = _fit_normalise_indices(df, np.array([-5, 2]), None, None)
    assert train.tolist() == [0, 2]


def test_out_of_range_negative_position_is_refused():
    """-6 on a five-row frame is out of bounds and raises."""
    df = np.zeros((5, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        _fit_normalise_indices(df, np.array([-6]), None, None)
