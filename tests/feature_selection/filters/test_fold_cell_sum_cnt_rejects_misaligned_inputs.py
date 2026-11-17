"""fold_cell_sum_cnt feeds an unchecked njit kernel: misaligned lengths and out-of-range codes must raise, not read out of bounds."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._fold_cell_stats import fold_cell_sum_cnt


def test_fold_cell_sum_cnt_rejects_short_target():
    """A target shorter than the codes is rejected."""
    with pytest.raises(ValueError):
        fold_cell_sum_cnt(np.zeros(5, np.int64), np.zeros(3), np.zeros(5, np.int64), -1, 2)


def test_fold_cell_sum_cnt_rejects_short_valid_mask():
    """A valid mask shorter than the codes is rejected."""
    with pytest.raises(ValueError):
        fold_cell_sum_cnt(np.zeros(5, np.int64), np.zeros(5), np.zeros(5, np.int64), -1, 2, valid=np.ones(2, bool))


def test_fold_cell_sum_cnt_rejects_code_outside_cells():
    """A code at or beyond n_cells (or negative) is rejected."""
    with pytest.raises(ValueError):
        fold_cell_sum_cnt(np.array([0, 2]), np.zeros(2), np.zeros(2, np.int64), -1, 2)
    with pytest.raises(ValueError):
        fold_cell_sum_cnt(np.array([0, -1]), np.zeros(2), np.zeros(2, np.int64), -1, 2)


def test_fold_cell_sum_cnt_valid_input_unchanged():
    """Well-formed input still returns the per-cell sums and counts."""
    s, c = fold_cell_sum_cnt(np.array([0, 1, 1]), np.array([1.0, 2.0, 3.0]), np.array([0, 0, 1]), 1, 2)
    assert s.tolist() == [1.0, 2.0] and c.tolist() == [1.0, 1.0]
