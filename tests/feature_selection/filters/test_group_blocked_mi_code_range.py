"""group_blocked_mi validates codes against n_bins before the unchecked njit histogram."""
import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory.shared import group_blocked_mi, prepare_group_segments


def _setup():
    """Two groups of 50 rows with 3-bin codes."""
    rng = np.random.default_rng(0)
    g = np.repeat([0, 1], 50)
    x = rng.integers(0, 3, 100).astype(np.int32)
    y = rng.integers(0, 3, 100).astype(np.int32)
    si, off = prepare_group_segments(g)
    return x, y, si, off


def test_group_blocked_mi_code_above_n_bins_raises():
    """A code equal to n_bins must raise, not write past the histogram."""
    x, y, si, off = _setup()
    x[5] = 3
    with pytest.raises(ValueError):
        group_blocked_mi(x, y, si, off, 3, 3, min_rows=10)


def test_group_blocked_mi_short_sort_idx_is_nan():
    """A sort_idx shorter than the codes is a misalignment: 'cannot decide'."""
    x, y, si, off = _setup()
    assert np.isnan(group_blocked_mi(x, y, si[:90], off, 3, 3, min_rows=10))


def test_group_blocked_mi_valid_input_finite():
    """Valid input still yields a finite value."""
    x, y, si, off = _setup()
    assert np.isfinite(group_blocked_mi(x, y, si, off, 3, 3, min_rows=10))
