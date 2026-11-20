"""Missing group ids that land on both sides of a split are reported as leakage (audit 2026-11-20, finding 23-F3)."""
import numpy as np
import pytest

from mlframe.evaluation.group_leakage_guard import assert_no_group_leakage


def test_nan_group_on_both_sides_is_a_leak():
    """Two NaN group ids in train and test: the same missing group leaks; the guard must raise (it passed silently before)."""
    groups = np.array(["a", "b", np.nan, np.nan, "c", "d"], dtype=object)
    splits = [(np.array([0, 2, 4]), np.array([1, 3, 5]))]
    with pytest.raises(ValueError):
        assert_no_group_leakage(groups, splits)
