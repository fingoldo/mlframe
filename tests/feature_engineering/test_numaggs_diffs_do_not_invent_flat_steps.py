"""Relative changes from a zero base must not be invented as flat steps."""

import numpy as np

from mlframe.feature_engineering.numerical import _defined_relative_changes, get_numaggs_names, numaggs_over_matrix_rows


def test_a_step_from_zero_is_left_out_not_zeroed():
    """x/0 and 0/0 used to become 0.0, so [0, 5, 5] looked like the flat [5, 5, 5]."""
    np.testing.assert_allclose(_defined_relative_changes(np.array([5.0, 5.0]), np.array([0.0, 5.0])), [0.0])
    np.testing.assert_allclose(_defined_relative_changes(np.array([0.0]), np.array([0.0])), [])


def test_a_row_starting_at_zero_is_no_longer_identical_to_a_flat_row():
    names = get_numaggs_names(return_float32=True)
    rows = np.array([[0.0, 5.0, 10.0, 10.0], [5.0, 5.0, 5.0, 5.0]])
    out = numaggs_over_matrix_rows(rows, {}, use_diffs=True)
    assert out.shape == (2, len(names))
    assert not np.array_equal(out[0], out[1], equal_nan=True), "a jump followed by a rise must not read as a flat series"
