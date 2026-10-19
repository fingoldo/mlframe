"""Stacking the batches of a family into the (K_total, n) code matrix equals the plain transposed stack, and adjacent column views of one array are merged into one block only when that is safe."""

import numpy as np
import pytest

from mlframe.feature_selection.filters import _family_maxt_gate as fg


def _reference(mats):
    """The transposed horizontal stack of the batches, in the narrowest integer width that holds the codes."""
    full = np.hstack([np.asarray(m) for m in mats])
    top = int(full.max()) if full.size else 0
    dt = np.int8 if top <= 127 else (np.int16 if top <= 32767 else np.int64)
    return np.ascontiguousarray(full.T.astype(dt))


@pytest.fixture
def base():
    """A row-major code matrix whose columns the batches are cut from."""
    return np.random.default_rng(0).integers(0, 10, (1500, 90)).astype(np.int8)


def _blocks(mats):
    """The merge decision for ``mats``: ``(number of blocks, number of merged blocks)``."""
    blocks = fg._column_runs(mats, [m.shape[1] for m in mats])
    return len(blocks), sum(1 for _, merged in blocks if merged)


def test_adjacent_single_column_views_merge_into_one_block_and_stack_equal(base):
    """The pair family's shape (one single-column view per configuration, in order) becomes one block and equals the plain stack."""
    mats = [base[:, j : j + 1] for j in range(60)]
    assert _blocks(mats) == (1, 1)
    np.testing.assert_array_equal(fg._stack_codes_transposed(mats, [1] * 60), _reference(mats))


def test_adjacent_wider_blocks_merge(base):
    """Adjacent column blocks of the same width merge as well."""
    mats = [base[:, j : j + 6] for j in range(0, 60, 6)]
    assert _blocks(mats) == (1, 1)
    np.testing.assert_array_equal(fg._stack_codes_transposed(mats, [6] * 10), _reference(mats))


@pytest.mark.parametrize("name", ["shuffled", "gap", "reversed"])
def test_non_adjacent_batches_stay_separate_and_stack_equal(base, name):
    """Columns out of order, with a gap, or reversed are not merged across the break, and the stack is still the plain one."""
    idx = {"shuffled": list(np.random.default_rng(1).permutation(40)), "gap": list(range(0, 80, 2)), "reversed": list(range(39, -1, -1))}[name]
    mats = [base[:, j : j + 1] for j in idx]
    n_blocks, _ = _blocks(mats)
    assert n_blocks > 1
    np.testing.assert_array_equal(fg._stack_codes_transposed(mats, [1] * len(idx)), _reference(mats))


def test_views_with_different_strides_or_dtype_are_not_merged(base):
    """A view with a step along the columns, or one of another dtype, ends a run."""
    stepped = base[:, ::2]
    mats = [stepped[:, 0:1], stepped[:, 1:2]]
    assert _blocks(mats)[0] == 2
    mixed = [base[:, 0:1], base[:, 1:2].astype(np.int16), base[:, 2:3]]
    assert _blocks(mixed)[0] == 3
    np.testing.assert_array_equal(fg._stack_codes_transposed(mixed, [1, 1, 1]), _reference(mixed))


def test_run_resumes_after_a_break(base):
    """Two adjacent runs separated by an out-of-place batch become two merged blocks plus the stray one."""
    mats = [base[:, 0:1], base[:, 1:2], base[:, 50:51], base[:, 10:11], base[:, 11:12]]
    assert _blocks(mats) == (3, 2)
    np.testing.assert_array_equal(fg._stack_codes_transposed(mats, [1] * 5), _reference(mats))


def test_wide_codes_widen_the_result_type_and_the_base_is_untouched(base):
    """Codes above 127 widen the stack to int16; stacking never writes to the array the views are cut from."""
    wide = base.astype(np.int16)
    wide[3, 5] = 300
    keep = wide.copy()
    mats = [wide[:, j : j + 1] for j in range(20)]
    got = fg._stack_codes_transposed(mats, [1] * 20)
    assert got.dtype == np.int16
    np.testing.assert_array_equal(got, _reference(mats))
    np.testing.assert_array_equal(wide, keep)


def test_stack_of_independent_arrays_matches(base):
    """Batches that are independent arrays (no shared memory) are stacked as before."""
    mats = [np.ascontiguousarray(base[:, j : j + 3]) for j in range(0, 30, 3)]
    assert _blocks(mats)[1] == 0
    np.testing.assert_array_equal(fg._stack_codes_transposed(mats, [3] * 10), _reference(mats))


def test_single_column_views_with_an_arbitrary_column_stride_merge(base):
    """The pair family's views report the row stride for their length-1 axis; they still merge when their memory is adjacent."""
    n = base.shape[0]
    row = base.strides[0]
    mats = [np.lib.stride_tricks.as_strided(base[:, j:], shape=(n, 1), strides=(row, row)) for j in range(50)]
    assert mats[0].strides == (row, row)
    assert _blocks(mats) == (1, 1)
    np.testing.assert_array_equal(fg._stack_codes_transposed(mats, [1] * 50), _reference(mats))
