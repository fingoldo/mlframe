"""``_stack_codes_transposed`` copies runs of single-column views that are consecutive rows of one parent as a block and equals the plain transposed stack."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _family_maxt_gate as G


def _reference(mats):
    """The batches concatenated column-wise and transposed."""
    return np.concatenate([np.asarray(m).reshape(m.shape[0], -1) for m in mats], axis=1).T


def _parent(dtype, m=40, n=500, hi=100, seed=0):
    """A (m, n) C-order code block."""
    return np.random.default_rng(seed).integers(0, hi, (m, n)).astype(dtype)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int64])
def test_consecutive_rows_equal_the_plain_stack(dtype):
    """Rows 3..14 of a parent, seen as (n, 1) views, stack to the same matrix as the generic path."""
    block = _parent(dtype)
    mats = [block[k].reshape(-1, 1) for k in range(3, 15)]
    got = G._stack_codes_transposed(mats, [1] * len(mats))
    assert np.array_equal(got, _reference(mats))
    segs = G._row_run_segments(mats, [1] * len(mats))
    assert len(segs) == 1 and segs[0][0] is not None and segs[0][0].shape == (12, block.shape[1])


def test_gaps_and_other_batches_keep_the_order():
    """A run, a gap in the rows, a strided column of a wide matrix, a two-column batch and another run: same result as the reference, runs found where they are."""
    block = _parent(np.int8)
    wide = np.random.default_rng(1).integers(0, 50, (block.shape[1], 9)).astype(np.int8)
    two = np.random.default_rng(2).integers(0, 50, (block.shape[1], 2)).astype(np.int8)
    mats = [block[k].reshape(-1, 1) for k in (0, 1, 2)] + [block[7].reshape(-1, 1)] + [wide[:, 4:5], two] + [block[k].reshape(-1, 1) for k in (20, 21)]
    widths = [1, 1, 1, 1, 1, 2, 1, 1]
    got = G._stack_codes_transposed(mats, widths)
    assert np.array_equal(got, _reference(mats))
    kinds = [(s[0] is not None) for s in G._row_run_segments(mats, widths)]
    assert kinds[0] is True and kinds[-1] is True


def test_codes_width_follows_the_largest_code():
    """A run holding a code above 127 widens the result to int16, as for the generic path."""
    block = _parent(np.int16, hi=300)
    block[5, 3] = 299
    mats = [block[k].reshape(-1, 1) for k in (4, 5, 6)]
    got = G._stack_codes_transposed(mats, [1, 1, 1])
    assert got.dtype == np.int16 and np.array_equal(got, _reference(mats))


def test_single_view_is_not_a_run():
    """One row view alone takes the generic path."""
    block = _parent(np.int8)
    mats = [block[2].reshape(-1, 1)]
    assert G._row_run_segments(mats, [1])[0][0] is None
    assert np.array_equal(G._stack_codes_transposed(mats, [1]), _reference(mats))
