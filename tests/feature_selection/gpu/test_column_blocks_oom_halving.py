"""Column blocks join in order, and a block that runs out of device memory is retried at half the width."""

import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._device_blocks import first_block, run_column_blocks


def test_pieces_cover_all_columns_in_order():
    """Blocks of three columns over ten columns give four pieces that concatenate to 0..9."""
    pieces = run_column_blocks(cp, 10, 3, lambda lo, hi: list(range(lo, hi)))
    assert [c for p in pieces for c in p] == list(range(10))
    assert [len(p) for p in pieces] == [3, 3, 3, 1]


def test_out_of_memory_halves_the_block():
    """A work function that fails above two columns finishes with blocks of at most two."""
    seen = []

    def work(lo, hi):
        """Fails with an out-of-memory error for blocks wider than two columns."""
        seen.append(hi - lo)
        if hi - lo > 2:
            raise cp.cuda.memory.OutOfMemoryError(1, 0, 0)
        return list(range(lo, hi))

    pieces = run_column_blocks(cp, 9, 8, work)
    assert [c for p in pieces for c in p] == list(range(9))
    assert max(seen[-5:]) <= 2 and seen[0] == 8


def test_a_single_column_that_does_not_fit_raises():
    """The error surfaces (the caller's host fallback decides) when even one column fails."""
    def work(lo, hi):
        """Always out of memory."""
        raise cp.cuda.memory.OutOfMemoryError(1, 0, 0)

    with pytest.raises(cp.cuda.memory.OutOfMemoryError):
        run_column_blocks(cp, 3, 2, work)


def test_first_block_is_at_least_one_column():
    """Even a huge matrix gets a block of one column."""
    assert first_block(cp, 10**9, 50, 8, 100) == 1
