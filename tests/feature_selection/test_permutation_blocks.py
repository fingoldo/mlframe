"""Permutations handed out block by block equal the eager stack and leave the generator where the stack left it."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._null_max_perm_gpu import PermutationBlocks


def test_blocks_equal_the_eager_stack_and_advance_the_generator_alike():
    """Blocks of 4 out of 10 permutations of 1000 rows equal the stack, and the generator's next draw agrees afterwards."""
    eager_rng, lazy_rng = np.random.default_rng(7), np.random.default_rng(7)
    eager = np.stack([eager_rng.permutation(1000) for _ in range(10)])
    blocks = PermutationBlocks(lambda _i: lazy_rng.permutation(1000), 10)
    got = np.concatenate([blocks[p0 : p0 + 4] for p0 in range(0, len(blocks), 4)])
    np.testing.assert_array_equal(got, eager)
    assert got.dtype == np.int32
    assert eager_rng.integers(1 << 30) == lazy_rng.integers(1 << 30)


def test_drain_finishes_the_draws_after_an_early_stop():
    """Stopping after one block and draining leaves the generator where drawing all ten leaves it."""
    a, b = np.random.default_rng(3), np.random.default_rng(3)
    for _ in range(10):
        a.permutation(50)
    blocks = PermutationBlocks(lambda _i: b.permutation(50), 10)
    blocks[0:3]
    blocks.drain()
    assert a.integers(1 << 30) == b.integers(1 << 30)


def test_out_of_order_read_is_refused():
    """A block that does not start where the previous one ended raises instead of silently drawing other permutations."""
    blocks = PermutationBlocks(lambda i: np.arange(5) + i, 6)
    blocks[0:2]
    with pytest.raises(IndexError):
        blocks[4:6]
