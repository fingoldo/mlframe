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


def test_device_permutations_are_permutations_and_reproducible():
    """Above the row threshold the binned-agg null draws device permutations from one seed of the caller's generator: valid permutations, the same for the same seed."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters._null_max_perm_gpu import _permuted_labels

    n = 5000
    y8 = cp.asarray((np.arange(n) % 3).astype(np.uint8))

    def labels(seed):
        """The permuted labels of three device permutations drawn from ``seed``."""
        rs = cp.random.RandomState(seed)
        blocks = PermutationBlocks(lambda _i: rs.permutation(n), 3)
        return cp.asnumpy(_permuted_labels(cp, y8, blocks, 0, 3))

    a, b = labels(11), labels(11)
    np.testing.assert_array_equal(a, b)
    for row in a:
        assert np.bincount(row, minlength=3).tolist() == np.bincount(np.arange(n) % 3, minlength=3).tolist()


def test_packed_labels_are_reused_for_the_same_key_and_give_equal_maxima():
    """Two calls with the same labels key score the same packed permutations: equal null maxima, and the second call draws nothing."""
    cp = pytest.importorskip("cupy")
    from mlframe.feature_selection.filters import _null_max_perm_gpu as G

    rng = np.random.default_rng(0)
    n, k = 3000, 6
    x = cp.asarray(rng.standard_normal((k, n)).astype(np.float32))
    interior = cp.asarray(np.sort(rng.standard_normal((7, k)), axis=0))
    y = cp.asarray((rng.random(n) < 0.4).astype(np.int64))
    G._LABEL_BITS_CACHE.clear()
    draws = []

    def blocks():
        """Ten seeded host permutations, counting the draws."""
        g = np.random.default_rng(5)
        return G.PermutationBlocks(lambda _i: draws.append(1) or g.permutation(n), 10)

    a = G.null_max_mi_perms_gpu(cp, x, interior, y, blocks(), 8, 2, labels_key=("t", 1))
    first = len(draws)
    second_blocks = blocks()
    b = G.null_max_mi_perms_gpu(cp, x, interior, y, second_blocks, 8, 2, labels_key=("t", 1))
    np.testing.assert_array_equal(a, b)
    assert first == 10 and len(draws) == first and not second_blocks.drawn()
    G._LABEL_BITS_CACHE.clear()
