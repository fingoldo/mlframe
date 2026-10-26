"""The parallel permutation block is a function of the generator state, not of the thread count, and every row is a permutation."""
import numba
import numpy as np

from mlframe.feature_selection.filters import _family_maxt_gate as gate


def _block(seed: int, y: np.ndarray, m: int) -> np.ndarray:
    """One block drawn from an empty cache."""
    gate._perm_block_cache.clear()
    gate._perm_block_cache_bytes = 0
    return np.array(gate._permutation_block(np.random.default_rng(seed), y, m))


def test_rows_are_permutations_and_thread_count_is_irrelevant():
    """Every row holds each index once, and one thread gives the same block as many."""
    y = np.random.default_rng(0).integers(0, 3, 700)
    many = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        one = _block(4, y, 40)
        numba.set_num_threads(many)
        par = _block(4, y, 40)
    finally:
        numba.set_num_threads(many)
    np.testing.assert_array_equal(one, par)
    assert all(np.array_equal(np.sort(row), np.arange(700)) for row in par)
    assert not np.array_equal(par[0], par[1])


def test_the_generator_advances_by_one_draw_per_block():
    """Two consecutive blocks differ, and a replay from the same state reproduces both."""
    y = np.arange(500) % 2
    gate._perm_block_cache.clear()
    gate._perm_block_cache_bytes = 0
    rng = np.random.default_rng(9)
    first = np.array(gate._permutation_block(rng, y, 8))
    second = np.array(gate._permutation_block(rng, y, 8))
    assert not np.array_equal(first, second)
    gate._perm_block_cache.clear()
    gate._perm_block_cache_bytes = 0
    rng = np.random.default_rng(9)
    np.testing.assert_array_equal(first, np.array(gate._permutation_block(rng, y, 8)))
    np.testing.assert_array_equal(second, np.array(gate._permutation_block(rng, y, 8)))
