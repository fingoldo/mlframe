"""The family gate's cached permutation blocks must reproduce the exact label-permutation stream of a fresh draw.

A cache hit restores the generator state recorded at the original draw, so a second family scored from the cache sees the same
permutations, in the same order, as a family that draws them from scratch.
"""

import numpy as np

from mlframe.feature_selection.filters import _family_maxt_gate as gate


def _fresh_null(cols, n_x, y, n_y, n_perm, seed):
    """The null drawn with an empty block cache (every permutation block is generated, none replayed)."""
    gate._perm_block_cache.clear()
    gate._perm_block_cache_bytes = 0
    return gate.family_null_max(cols, n_x, y, n_y, n_perm, np.random.default_rng(seed))


def test_cached_permutation_blocks_match_fresh_draws_across_families():
    """Cached permutation blocks give the same draws as fresh ones for every family."""
    rng = np.random.default_rng(3)
    n, k = 600, 12
    cols = np.ascontiguousarray(rng.integers(0, 4, size=(k, n)).astype(np.int64))
    n_x = (cols.max(axis=1) + 1).astype(np.int64)
    y = rng.integers(0, 3, size=n).astype(np.int64)
    expected = _fresh_null(cols, n_x, y, 3, 600, seed=0)
    gate._perm_block_cache.clear()
    gate._perm_block_cache_bytes = 0
    first = gate.family_null_max(cols, n_x, y, 3, 600, np.random.default_rng(0))
    second = gate.family_null_max(cols, n_x, y, 3, 600, np.random.default_rng(0))
    np.testing.assert_allclose(first, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(second, expected, rtol=0, atol=1e-12)
    assert len(gate._perm_block_cache) > 0
