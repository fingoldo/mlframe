"""The family gate's cached permutation blocks must reproduce the exact label-permutation stream of a fresh per-permutation draw.

A cache hit restores the generator state recorded at the original draw, so a second family scored from the cache sees the same
permutations, in the same order, as a family that draws them from scratch.
"""

import numpy as np

from mlframe.feature_selection.filters import _family_maxt_gate as gate


def _reference_null(cols, n_x, y, n_y, n_perm, seed, block=256):
    """Reference permutation-null draws, computed without the block cache."""
    rng = np.random.default_rng(seed)
    out = np.empty(int(n_perm), dtype=np.float64)
    for start in range(0, int(n_perm), block):
        m = min(block, int(n_perm) - start)
        perms = np.empty((m, y.shape[0]), dtype=np.int64)
        for i in range(m):
            perms[i] = y[rng.permutation(y.shape[0])]
        out[start : start + m] = gate.family_null_max_njit(cols, n_x, perms, int(n_y))
    return out


def test_cached_permutation_blocks_match_fresh_draws_across_families():
    """Cached permutation blocks give the same draws as fresh ones for every family."""
    rng = np.random.default_rng(3)
    n, k = 600, 12
    cols = np.ascontiguousarray(rng.integers(0, 4, size=(k, n)).astype(np.int64))
    n_x = (cols.max(axis=1) + 1).astype(np.int64)
    y = rng.integers(0, 3, size=n).astype(np.int64)
    expected = _reference_null(cols, n_x, y, 3, 600, seed=0)
    first = gate.family_null_max(cols, n_x, y, 3, 600, np.random.default_rng(0))
    second = gate.family_null_max(cols, n_x, y, 3, 600, np.random.default_rng(0))
    np.testing.assert_allclose(first, expected, rtol=0, atol=1e-12)
    np.testing.assert_allclose(second, expected, rtol=0, atol=1e-12)
    assert len(gate._perm_block_cache) > 0
