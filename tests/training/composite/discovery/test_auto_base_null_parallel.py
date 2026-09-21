"""The auto-base permutation null must be computed in parallel without changing a single null MI.

The null block-shuffled each feature's bin codes ``auto_base_null_perms`` times and took the MI against y's codes, one
Python iteration per (column, permutation), serially. The permutations are now drawn up front in the order the loop
consumed the generator, and one parallel kernel scores them, calling the same gather and MI kernels.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery._collinear_numba import block_shuffle_gather
from mlframe.training.composite.discovery._null_mi_numba import draw_null_permutations, null_mis_binned, shuffle_by
from mlframe.training.composite.discovery.screening import _mi_from_binned_pair


def _loop_reference(col_codes, y_codes, rng, n_perms, block_len, nbins):
    """The original per-permutation loop: one live draw and one MI call per permutation."""
    out = np.empty(n_perms)
    for p in range(n_perms):
        if block_len <= 1:
            shuffled = col_codes.copy()
            rng.shuffle(shuffled)
        else:
            n_blocks = (col_codes.size + block_len - 1) // block_len
            shuffled = block_shuffle_gather(col_codes, rng.permutation(n_blocks), block_len)
        out[p] = _mi_from_binned_pair(shuffled, y_codes, nbins=nbins)
    return out


def _codes(n: int, nbins: int, seed: int):
    """A correlated pair of binned columns."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 0.6 * x + rng.normal(size=n)
    edges = np.linspace(0, 1, nbins + 1)[1:-1]
    xc = np.clip(np.searchsorted(np.quantile(x, edges), x, side="right"), 0, nbins - 1).astype(np.int64)
    yc = np.clip(np.searchsorted(np.quantile(y, edges), y, side="right"), 0, nbins - 1).astype(np.int64)
    return xc, yc


@pytest.mark.parametrize("block_len", [1, 7, 50])
def test_every_null_mi_is_identical_to_the_loop(block_len):
    """Same generator state in, same null MIs out - for element shuffles and for block shuffles."""
    xc, yc = _codes(5003, 16, 0)
    ref = _loop_reference(xc, yc, np.random.default_rng(123), 20, block_len, 16)
    perms = draw_null_permutations(np.random.default_rng(123), xc.size, 20, block_len)
    got = null_mis_binned(xc, yc, perms, block_len, 16)
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("block_len", [1, 7])
def test_the_generator_is_left_where_the_loop_left_it(block_len):
    """The next column's draws must be unchanged, so the generator must advance exactly as the loop advanced it."""
    xc, yc = _codes(2001, 8, 1)
    a, b = np.random.default_rng(9), np.random.default_rng(9)
    _loop_reference(xc, yc, a, 20, block_len, 8)
    draw_null_permutations(b, xc.size, 20, block_len)
    assert a.integers(0, 2**62) == b.integers(0, 2**62)


def test_the_value_fallback_path_shuffles_like_the_loop():
    """Columns scored on values rather than codes get the identical shuffled array from the pre-drawn permutation."""
    rng_loop, rng_pre = np.random.default_rng(5), np.random.default_rng(5)
    col = np.random.default_rng(6).normal(size=997)
    for block_len in (1, 13):
        if block_len <= 1:
            expected = col.copy()
            rng_loop.shuffle(expected)
        else:
            expected = block_shuffle_gather(col, rng_loop.permutation((col.size + block_len - 1) // block_len), block_len)
        perm = draw_null_permutations(rng_pre, col.size, 1, block_len)[0]
        np.testing.assert_array_equal(shuffle_by(col, perm, block_len), expected)
