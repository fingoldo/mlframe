"""The columns of a bootstrap rank-MI call are split into tiles per thread; the tile width must not change a single value."""

import numpy as np

from mlframe.feature_selection.filters import _bootstrap_rank_mi_njit as B


def _inputs(n=3000, k=11, seed=1):
    """A float32 matrix with a distinct, a rounded (tied) and a constant-ish column, a binary target and 4 resamples."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(size=(n, k)).astype(np.float32)
    arr[:, 3] = np.round(arr[:, 3], 1)
    arr[:, 5] = np.round(arr[:, 5], 0)
    y = (arr[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(np.int64)
    reps = [np.sort(rng.integers(0, n, int(0.8 * n))).astype(np.int64) for _ in range(4)]
    return arr, y, reps


def test_tile_width_is_at_most_16_and_covers_every_thread():
    """Few columns are spread one per thread; many columns keep tiles of 16."""
    import numba

    threads = numba.get_num_threads()
    assert B._tile_for(1) == 1
    assert B._tile_for(16 * threads) == 16
    assert B._tile_for(2 * threads) == 2
    assert 1 <= B._tile_for(16) <= 16


def test_columns_scored_together_equal_columns_scored_alone():
    """One call over all columns equals the columns scored one at a time (a tile of one)."""
    arr, y, reps = _inputs()
    together, ok = B.bootstrap_rank_mi(arr, y, reps, 10, try_tied_device=False)
    alone = [B.bootstrap_rank_mi(np.ascontiguousarray(arr[:, [j]]), y, reps, 10, try_tied_device=False) for j in range(arr.shape[1])]
    assert np.array_equal(together, np.hstack([a[0] for a in alone]))
    assert ok.tolist() == [bool(a[1][0]) for a in alone]
