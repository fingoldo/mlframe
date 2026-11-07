"""Scoring the resamples a block at a time (when their tables would exceed the byte budget) gives the same MI as scoring them all at once."""

import numpy as np

from mlframe.feature_selection.filters import _bootstrap_rank_mi_njit as B


def test_blocks_of_resamples_give_the_same_mi(monkeypatch):
    """Seven resamples in blocks of two equal the single call, rows in resample order, ``ok`` unchanged."""
    rng = np.random.default_rng(2)
    n = 5000
    X = rng.normal(size=(n, 5)).astype(np.float32)
    X[:, 1] = np.round(X[:, 1], 1)
    y = (rng.random(n) < 0.4).astype(np.int64)
    reps = [rng.integers(0, n, size=4000) for _ in range(7)]
    whole, ok_whole = B.bootstrap_rank_mi(X, y, reps, 10)
    monkeypatch.setattr(B, "_RESAMPLE_TABLE_BYTES", 2 * (8 * n + 10 * 4000))
    blocked, ok_blocked = B.bootstrap_rank_mi(X, y, reps, 10)
    np.testing.assert_array_equal(blocked, whole)
    assert (ok_blocked == ok_whole).all() and blocked.shape == (7, 5)
