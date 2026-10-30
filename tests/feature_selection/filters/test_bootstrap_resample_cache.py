"""The seeded bootstrap resamples and the tables built from them are kept for the next call with the same seed, sizes and labels; the scores do not change."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import _bootstrap_rank_mi_njit as R
from mlframe.feature_selection.filters import _orthogonal_bootstrap_mi_fe as O


def _data(n=3000, k=5, seed=0):
    """A float32 frame with a continuous, a tied and a signal column and a binary label."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, k)).astype(np.float32)
    x[:, 1] = np.round(x[:, 1], 1)
    y = (x[:, 0] + 0.5 * rng.standard_normal(n) > 0).astype(np.int64)
    return x, y


def test_the_same_seed_sizes_and_labels_return_the_same_resample_list():
    """A repeated request is served from the cache (same object); another seed, size or label vector draws again."""
    _, y = _data()
    a = O.seeded_bootstrap_rows(0, y, len(y), 2400, 6)
    assert O.seeded_bootstrap_rows(0, y, len(y), 2400, 6) is a
    assert O.seeded_bootstrap_rows(1, y, len(y), 2400, 6) is not a
    b = O.seeded_bootstrap_rows(0, y, len(y), 2400, 6)
    rng = np.random.default_rng(0)
    for got in b:
        assert np.array_equal(got, O._bootstrap_rows(rng, y, len(y), 2400))
    assert O.seeded_bootstrap_rows(0, 1 - y, len(y), 2400, 6) is not b


def test_rank_mi_with_cached_tables_equals_a_fresh_build():
    """Scoring two column blocks against the same resamples (the second reads the cached tables) gives the same MI as scoring each with freshly built tables."""
    x, y = _data()
    reps = O.seeded_bootstrap_rows(0, y, len(y), 2400, 5)
    R._PREP_CACHE.clear()
    first, _ = R.bootstrap_rank_mi(x[:, :3], y, reps, 10)
    assert R._PREP_CACHE.get("latest") is not None
    second, _ = R.bootstrap_rank_mi(x[:, 2:], y, reps, 10)
    R._PREP_CACHE.clear()
    fresh_second, _ = R.bootstrap_rank_mi(x[:, 2:], y, reps, 10)
    R._PREP_CACHE.clear()
    fresh_first, _ = R.bootstrap_rank_mi(x[:, :3], y, reps, 10)
    assert np.array_equal(first, fresh_first) and np.array_equal(second, fresh_second)
