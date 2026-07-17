"""Regression tests for cross-round maxT permutation-null gain-floor caching (2026-07-09 fix).

Before this fix, ``compute_fdr_gain_floor`` -> ``pooled_permutation_null_gain_floor`` recomputed a
full (n_permutations x n_candidates x n_rows) histogram+MI pass on every ``screen_predictors()`` call,
even when the SAME raw-column candidate pool + seed recurred across a fit's 2-3 rounds (mathematically
identical output, since the underlying data for those columns never changes). ``maxt_floor_cache`` is
an optional dict the caller threads across rounds; a cache hit skips the recomputation entirely.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._screen_predictors_prescreen import compute_fdr_gain_floor


def _make_wide_pool(n=3000, p=40, seed=0):
    rng = np.random.default_rng(seed)
    factors_data = rng.integers(0, 5, size=(n, p + 1)).astype(np.int32)
    factors_nbins = np.array([5] * (p + 1), dtype=np.int32)
    x = list(range(p))  # candidate pool (wide: default screen_fdr_min_features=30)
    y = [p]  # target column index
    return factors_data, factors_nbins, x, y


def _common_kwargs(**overrides):
    base = dict(
        screen_fdr_null_permutations=25,
        screen_fdr_null_quantile=0.95,
        screen_fdr_min_features=30,
        screen_fdr_target_oversplit_ratio=1.0,
        screen_fdr_min_rows_per_joint_cell=8.0,
        cardinality_bias_correction=True,
        random_seed=7,
        verbose=0,
    )
    base.update(overrides)
    return base


def test_cache_none_disables_caching_legacy_behavior():
    """maxt_floor_cache=None (default) must behave exactly as before -- always recompute."""
    fd, fn, x, y = _make_wide_pool(seed=1)
    floor1 = compute_fdr_gain_floor(fd, fn, x, y, **_common_kwargs())
    floor2 = compute_fdr_gain_floor(fd, fn, x, y, **_common_kwargs())
    assert floor1 == pytest.approx(floor2)  # deterministic given fixed seed either way


def test_cache_hit_returns_identical_floor_to_a_fresh_computation():
    fd, fn, x, y = _make_wide_pool(seed=2)
    cache: dict = {}
    floor_cold = compute_fdr_gain_floor(fd, fn, x, y, maxt_floor_cache=cache, **_common_kwargs())
    assert len(cache) == 1, "first call with a cache dict must populate exactly one entry"

    floor_warm = compute_fdr_gain_floor(fd, fn, x, y, maxt_floor_cache=cache, **_common_kwargs())
    assert floor_warm == floor_cold  # bit-identical: served from cache, not recomputed
    assert len(cache) == 1, "second call on the same key must be a cache HIT, not a new entry"


def test_cache_miss_on_different_pool_recomputes_and_adds_entry():
    fd, fn, x, y = _make_wide_pool(seed=3)
    cache: dict = {}
    compute_fdr_gain_floor(fd, fn, x, y, maxt_floor_cache=cache, **_common_kwargs())
    assert len(cache) == 1

    x_wider = x + [len(x)]  # a different (wider) pool -- must NOT hit the same cache entry
    fd2, fn2, _, _ = _make_wide_pool(seed=3, p=len(x_wider))
    compute_fdr_gain_floor(fd2, fn2, x_wider, y, maxt_floor_cache=cache, **_common_kwargs())
    assert len(cache) == 2, "a genuinely different candidate pool must produce a distinct cache entry"


def test_cache_miss_on_different_seed_recomputes():
    fd, fn, x, y = _make_wide_pool(seed=4)
    cache: dict = {}
    compute_fdr_gain_floor(fd, fn, x, y, maxt_floor_cache=cache, **_common_kwargs(random_seed=1))
    compute_fdr_gain_floor(fd, fn, x, y, maxt_floor_cache=cache, **_common_kwargs(random_seed=2))
    assert len(cache) == 2, "different random_seed must not collide in the cache key"


def test_cache_avoids_recomputation_call_count():
    """Direct proof the cache actually skips the expensive kernel on a hit."""
    fd, fn, x, y = _make_wide_pool(seed=6)
    cache: dict = {}
    call_count = {"n": 0}

    import mlframe.feature_selection.filters._permutation_null as permnull_mod

    original = permnull_mod.pooled_permutation_null_gain_floor

    def _counting(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    permnull_mod.pooled_permutation_null_gain_floor = _counting
    try:
        compute_fdr_gain_floor(fd, fn, x, y, maxt_floor_cache=cache, **_common_kwargs())
        compute_fdr_gain_floor(fd, fn, x, y, maxt_floor_cache=cache, **_common_kwargs())
        compute_fdr_gain_floor(fd, fn, x, y, maxt_floor_cache=cache, **_common_kwargs())
    finally:
        permnull_mod.pooled_permutation_null_gain_floor = original

    assert call_count["n"] == 1, f"expected exactly 1 real computation across 3 calls with a warm cache, got {call_count['n']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
