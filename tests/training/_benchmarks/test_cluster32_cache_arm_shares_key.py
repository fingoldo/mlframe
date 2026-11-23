"""Cluster 32 regression: the cached benchmark arm uses one cache key across splits so it can hit (audit 2026-11-20, finding 32-F18)."""

import inspect

from mlframe.training._benchmarks import bench_recurrent_numpy_cache as bench


def test_cached_arm_uses_one_key_for_all_splits():
    """``_cached`` keys every split identically; a per-split key never hits across splits (32-F18)."""
    src = inspect.getsource(bench._cached)
    assert 'cache_key=("shared", id(df))' in src
    assert "cache_key=(_split, id(df))" not in src
