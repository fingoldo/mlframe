"""Cluster 32 regression: the cached arm of the recurrent numpy-coercion benchmark uses one cache key for all three splits (audit 2026-11-20, finding 32-F18).

A per-split key never hit across splits, so the arm measured the same work as the uncached one and could not show a cache benefit.
"""

import pandas as pd

from mlframe.training._benchmarks import bench_recurrent_numpy_cache as bench
from mlframe.training.core import _phase_recurrent


def test_the_cached_arm_asks_for_the_same_key_on_every_split(monkeypatch):
    """The three coercions of one trial carry one cache key and one shared cache dict, so the second and third can hit (32-F18)."""
    calls = []

    def spy(df, cache=None, cache_key=None):
        """Record the key and the identity of the cache."""
        calls.append((cache_key, id(cache)))

    monkeypatch.setattr(_phase_recurrent, "_coerce_features_to_float32", spy)
    bench._cached(pd.DataFrame({"a": [1.0, 2.0]}), trials=1)
    assert len(calls) == 3
    assert len({key for key, _ in calls}) == 1
    assert len({cache_id for _, cache_id in calls}) == 1
