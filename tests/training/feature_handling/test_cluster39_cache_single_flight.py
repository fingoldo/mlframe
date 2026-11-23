"""Cluster 39 regression: concurrent misses on one key compute it once (audit 2026-11-20, finding 39-F29)."""

import threading
import time

from mlframe.training.feature_handling.cache import FeatureCache
from mlframe.training.feature_handling.config import CacheConfig
from mlframe.training.feature_handling.fingerprint import InMemoryKey


def _key(column="a"):
    """A fixed in-memory key for the test column."""
    return InMemoryKey(session_id="s", df_token=1, train_idx_token=2, column=column, params_canonical_hash="p", provider_signature="v")


def test_concurrent_misses_on_one_key_compute_once():
    """Four threads missing the same key run the compute function once and all receive its value (39-F29)."""
    cache = FeatureCache(CacheConfig(persistence="off"))
    calls = []
    barrier = threading.Barrier(4)
    results = []

    def _compute():
        """Slow compute that records each invocation."""
        calls.append(1)
        time.sleep(0.2)
        return "value"

    def _worker():
        """Request the key after all threads are ready."""
        barrier.wait()
        results.append(cache.get_or_compute(_key(), _compute))

    threads = [threading.Thread(target=_worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert len(calls) == 1
    assert results == ["value"] * 4


def test_distinct_keys_are_not_serialised_into_one_computation():
    """Control: different keys each compute their own value (39-F29)."""
    cache = FeatureCache(CacheConfig(persistence="off"))
    assert cache.get_or_compute(_key("a"), lambda: 1) == 1
    assert cache.get_or_compute(_key("b"), lambda: 2) == 2
