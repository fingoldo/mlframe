"""Regression tests for PipelineCache observability (P1).

Validates the new counters (``n_hits`` / ``n_misses``), the
``cache_size_bytes`` accessor, and ``__repr__``. Also asserts the
overhead per get/set is microsecond-scale (negligible).
"""

from __future__ import annotations

import time

import pandas as pd

from mlframe.training.strategies import PipelineCache


def test_pipeline_cache_counts_hits_and_misses():
    """Pipeline cache counts hits and misses."""
    cache = PipelineCache()
    df = pd.DataFrame({"a": [1, 2, 3]})

    # 3 sets, 5 gets (2 hits, 3 misses)
    cache.set("k_a", df, df, df)
    cache.set("k_b", df, df, df)
    cache.set("k_c", df, df, df)
    assert cache.get("k_a") is not None
    assert cache.get("k_a") is not None
    assert cache.get("k_missing_1") is None
    assert cache.get("k_missing_2") is None
    assert cache.get("k_missing_3") is None

    assert cache.n_hits == 2
    assert cache.n_misses == 3


def test_pipeline_cache_repr_renders_counters():
    """Pipeline cache repr renders counters."""
    cache = PipelineCache()
    cache.set("only_key", None, None, None)
    cache.get("only_key")
    cache.get("nope")
    text = repr(cache)
    assert "PipelineCache(" in text
    assert "keys=1" in text
    assert "hits=1" in text
    assert "misses=1" in text


def test_pipeline_cache_size_bytes_grows_when_keys_added():
    """Pipeline cache size bytes grows when keys added."""
    cache = PipelineCache()
    base = cache.cache_size_bytes()
    cache.set("k", pd.DataFrame({"x": list(range(1000))}), None, None)
    grown = cache.cache_size_bytes()
    assert grown > base


def test_pipeline_cache_observability_overhead_is_negligible():
    """Counters + verbose=False guard must add < ~1us per call."""
    cache = PipelineCache(verbose=False)
    cache.set("k", None, None, None)
    n = 50_000
    t0 = time.perf_counter()
    for _ in range(n):
        cache.get("k")
    elapsed = time.perf_counter() - t0
    # Per-call overhead well under 10us is enough headroom for slow CI boxes.
    assert (elapsed / n) < 1e-5, f"PipelineCache.get overhead too high: {elapsed / n:.3e}s/call"
