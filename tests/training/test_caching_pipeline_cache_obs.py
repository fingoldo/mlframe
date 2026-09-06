"""Regression tests for PipelineCache observability (P1).

Validates the new counters (``n_hits`` / ``n_misses``), the
``cache_size_bytes`` accessor, and ``__repr__``. Also asserts the
overhead per get/set is microsecond-scale (negligible).
"""

from __future__ import annotations


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


def test_pipeline_cache_observability_costs_nothing_when_quiet(monkeypatch):
    """The counters must still move with ``verbose=False``, and the log formatting must not run.

    That is what the per-call budget stood in for. 10us per call is a claim about the host: the nightly
    coverage job traces every line and a contended worker multiplies per-call Python overhead severalfold,
    on entirely correct code -- while an unguarded ``logger.info`` could still slip under the budget on a
    fast box with logging disabled, which is the regression it names.
    """
    from mlframe.training.strategies import pipeline_cache as pc

    cache = PipelineCache(verbose=False)
    cache.set("k", None, None, None)

    logged = []
    monkeypatch.setattr(pc.logger, "info", lambda *a, **k: logged.append(1))

    hits_before, misses_before = cache.n_hits, cache.n_misses
    for _ in range(50):
        cache.get("k")
    for _ in range(10):
        cache.get("absent")

    assert not logged, f"a quiet cache emitted {len(logged)} log lines; the verbose guard is not in front of the formatting"
    assert cache.n_hits - hits_before == 50, f"the hit counter moved by {cache.n_hits - hits_before}, not 50"
    assert cache.n_misses - misses_before == 10, f"the miss counter moved by {cache.n_misses - misses_before}, not 10"

    # And the guard must be a guard, not a removal: a verbose cache still reports.
    loud = PipelineCache(verbose=True)
    loud.set("k", None, None, None)
    loud.get("k")
    assert logged, "a verbose cache emitted nothing; the observability this test is named for is gone"
