"""Sensor + biz_value tests for :mod:`mlframe.training.suite_artefact_cache`.

Coverage:
* ``SuiteKeyBuilder.build`` is deterministic on identical inputs.
* Every contributing slot (df_fp, config_canonical, mlframe_models, lib_versions, random_seed, extra) flips the digest -- a regression that drops a slot from the fold would let two semantically-different artefacts share a cache slot.
* ``SuiteArtefactCache.put`` -> ``.get`` round-trips a Python object.
* Cache hit on second invocation of a ``@cache_artefact``-decorated function is materially faster than the first (the trick is doing meaningful work for the bench to register a hit -> miss delta).
* Cache miss when the cache key changes (config / args / models slot diff).
* Size gate: an artefact whose ``size_estimate`` exceeds the cache ``bytes_limit`` is REFUSED (``put`` returns False, the file is NOT written).
* LRU eviction: writing N+1 entries to an N-entry cap evicts the oldest entry.
* Bytes-budget eviction: writing entries that collectively exceed ``bytes_limit`` evicts oldest entries until under budget.
* Sentinel: a cached ``None`` value is distinguishable from a cache miss.
"""
from __future__ import annotations

import os
import time

import pytest

from mlframe.training.suite_artefact_cache import (
    DEFAULT_BYTES_LIMIT,
    SuiteArtefactCache,
    SuiteKeyBuilder,
    cache_artefact,
    get_default_cache,
    set_default_cache,
)


# --------------------------------------------------------------------------
# SuiteKeyBuilder
# --------------------------------------------------------------------------


def test_key_builder_deterministic_on_identical_inputs():
    k1 = SuiteKeyBuilder.build(
        df_fp="deadbeef" * 4,
        config_canonical={"a": 1, "b": [2, 3]},
        mlframe_models=["cb", "lgb"],
        lib_versions={"sklearn": "1.6.0", "polars": "1.10.0"},
        random_seed=42,
    )
    k2 = SuiteKeyBuilder.build(
        df_fp="deadbeef" * 4,
        config_canonical={"a": 1, "b": [2, 3]},
        mlframe_models=["cb", "lgb"],
        lib_versions={"sklearn": "1.6.0", "polars": "1.10.0"},
        random_seed=42,
    )
    assert k1 == k2
    assert len(k1) == 32  # blake2b 16-byte hex


def test_key_builder_invariant_to_model_order():
    # frozenset-sort of models means ["cb", "lgb"] == ["lgb", "cb"] for cache purposes.
    k1 = SuiteKeyBuilder.build(df_fp="x", config_canonical={}, mlframe_models=["cb", "lgb"])
    k2 = SuiteKeyBuilder.build(df_fp="x", config_canonical={}, mlframe_models=["lgb", "cb"])
    assert k1 == k2


def test_key_builder_invariant_to_lib_versions_order():
    # Dict order is unstable across Python versions; sort_keys must canonicalise.
    k1 = SuiteKeyBuilder.build(df_fp="x", config_canonical={}, lib_versions={"a": "1", "b": "2"})
    k2 = SuiteKeyBuilder.build(df_fp="x", config_canonical={}, lib_versions={"b": "2", "a": "1"})
    assert k1 == k2


@pytest.mark.parametrize(
    "slot,base,mutated",
    [
        ("df_fp", {"df_fp": "aaa"}, {"df_fp": "bbb"}),
        ("config_canonical", {"config_canonical": {"foo": 1}}, {"config_canonical": {"foo": 2}}),
        ("mlframe_models", {"mlframe_models": ["cb"]}, {"mlframe_models": ["cb", "lgb"]}),
        ("lib_versions", {"lib_versions": {"sklearn": "1.6.0"}}, {"lib_versions": {"sklearn": "1.5.0"}}),
        ("random_seed", {"random_seed": 42}, {"random_seed": 43}),
        ("extra", {"extra": {"target": "t1"}}, {"extra": {"target": "t2"}}),
    ],
)
def test_key_builder_changes_when_any_slot_diff(slot, base, mutated):
    # Common defaults so the only difference is the named slot.
    defaults = dict(df_fp="x", config_canonical={"k": 0})
    k_base = SuiteKeyBuilder.build(**{**defaults, **base})
    k_mut = SuiteKeyBuilder.build(**{**defaults, **mutated})
    assert k_base != k_mut, f"slot {slot!r} did not affect the cache key"


def test_key_builder_none_seed_distinct_from_zero_seed():
    # Explicit None should NOT collide with seed=0 (the common "implicit default").
    k_none = SuiteKeyBuilder.build(df_fp="x", config_canonical={}, random_seed=None)
    k_zero = SuiteKeyBuilder.build(df_fp="x", config_canonical={}, random_seed=0)
    assert k_none != k_zero


# --------------------------------------------------------------------------
# SuiteArtefactCache round-trip
# --------------------------------------------------------------------------


def test_cache_put_get_round_trip(tmp_path):
    cache = SuiteArtefactCache(cache_dir=str(tmp_path), bytes_limit=1_000_000)
    key = SuiteKeyBuilder.build(df_fp="x", config_canonical={"k": 0})
    assert cache.get(key) is None  # miss before put
    payload = {"alpha": [1, 2, 3], "beta": "hello"}
    assert cache.put(key, payload) is True
    got = cache.get(key)
    assert got == payload


def test_cache_get_uses_sentinel_for_cached_none(tmp_path):
    cache = SuiteArtefactCache(cache_dir=str(tmp_path), bytes_limit=1_000_000)
    key = SuiteKeyBuilder.build(df_fp="x", config_canonical={})
    cache.put(key, None)
    # The public get with default=None can't distinguish "cached None" from "miss";
    # the decorator uses the private _MISS sentinel which IS distinguishable.
    assert cache.get(key, default=cache._MISS) is None
    fresh_key = SuiteKeyBuilder.build(df_fp="y", config_canonical={})
    assert cache.get(fresh_key, default=cache._MISS) is cache._MISS


# --------------------------------------------------------------------------
# Decorator + speedup
# --------------------------------------------------------------------------


def test_cache_artefact_hit_faster_than_miss(tmp_path):
    # Use isolated cache, NOT the default singleton, so the test doesn't leak / get leaked into.
    cache = SuiteArtefactCache(cache_dir=str(tmp_path), bytes_limit=10_000_000)

    @cache_artefact("slow_func_under_test", cache=cache)
    def slow(x):
        # Meaningful enough to dominate the cache I/O so the speedup is visible across noise.
        time.sleep(0.05)
        return x * x

    t0 = time.perf_counter()
    r1 = slow(7)
    miss_dt = time.perf_counter() - t0

    t1 = time.perf_counter()
    r2 = slow(7)
    hit_dt = time.perf_counter() - t1

    assert r1 == 49 and r2 == 49
    # The hit MUST be materially faster than the miss; the sleep is 50ms vs disk-load ~ms.
    # 3x is generous enough to absorb Windows CI noise.
    assert hit_dt * 3 < miss_dt, f"hit_dt={hit_dt:.4f}s, miss_dt={miss_dt:.4f}s -- cache not hit"


def test_cache_artefact_miss_on_different_args(tmp_path):
    cache = SuiteArtefactCache(cache_dir=str(tmp_path), bytes_limit=10_000_000)
    call_count = {"n": 0}

    @cache_artefact("counter_func", cache=cache)
    def counted(x):
        call_count["n"] += 1
        return x + 1

    counted(1)
    counted(1)  # hit
    counted(2)  # miss (different arg)
    counted(2)  # hit
    assert call_count["n"] == 2  # only two distinct invocations actually ran


# --------------------------------------------------------------------------
# Size gate + eviction
# --------------------------------------------------------------------------


def test_cache_refuses_oversize_via_size_estimate(tmp_path):
    cache = SuiteArtefactCache(cache_dir=str(tmp_path), bytes_limit=1024)
    key = SuiteKeyBuilder.build(df_fp="big", config_canonical={})
    # Caller knows the artefact is >1KB; cache MUST refuse before writing.
    accepted = cache.put(key, "small string", size_estimate=2048)
    assert accepted is False
    assert cache.get(key) is None
    # No file written.
    assert key not in cache


def test_cache_rolls_back_oversize_written_blob(tmp_path):
    # Tiny budget so even a small dict overshoots once pickled with the sidecar overhead.
    cache = SuiteArtefactCache(cache_dir=str(tmp_path), bytes_limit=32)
    key = SuiteKeyBuilder.build(df_fp="x", config_canonical={})
    # Large enough payload that the resulting pickle exceeds 32 bytes.
    payload = {"data": list(range(200))}
    accepted = cache.put(key, payload)
    assert accepted is False
    # Rollback removed the file.
    assert key not in cache
    assert cache.get(key) is None


def test_cache_evicts_oldest_when_bytes_budget_exceeded(tmp_path):
    # 4kB budget; each entry is ~200 bytes after pickle. Write 30 entries -> some evicted.
    cache = SuiteArtefactCache(cache_dir=str(tmp_path), bytes_limit=4_096)
    keys = []
    for i in range(30):
        k = SuiteKeyBuilder.build(df_fp=f"fp{i}", config_canonical={"i": i})
        keys.append(k)
        # ~200-byte string payload after pickle protocol+sidecar.
        cache.put(k, "x" * 100)
    # Total bytes stayed at or below budget after eviction sweeps.
    assert cache.total_bytes() <= 4_096
    # Some early keys were evicted; some recent keys are still hits.
    early_misses = sum(1 for k in keys[:5] if cache.get(k) is None)
    recent_hits = sum(1 for k in keys[-5:] if cache.get(k) == "x" * 100)
    assert early_misses >= 1, "expected at least one early entry to be evicted under bytes pressure"
    assert recent_hits >= 1, "expected at least one recent entry to be retained"


def test_cache_evicts_oldest_when_entry_cap_exceeded(tmp_path):
    cache = SuiteArtefactCache(cache_dir=str(tmp_path), bytes_limit=10_000_000, max_entries=3)
    keys = [
        SuiteKeyBuilder.build(df_fp=f"fp{i}", config_canonical={"i": i}) for i in range(5)
    ]
    for k in keys:
        cache.put(k, f"value_{k}")
    assert len(cache) <= 3
    # The most-recent 3 entries survived; the earliest 2 evicted.
    assert cache.get(keys[-1]) == f"value_{keys[-1]}"
    assert cache.get(keys[0]) is None


# --------------------------------------------------------------------------
# Diagnostic + edge cases
# --------------------------------------------------------------------------


def test_cache_clear_removes_everything(tmp_path):
    cache = SuiteArtefactCache(cache_dir=str(tmp_path), bytes_limit=10_000_000)
    for i in range(5):
        cache.put(f"k{i:032x}", {"i": i})
    assert len(cache) == 5
    removed = cache.clear()
    assert removed == 5
    assert len(cache) == 0


def test_cache_get_returns_default_on_corrupt_sidecar(tmp_path):
    # Write a legitimate entry, then corrupt its sidecar -- the next get must miss, not raise.
    cache = SuiteArtefactCache(cache_dir=str(tmp_path), bytes_limit=10_000_000)
    key = SuiteKeyBuilder.build(df_fp="x", config_canonical={})
    cache.put(key, {"alpha": 1})
    sidecar = cache._path(key) + ".sha256"
    assert os.path.exists(sidecar)
    with open(sidecar, "w", encoding="utf-8") as fh:
        fh.write("0" * 64 + "  fake\n")
    sentinel = object()
    assert cache.get(key, default=sentinel) is sentinel


def test_default_cache_singleton_is_stable_within_process():
    # Reset state for the duration of this test only, then restore.
    prev = get_default_cache()
    try:
        set_default_cache(None)
        c1 = get_default_cache()
        c2 = get_default_cache()
        assert c1 is c2
    finally:
        set_default_cache(prev)


def test_default_bytes_limit_matches_claude_md_ceiling():
    # CLAUDE.md mandates a 2 GB streaming threshold above which artefacts must NOT be cached in
    # memory; the default must match so an operator who doesn't read the env-var docs still
    # gets the safe-by-default behaviour.
    assert DEFAULT_BYTES_LIMIT == 2_000_000_000


def test_cache_artefact_decorator_preserves_function_name(tmp_path):
    cache = SuiteArtefactCache(cache_dir=str(tmp_path), bytes_limit=10_000_000)

    @cache_artefact("my_thing", cache=cache)
    def my_thing(x):
        """Docstring stays."""
        return x

    assert my_thing.__name__ == "my_thing"
    assert "Docstring stays" in (my_thing.__doc__ or "")
    assert getattr(my_thing, "__wrapped_artefact_name__", None) == "my_thing"
