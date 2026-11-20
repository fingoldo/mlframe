"""Audit 2026-11-20 finding 39-F19: low system RAM evicts one entry per check, not the whole cache; a size overrun still evicts until under the limit."""

from types import SimpleNamespace

import pytest

from mlframe.training.feature_handling import cache as cache_mod
from mlframe.training.feature_handling.cache import FeatureCache, _CacheEntry


def _cache(ram_max_gb=None, ram_reserve_gb=1.0):
    """A FeatureCache with three small in-memory entries."""
    cfg = SimpleNamespace(ram_max_gb=ram_max_gb, ram_reserve_gb=ram_reserve_gb, eviction_strategy="lru")
    c = FeatureCache(cfg)
    for name in ("a", "b", "c"):
        c._mem[name] = _CacheEntry(value=None, size_bytes=10, recompute_time_s=1.0)
    return c


def _low_ram(monkeypatch, available):
    """Pretend system RAM available is ``available`` bytes."""
    monkeypatch.setattr(cache_mod.psutil, "virtual_memory", lambda: SimpleNamespace(available=available))


def test_low_ram_alone_evicts_exactly_one_entry(monkeypatch):
    """Reserve breached but the cache is within its size limit: one eviction, then return."""
    _low_ram(monkeypatch, available=0)
    c = _cache(ram_max_gb=None, ram_reserve_gb=1.0)
    c._evict_if_needed_locked()
    assert len(c._mem) == 2
    assert c._evictions == 1


def test_size_overrun_still_evicts_until_under_the_limit(monkeypatch):
    """Cache of 30 bytes against a 15-byte limit: evicts until 10 bytes remain (two evictions), with RAM healthy."""
    _low_ram(monkeypatch, available=10**12)
    c = _cache(ram_max_gb=15e-9, ram_reserve_gb=None)
    c._evict_if_needed_locked()
    assert sum(e.size_bytes for e in c._mem.values()) <= 15
    assert c._evictions == 2


@pytest.mark.parametrize("available", [10**12])
def test_healthy_ram_and_size_evicts_nothing(monkeypatch, available):
    """Neither condition holds: nothing is evicted."""
    _low_ram(monkeypatch, available=available)
    c = _cache(ram_max_gb=None, ram_reserve_gb=1.0)
    c._evict_if_needed_locked()
    assert len(c._mem) == 3
