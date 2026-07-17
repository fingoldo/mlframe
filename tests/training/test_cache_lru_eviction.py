"""LRU eviction caps for DiscoveryCache + LocalDiskBackend.

R&D workflows that loop discovery with varying hyperparams grow the cache
unboundedly without a cap. The opt-in ``max_entries`` / ``max_size_mb``
options keep the on-disk footprint bounded by evicting least-recently-
accessed entries.

Pre-fix this file used ``time.sleep(0.01)`` between writes to coax wall-clock LRU
timestamps into strict ordering. That's flaky on slow CI (sub-millisecond clock granularity
can collide). We now monkeypatch the cache modules' ``time.time`` to a monotonic counter so
LRU ordering is exact and the tests are deterministic.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def monotonic_lru_clock(monkeypatch):
    """Force a strictly-increasing fake clock for both cache modules so LRU ordering is
    deterministic without wall-clock sleeps."""
    from mlframe.training.composite import cache as cc_mod
    from mlframe.training.feature_handling import cache_backend as cb_mod

    counter = {"t": 1_700_000_000.0}

    def _tick():
        counter["t"] += 1.0
        return counter["t"]

    # Both modules `import time` and call `time.time()`. We replace the time module bound on
    # each so the patch only affects the cache modules, not unrelated callers.
    import time as real_time
    import types

    fake_time_mod = types.SimpleNamespace(time=_tick, sleep=real_time.sleep)
    monkeypatch.setattr(cc_mod, "time", fake_time_mod, raising=False)
    monkeypatch.setattr(cb_mod, "time", fake_time_mod, raising=False)
    return counter


def test_discovery_cache_evicts_oldest_when_over_max_entries(tmp_path, monotonic_lru_clock):
    """When max_entries=3 and 5 keys are written, the 2 oldest evict.

    Note: ``DiscoveryCache`` hashes non-pure-hex keys via blake2b before writing
    them to disk; on-disk file names are ``<digest>__h<len>.pkl``, not the
    literal key. Verify eviction via ``cache.get()`` (which round-trips through
    the same hash) plus an on-disk file-count check.
    """
    from mlframe.training.composite.cache import DiscoveryCache

    cache = DiscoveryCache(str(tmp_path), max_entries=3)
    for i in range(5):
        cache.set(f"key{i:04x}", {"i": i})

    # Only 3 .pkl files should remain on disk.
    pkls = sorted(p.name for p in tmp_path.glob("*.pkl"))
    assert len(pkls) == 3, f"expected 3 entries after cap, got {pkls}"

    # The first 2 inserted (key0000, key0001) should be gone; the last 3
    # (key0002, key0003, key0004) should be present.
    assert cache.get("key0000") is None
    assert cache.get("key0001") is None
    assert cache.get("key0002") == {"i": 2}
    assert cache.get("key0003") == {"i": 3}
    assert cache.get("key0004") == {"i": 4}


def test_discovery_cache_get_refreshes_lru(tmp_path, monotonic_lru_clock):
    """A get() bumps LRU so the touched key survives the next eviction."""
    from mlframe.training.composite.cache import DiscoveryCache

    cache = DiscoveryCache(str(tmp_path), max_entries=3)
    for i in range(3):
        cache.set(f"key{i:04x}", {"i": i})

    # Touch the oldest so it becomes MRU; then add a fourth key.
    assert cache.get("key0000") == {"i": 0}
    cache.set("key0003", {"i": 3})

    # key0001 was the new oldest and should be the one evicted; key0000
    # survived because get() refreshed its LRU timestamp. Verify via the
    # cache API rather than literal file names (DiscoveryCache hashes
    # non-hex keys before disk).
    assert cache.get("key0000") == {"i": 0}
    assert cache.get("key0001") is None
    assert cache.get("key0002") == {"i": 2}
    assert cache.get("key0003") == {"i": 3}


def test_discovery_cache_no_cap_means_unbounded(tmp_path):
    """Default constructor (no caps) preserves pre-cap semantics. No clock-control needed
    because no eviction runs."""
    from mlframe.training.composite.cache import DiscoveryCache

    cache = DiscoveryCache(str(tmp_path))
    for i in range(10):
        cache.set(f"k{i:04x}", i)
    pkls = list(tmp_path.glob("*.pkl"))
    assert len(pkls) == 10


def test_local_disk_backend_evicts_oldest_when_over_max_entries(tmp_path, monotonic_lru_clock):
    """LocalDiskBackend honours max_entries the same way."""
    from mlframe.training.feature_handling.cache_backend import LocalDiskBackend

    be = LocalDiskBackend(str(tmp_path / "lru_root"), max_entries=2)
    for i in range(4):
        be.write(f"k{i:04x}", f"payload {i}".encode())

    # Only 2 newest should remain.
    keys = sorted(be.list_keys())
    assert len(keys) == 2, f"expected 2 entries after cap, got {keys}"
    assert "k0002" in keys
    assert "k0003" in keys
    assert not be.exists("k0000")
    assert not be.exists("k0001")


def test_local_disk_backend_max_size_mb_evicts_to_fit(tmp_path, monotonic_lru_clock):
    """Size cap evicts oldest entries until total bytes fit."""
    from mlframe.training.feature_handling.cache_backend import LocalDiskBackend

    # ~600 KiB total budget; each payload is ~512 KiB so two fit but
    # three don't.
    be = LocalDiskBackend(str(tmp_path / "size_root"), max_size_mb=0.6)
    payload = b"x" * (512 * 1024)
    for i in range(3):
        be.write(f"k{i}", payload)

    keys = set(be.list_keys())
    # Oldest evicted under size cap.
    assert "k0" not in keys
    assert "k2" in keys
