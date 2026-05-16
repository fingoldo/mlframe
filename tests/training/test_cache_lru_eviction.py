"""LRU eviction caps for DiscoveryCache + LocalDiskBackend.

R&D workflows that loop discovery with varying hyperparams grow the cache
unboundedly without a cap. The opt-in ``max_entries`` / ``max_size_mb``
options keep the on-disk footprint bounded by evicting least-recently-
accessed entries.
"""
from __future__ import annotations

import time

import pytest


def test_discovery_cache_evicts_oldest_when_over_max_entries(tmp_path):
    """When max_entries=3 and 5 keys are written, the 2 oldest evict."""
    from mlframe.training.composite_cache import DiscoveryCache

    cache = DiscoveryCache(str(tmp_path), max_entries=3)
    for i in range(5):
        cache.set(f"key{i:04x}", {"i": i})
        # Stagger the LRU timestamps so the order is deterministic; the
        # sidecar uses time.time() which is wall-clock and could collide
        # at sub-millisecond granularity.
        time.sleep(0.01)

    # Only 3 .pkl files should remain on disk.
    pkls = sorted(p.name for p in tmp_path.glob("*.pkl"))
    assert len(pkls) == 3, f"expected 3 entries after cap, got {pkls}"

    # The first 2 inserted (key0000, key0001) should be gone; the last 3
    # (key0002, key0003, key0004) should be present.
    assert "key0002.pkl" in pkls
    assert "key0003.pkl" in pkls
    assert "key0004.pkl" in pkls
    assert cache.get("key0000") is None
    assert cache.get("key0001") is None
    assert cache.get("key0004") == {"i": 4}


def test_discovery_cache_get_refreshes_lru(tmp_path):
    """A get() bumps LRU so the touched key survives the next eviction."""
    from mlframe.training.composite_cache import DiscoveryCache

    cache = DiscoveryCache(str(tmp_path), max_entries=3)
    for i in range(3):
        cache.set(f"key{i:04x}", {"i": i})
        time.sleep(0.01)

    # Touch the oldest so it becomes MRU; then add a fourth key.
    assert cache.get("key0000") == {"i": 0}
    time.sleep(0.01)
    cache.set("key0003", {"i": 3})

    pkls = {p.name for p in tmp_path.glob("*.pkl")}
    # key0001 was the new oldest and should be the one evicted; key0000
    # survived because get() refreshed its LRU timestamp.
    assert "key0000.pkl" in pkls
    assert "key0001.pkl" not in pkls
    assert "key0002.pkl" in pkls
    assert "key0003.pkl" in pkls


def test_discovery_cache_no_cap_means_unbounded(tmp_path):
    """Default constructor (no caps) preserves pre-cap semantics."""
    from mlframe.training.composite_cache import DiscoveryCache

    cache = DiscoveryCache(str(tmp_path))
    for i in range(10):
        cache.set(f"k{i:04x}", i)
    pkls = list(tmp_path.glob("*.pkl"))
    assert len(pkls) == 10


def test_local_disk_backend_evicts_oldest_when_over_max_entries(tmp_path):
    """LocalDiskBackend honours max_entries the same way."""
    from mlframe.training.feature_handling.cache_backend import LocalDiskBackend

    be = LocalDiskBackend(str(tmp_path / "lru_root"), max_entries=2)
    for i in range(4):
        be.write(f"k{i:04x}", f"payload {i}".encode("utf-8"))
        time.sleep(0.01)

    # Only 2 newest should remain.
    keys = sorted(be.list_keys())
    assert len(keys) == 2, f"expected 2 entries after cap, got {keys}"
    assert "k0002" in keys
    assert "k0003" in keys
    assert not be.exists("k0000")
    assert not be.exists("k0001")


def test_local_disk_backend_max_size_mb_evicts_to_fit(tmp_path):
    """Size cap evicts oldest entries until total bytes fit."""
    from mlframe.training.feature_handling.cache_backend import LocalDiskBackend

    # ~600 KiB total budget; each payload is ~512 KiB so two fit but
    # three don't.
    be = LocalDiskBackend(str(tmp_path / "size_root"), max_size_mb=0.6)
    payload = b"x" * (512 * 1024)
    for i in range(3):
        be.write(f"k{i}", payload)
        time.sleep(0.01)

    keys = set(be.list_keys())
    # Oldest evicted under size cap.
    assert "k0" not in keys
    assert "k2" in keys
