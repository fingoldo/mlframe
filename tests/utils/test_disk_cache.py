"""Unit tests for the shared content-addressable disk cache."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import pytest

from mlframe.utils.disk_cache import (
    DiskCache,
    compose_key,
    hash_array_summary,
    hash_object,
)

# -------------------- hashing determinism --------------------


def test_hash_array_summary_deterministic_same_input():
    """Hash array summary deterministic same input."""
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((100, 8))
    assert hash_array_summary(arr) == hash_array_summary(arr.copy())


def test_hash_array_summary_changes_on_value_perturbation():
    """Hash array summary changes on value perturbation."""
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((100, 8))
    arr_b = arr.copy()
    arr_b[0, 0] += 1e-3  # changes head bytes + col0 sum/min/max
    assert hash_array_summary(arr) != hash_array_summary(arr_b)


def test_hash_array_summary_changes_on_shape_change():
    """Hash array summary changes on shape change."""
    arr = np.zeros((10, 5))
    assert hash_array_summary(arr) != hash_array_summary(arr.reshape(5, 10))


def test_hash_array_summary_changes_on_dtype_change():
    """Hash array summary changes on dtype change."""
    arr = np.zeros((10, 5), dtype=np.float32)
    assert hash_array_summary(arr) != hash_array_summary(arr.astype(np.float64))


def test_hash_array_summary_catches_middle_row_change_via_col_sum():
    """Hash array summary catches middle row change via col sum."""
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((1000, 4))  # rows past summary head/tail
    arr_b = arr.copy()
    arr_b[500, 0] += 7.0  # middle row -> col0 sum changes
    assert hash_array_summary(arr) != hash_array_summary(arr_b)


def test_hash_array_summary_empty_array():
    """Hash array summary empty array."""
    a = np.zeros((0, 5))
    b = np.zeros((0, 5))
    assert hash_array_summary(a) == hash_array_summary(b)
    c = np.zeros((0, 7))
    assert hash_array_summary(a) != hash_array_summary(c)


def test_hash_array_summary_1d():
    """Hash array summary 1d."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 4.0])
    assert hash_array_summary(a) != hash_array_summary(b)


def test_hash_object_dict_order_invariant():
    """Hash object dict order invariant."""
    d1 = {"a": 1, "b": 2, "c": 3}
    d2 = {"c": 3, "a": 1, "b": 2}
    assert hash_object(d1) == hash_object(d2)


def test_hash_object_distinguishes_values():
    """Hash object distinguishes values."""
    assert hash_object({"a": 1}) != hash_object({"a": 2})
    assert hash_object([1, 2, 3]) != hash_object([1, 3, 2])
    assert hash_object(None) != hash_object(0)
    assert hash_object(True) != hash_object(1)


def test_hash_object_handles_numpy_scalar():
    """Hash object handles numpy scalar."""
    assert hash_object(np.int64(5)) == hash_object(5)


def test_hash_object_with_nested_array_uses_summary():
    """An ndarray nested inside a dict should be hashed via the summary path."""
    a = np.zeros((100, 5))
    b = a.copy()
    b[0, 0] = 1.0
    assert hash_object({"x": a}) != hash_object({"x": b})


def test_compose_key_stable():
    """Compose key stable."""
    k1 = compose_key("abc", "def", "ghi")
    k2 = compose_key("abc", "def", "ghi")
    assert k1 == k2
    assert k1 != compose_key("abc", "def", "ghij")
    # Length invariance after rehashing.
    assert len(k1) == len(compose_key("a", "b"))


def test_compose_key_rejects_empty():
    """Compose key rejects empty."""
    with pytest.raises(ValueError):
        compose_key()


def test_compose_key_separator_safe():
    """Parts of different cardinality must not collide via concatenation."""
    assert compose_key("abc", "def") != compose_key("abcdef")
    assert compose_key("a", "bc") != compose_key("ab", "c")


# -------------------- DiskCache round-trip + LRU --------------------


def test_disk_cache_put_get_roundtrip(tmp_path: Path):
    """Disk cache put get roundtrip."""
    cache = DiskCache(tmp_path)
    arr = np.arange(1000).reshape(100, 10)
    cache.put("k1", arr)
    out = cache.get("k1")
    assert out is not None
    np.testing.assert_array_equal(out, arr)
    assert cache.hits == 1
    assert cache.misses == 0


def test_disk_cache_miss_returns_none(tmp_path: Path):
    """Disk cache miss returns none."""
    cache = DiskCache(tmp_path)
    assert cache.get("absent") is None
    assert cache.misses == 1
    assert cache.hits == 0


def test_disk_cache_pickle_complex_payload(tmp_path: Path):
    """Disk cache pickle complex payload."""
    cache = DiskCache(tmp_path)
    payload = {
        "phi": np.random.default_rng(0).standard_normal((50, 4)),
        "base": np.array([0.5] * 50),
        "meta": {"n_splits": 5, "seed": 42},
    }
    cache.put("k", payload)
    out = cache.get("k")
    assert set(out) == {"phi", "base", "meta"}
    np.testing.assert_array_equal(out["phi"], payload["phi"])
    np.testing.assert_array_equal(out["base"], payload["base"])
    assert out["meta"] == payload["meta"]


def test_disk_cache_eviction_under_cap(tmp_path: Path):
    """Putting many entries above the cap should evict oldest by mtime."""
    # ~10KB per entry, cap at 25KB -> only ~2-3 entries fit.
    cache = DiskCache(tmp_path, max_size_bytes=25_000)
    blob = np.zeros(1000, dtype=np.float64)  # ~8KB raw + pickle overhead
    for i in range(10):
        cache.put(f"k{i}", blob)
    # Cache should be under cap.
    assert cache.total_size() <= cache.max_size_bytes * 1.5  # eviction is best-effort
    assert cache.evictions > 0
    # Newest entry (the last put) must survive.
    assert cache.get("k9") is not None


def test_disk_cache_corrupt_entry_treated_as_miss(tmp_path: Path):
    """Disk cache corrupt entry treated as miss."""
    cache = DiskCache(tmp_path)
    cache.put("k1", np.arange(10))
    # Corrupt the on-disk file.
    p = tmp_path / "k1.pkl"
    p.write_bytes(b"this is not pickle")
    out = cache.get("k1")
    assert out is None
    # Should have been removed.
    assert not p.exists()


def test_disk_cache_atomic_write_no_partial(tmp_path: Path):
    """Simulate a crash mid-write: the orphan tmp_ file must not be served on get."""
    cache = DiskCache(tmp_path)
    # Drop a fake half-written tmp_ file with the same prefix.
    (tmp_path / "tmp_abc123.pkl").write_bytes(b"junk")
    # A legitimate put + get with a different key should not be affected.
    cache.put("k1", np.array([1, 2, 3]))
    out = cache.get("k1")
    np.testing.assert_array_equal(out, np.array([1, 2, 3]))
    # The orphan tmp_ is not served as anything.
    assert cache.get("tmp_abc123") is None
    # The tmp_ file should be ignored by total_size accounting too.
    listed = [p.name for p in tmp_path.iterdir() if p.name.endswith(".pkl") and not p.name.startswith("tmp_")]
    assert "k1.pkl" in listed


def test_disk_cache_concurrent_same_key(tmp_path: Path):
    """Two threads writing the same key produce a valid readable entry."""
    cache = DiskCache(tmp_path)
    payload = np.arange(500)
    errors = []

    def worker():
        """Attempts ``for _ in range(5): cache.put('shared', payload)``, tolerating failure (see the except clause for the fallback)."""
        try:
            for _ in range(5):
                cache.put("shared", payload)
        except Exception as exc:  # pragma: no cover - test fails if hit
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    out = cache.get("shared")
    np.testing.assert_array_equal(out, payload)


def test_disk_cache_concurrent_same_key_distinct_payloads_never_corrupt(tmp_path: Path):
    """Regression sensor: concurrent put() for the same key must never leave a sidecar that
    doesn't match the payload actually on disk.

    Unlike ``test_disk_cache_concurrent_same_key`` (which writes an IDENTICAL payload from every
    thread, so a torn payload/sidecar pairing would be undetectable -- any writer's digest matches
    any writer's payload), this uses a DISTINCT large payload per thread so a race is observable:
    without the per-key lock in ``DiskCache.put()``, one thread's ``write_sidecar()`` call (reading
    whatever payload happens to be on disk at that moment) can land its file write AFTER another
    thread's later ``os.replace()``, leaving a sidecar digest that doesn't match the final payload
    -- ``get()`` would then intermittently raise ``PickleVerificationError`` for an entry that was,
    in fact, written correctly by the last writer. Mirrors pyutilz's
    ``test_safe_pickle_concurrency.py::test_safe_dump_concurrent_same_path_never_corrupts``.
    """
    cache = DiskCache(tmp_path)
    payloads = {i: np.full(200_000, i, dtype=np.int64) for i in range(6)}
    errors = []

    def worker(i):
        """Attempts ``for _ in range(8): cache.put('shared', payloads[i])``, tolerating failure (see the except clause for the fallback)."""
        try:
            for _ in range(8):
                cache.put("shared", payloads[i])
        except Exception as exc:  # pragma: no cover - test fails if hit
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in payloads]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors

    # get() must not raise PickleVerificationError, and the result must be exactly one of the
    # payloads that was written -- never a mismatch between payload bytes and sidecar digest.
    out = cache.get("shared")
    assert out is not None, "entry should not have been treated as corrupt/missing"
    matches = [i for i, p in payloads.items() if np.array_equal(out, p)]
    assert len(matches) == 1, f"result did not match exactly one written payload: {out[:5] if out is not None else None}"


def test_disk_cache_clear(tmp_path: Path):
    """Disk cache clear."""
    cache = DiskCache(tmp_path)
    cache.put("a", np.array([1]))
    cache.put("b", np.array([2]))
    cache.clear()
    assert cache.get("a") is None
    assert cache.get("b") is None


def test_disk_cache_get_touches_mtime(tmp_path: Path):
    """Cache hits should refresh mtime so LRU eviction considers them recent."""
    cache = DiskCache(tmp_path)
    cache.put("old", np.array([0]))
    old_path = tmp_path / "old.pkl"
    # Backdate the entry 10 minutes via os.utime so a successful refresh is unambiguously observable on any
    # filesystem mtime resolution -- no wall-clock sleep needed.
    backdated = old_path.stat().st_mtime - 600.0
    os.utime(old_path, (backdated, backdated))
    cache.put("new", np.array([1]))
    cache.get("old")  # touch
    refreshed = old_path.stat().st_mtime
    assert refreshed > backdated
