"""Tests for ``data_signature`` + ``make_discovery_cache_key`` + ``DiscoveryCache`` (R10c brainstorm round-2 extension E).

Disk-backed cache for CompositeTargetDiscovery results. R&D workflows that re-run discovery with the same data + config should hit the cache; differing data / config / random_state should miss.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import (
    DiscoveryCache,
    data_signature,
    make_discovery_cache_key,
)


class TestDataSignature:
    def test_same_df_same_signature(self) -> None:
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "a": rng.normal(size=2000),
                "b": rng.normal(size=2000),
                "y": rng.normal(size=2000),
            }
        )
        s1 = data_signature(df, "y", ["a", "b"], sample_n=500, random_state=0)
        s2 = data_signature(df, "y", ["a", "b"], sample_n=500, random_state=0)
        assert s1 == s2

    def test_different_target_different_signature(self) -> None:
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(size=1000)})
        s1 = data_signature(df, "a", ["b"], sample_n=500, random_state=0)
        s2 = data_signature(df, "b", ["a"], sample_n=500, random_state=0)
        assert s1 != s2

    def test_different_features_different_signature(self) -> None:
        rng = np.random.default_rng(2)
        df = pd.DataFrame(
            {
                "a": rng.normal(size=1000),
                "b": rng.normal(size=1000),
                "c": rng.normal(size=1000),
                "y": rng.normal(size=1000),
            }
        )
        s_ab = data_signature(df, "y", ["a", "b"], random_state=0)
        s_ac = data_signature(df, "y", ["a", "c"], random_state=0)
        assert s_ab != s_ac

    def test_modified_data_changes_signature(self) -> None:
        rng = np.random.default_rng(3)
        df1 = pd.DataFrame({"a": rng.normal(size=1000), "y": rng.normal(size=1000)})
        df2 = df1.copy()
        df2.loc[0, "y"] = 999.0  # one cell changed
        s1 = data_signature(df1, "y", ["a"], sample_n=1000, random_state=0)
        s2 = data_signature(df2, "y", ["a"], sample_n=1000, random_state=0)
        assert s1 != s2

    def test_empty_df_returns_stable_signature(self) -> None:
        df = pd.DataFrame({"a": [], "y": []})
        s = data_signature(df, "y", ["a"])
        assert isinstance(s, str)
        assert len(s) == 32  # blake2b 16-byte hex


class TestCacheKey:
    def test_reproducible(self) -> None:
        k1 = make_discovery_cache_key("abc", "target", "config_sig", 42)
        k2 = make_discovery_cache_key("abc", "target", "config_sig", 42)
        assert k1 == k2

    def test_different_inputs_different_keys(self) -> None:
        k1 = make_discovery_cache_key("abc", "target", "config_sig", 42)
        k2 = make_discovery_cache_key("xyz", "target", "config_sig", 42)
        k3 = make_discovery_cache_key("abc", "different", "config_sig", 42)
        k4 = make_discovery_cache_key("abc", "target", "different_config", 42)
        k5 = make_discovery_cache_key("abc", "target", "config_sig", 43)
        assert len({k1, k2, k3, k4, k5}) == 5  # all distinct


class TestDiscoveryCache:
    def test_set_and_get(self, tmp_path) -> None:
        tmp = str(tmp_path)
        cache = DiscoveryCache(tmp)
        cache.set("abc123", {"specs": [{"name": "spec1", "mi_gain": 0.5}]})
        out = cache.get("abc123")
        assert out == {"specs": [{"name": "spec1", "mi_gain": 0.5}]}

    def test_contains(self, tmp_path) -> None:
        tmp = str(tmp_path)
        cache = DiscoveryCache(tmp)
        assert "abc" not in cache
        cache.set("abc", "value")
        assert "abc" in cache

    def test_get_default_when_missing(self, tmp_path) -> None:
        tmp = str(tmp_path)
        cache = DiscoveryCache(tmp)
        assert cache.get("missing") is None
        assert cache.get("missing", default="fallback") == "fallback"

    def test_invalidate(self, tmp_path) -> None:
        tmp = str(tmp_path)
        cache = DiscoveryCache(tmp)
        cache.set("abc", "value")
        assert cache.invalidate("abc") is True
        assert cache.invalidate("abc") is False  # already gone
        assert "abc" not in cache

    def test_clear(self, tmp_path) -> None:
        tmp = str(tmp_path)
        cache = DiscoveryCache(tmp)
        for i in range(5):
            cache.set(f"key{i}", f"value{i}")
        removed = cache.clear()
        assert removed == 5
        for i in range(5):
            assert f"key{i}" not in cache

    def test_unsafe_key_sanitised_or_rejected(self, tmp_path) -> None:
        # Audit H-COMP-09: _safe_key now hashes any non-hex key via
        # blake2b (collision-proof; the legacy "strip non-alnum"
        # sanitiser collapsed abc-def and abcdef to the same filename).
        # Path-traversal protection is preserved by construction --
        # the hashed filename never contains "/" or "..".
        tmp = str(tmp_path)
        cache = DiscoveryCache(tmp)
        # Empty key -> ValueError.
        with pytest.raises(ValueError, match="empty"):
            cache.set("", "value")
        # Path-traversal characters get hashed into a safe filename; the
        # write succeeds but cannot escape ``cache_dir``.
        cache.set("../../etc/passwd", "value")
        outside_target = os.path.join(tmp, "..", "..", "etc", "passwd")
        assert not os.path.exists(outside_target), "Path-traversal must not escape cache dir"
        # And the value round-trips.
        assert cache.get("../../etc/passwd") == "value"

    def test_atomic_write_does_not_leave_partial_files(self, tmp_path) -> None:
        """Atomic write via tmp-file rename: even if pickle fails mid-write, no partial file is left at the target path."""
        tmp = str(tmp_path)
        cache = DiscoveryCache(tmp)
        # Build an unpicklable object to force the inner write to raise.
        unpicklable = lambda x: x
        try:
            cache.set("abc", unpicklable)
        except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
            pass
        # No file should exist at the target path.
        assert "abc" not in cache

    def test_get_corrupt_file_returns_default(self, tmp_path) -> None:
        """Manually corrupt a cache file -> ``get`` returns default rather than crash."""
        tmp = str(tmp_path)
        cache = DiscoveryCache(tmp)
        path = os.path.join(tmp, "abc.pkl")
        with open(path, "wb") as f:
            f.write(b"this is not a pickle stream")
        assert cache.get("abc", default="fallback") == "fallback"


class TestEndToEndScenario:
    def test_cache_hit_skips_recomputation(self, tmp_path) -> None:
        """The full R&D workflow: hash data + config, look up cache, fall back to expensive computation on miss, store result, second call hits."""
        tmp = str(tmp_path)
        cache = DiscoveryCache(tmp)
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=500), "y": rng.normal(size=500)})
        sig = data_signature(df, "y", ["a"], random_state=0)
        key = make_discovery_cache_key(sig, "y", "config_hash_42", 42)
        # Cache miss on first call.
        assert key not in cache
        # Simulate expensive computation result.
        specs = [{"name": "y__diff__a", "mi_gain": 0.42}]
        cache.set(key, specs)
        # Cache hit on second call.
        assert key in cache
        assert cache.get(key) == specs

    def test_modified_data_misses_cache(self, tmp_path) -> None:
        """Same key construction, but the data changed -> signature changes -> cache key changes -> miss."""
        tmp = str(tmp_path)
        cache = DiscoveryCache(tmp)
        rng = np.random.default_rng(0)
        df1 = pd.DataFrame({"a": rng.normal(size=500), "y": rng.normal(size=500)})
        sig1 = data_signature(df1, "y", ["a"], random_state=0)
        key1 = make_discovery_cache_key(sig1, "y", "config", 42)
        cache.set(key1, "result_for_df1")
        # Modify df.
        df2 = df1.copy()
        df2.loc[0, "y"] = 999.0
        sig2 = data_signature(df2, "y", ["a"], sample_n=1000, random_state=0)
        key2 = make_discovery_cache_key(sig2, "y", "config", 42)
        assert key1 != key2
        assert key2 not in cache
