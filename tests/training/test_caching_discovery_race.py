"""Regression tests for DiscoveryCache.get atomicity (P1).

Pre-fix ``get`` called ``os.path.exists`` then ``open``; on Windows a
delete-between-the-two raised ``FileNotFoundError`` to the caller.
Post-fix the existence check is gone and any open failure is treated
as a cache miss (returns ``default`` cleanly).
"""

from __future__ import annotations

from unittest import mock

from mlframe.training.composite.cache import DiscoveryCache


def test_get_returns_default_on_missing_file_without_raising(tmp_path):
    cache = DiscoveryCache(str(tmp_path))
    assert cache.get("deadbeef", default="fallback") == "fallback"


def test_get_treats_delete_between_check_and_open_as_miss(tmp_path):
    """Simulate the race: ``open`` raises FileNotFoundError mid-call."""
    cache = DiscoveryCache(str(tmp_path))
    # Pre-write so the key 'exists' from the caller's POV.
    cache.set("abc123", {"specs": []})
    # Now make the next open() raise FNF, mimicking a delete-during-read.
    with mock.patch(
        "builtins.open",
        side_effect=FileNotFoundError("simulated race-with-delete"),
    ):
        result = cache.get("abc123", default="missed")
    # Should return default cleanly, NOT bubble the exception.
    assert result == "missed"


def test_get_treats_corrupt_pickle_as_miss(tmp_path):
    """A truncated / corrupt file must not raise; return default."""
    import os
    cache = DiscoveryCache(str(tmp_path))
    path = cache._path("zzz999")
    with open(path, "wb") as f:
        f.write(b"not-a-real-pickle-blob")
    assert cache.get("zzz999", default="fallback") == "fallback"
    # Sentinel: writing then reading a valid value works.
    cache.set("zzz999", {"ok": True})
    assert cache.get("zzz999") == {"ok": True}
    if os.path.exists(path):
        os.remove(path)
