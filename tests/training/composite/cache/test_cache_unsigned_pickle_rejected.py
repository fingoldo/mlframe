"""An unsigned pickle in the composite discovery cache is not unpickled by default (DESER-10).

Before the fix the cache forced ``allow_unverified=True``, so a file with no sha256 sidecar was unpickled on read: whoever
could write to a shared cache directory got code execution. The environment variable is now the single switch.
"""

from __future__ import annotations

import pickle

import pytest


class _Marker:
    """A value whose unpickling leaves a visible trace, so the test can tell that the payload ran."""

    def __reduce__(self):
        """Return a callable that records that unpickling reached the payload."""
        return (_record, ())


_RAN: list[str] = []


def _record() -> str:
    """Record that the pickled callable executed."""
    _RAN.append("ran")
    return "ran"


def _write_unsigned(path, value) -> None:
    """Write ``value`` as a plain pickle with no sha256 sidecar next to it."""
    with open(path, "wb") as fh:
        pickle.dump(value, fh)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Start every test with the migration opt-out unset and the strict switch unset."""
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    monkeypatch.delenv("MLFRAME_DISCOVERY_CACHE_STRICT", raising=False)
    _RAN.clear()


def test_unsigned_entry_is_not_unpickled_by_default(tmp_path, monkeypatch):
    """With the opt-out unset, a sidecar-less cache file is a miss and its payload never runs."""
    from mlframe.training.composite.cache_store import DiscoveryCache

    cache = DiscoveryCache(str(tmp_path))
    key = "k"
    target = cache._path(key)
    _write_unsigned(target, _Marker())
    assert cache.get(key, default="MISS") == "MISS"
    assert _RAN == []


def test_unsigned_entry_is_readable_with_the_migration_opt_out(tmp_path, monkeypatch):
    """With MLFRAME_ALLOW_UNVERIFIED_PICKLE=1 the legacy entry is still read (the documented migration path)."""
    from mlframe.training.composite.cache_store import DiscoveryCache

    monkeypatch.setenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", "1")
    cache = DiscoveryCache(str(tmp_path))
    key = "k"
    target = cache._path(key)
    _write_unsigned(target, {"legacy": 1})
    assert cache.get(key, default="MISS") == {"legacy": 1}
