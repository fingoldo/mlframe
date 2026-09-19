"""DiscoveryCache: an entry written by one process run must be readable by the next, and a plain miss must not log at ERROR."""
from __future__ import annotations

import logging
import os

from mlframe.training.composite.cache_store import DiscoveryCache

KEY = "438023d2541e3d1192ab75bc85ccf0d8"
PAYLOAD = {"specs_export": [{"name": "a"}], "failures": [], "filter_drops": {"x": 1}}


def _deep_dir(tmp_path):
    """A cache directory path longer than 260 characters."""
    # > 260 chars so the Windows extended-length prefix path is actually exercised.
    d = tmp_path
    for i in range(12):
        d = d / f"nested_directory_level_{i:02d}"
    return str(d / ".discovery_cache")


def test_set_then_get_from_fresh_instance_long_path(tmp_path, caplog):
    """An entry written by one cache instance is read back by a fresh instance on a >260-char path."""
    cache_dir = _deep_dir(tmp_path)
    assert len(cache_dir) > 260
    writer = DiscoveryCache(cache_dir)
    writer.set(KEY, PAYLOAD)
    writer.close()
    del writer
    reader = DiscoveryCache(cache_dir)  # simulates run N+1
    with caplog.at_level(logging.DEBUG):
        got = reader.get(KEY, default="MISS")
    assert got == PAYLOAD
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert os.path.isfile(os.path.join(reader.cache_dir, KEY + ".pkl.sha256"))


def test_plain_miss_is_not_error(tmp_path, caplog):
    """A plain cache miss returns the default without logging a warning or error."""
    cache = DiscoveryCache(str(tmp_path / "dc"))
    with caplog.at_level(logging.DEBUG):
        assert cache.get("ab" * 16, default="MISS") == "MISS"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], [r.getMessage() for r in caplog.records]


def test_del_is_silent_when_logging_torn_down(tmp_path, monkeypatch, capsys):
    """__del__ with a failing flush and a torn-down logger neither raises nor prints."""
    import mlframe.training.composite.cache_store as cs

    cache = DiscoveryCache(str(tmp_path / "dc"))
    cache.set(KEY, PAYLOAD)
    cache._lru_dirty = True

    class _BrokenLogger:
        """Logger whose debug raises the TypeError seen at interpreter shutdown."""
        def debug(self, *a, **k):
            """Raise like a torn-down logging module."""
            raise TypeError("'NoneType' object is not callable")

    def _boom(*a, **k):
        """Flush that fails like a vanished disk."""
        raise OSError("disk gone")

    monkeypatch.setattr(cs, "logger", _BrokenLogger())
    flush_calls = []

    def _boom_recorded(*a, **k):
        """Record the flush attempt, then fail."""
        flush_calls.append(1)
        _boom()

    monkeypatch.setattr(cache, "_flush_lru", _boom_recorded)
    cache.__del__()  # must not raise
    assert flush_calls, "__del__ did not attempt the LRU flush"
    captured = capsys.readouterr()
    assert captured.err == "" and captured.out == "", captured
