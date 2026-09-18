"""DiscoveryCache: an entry written by one process run must be readable by the next, and a plain miss must not log at ERROR."""
from __future__ import annotations

import logging
import os

from mlframe.training.composite.cache_store import DiscoveryCache

KEY = "438023d2541e3d1192ab75bc85ccf0d8"
PAYLOAD = {"specs_export": [{"name": "a"}], "failures": [], "filter_drops": {"x": 1}}


def _deep_dir(tmp_path):
    # > 260 chars so the Windows extended-length prefix path is actually exercised.
    d = tmp_path
    for i in range(12):
        d = d / f"nested_directory_level_{i:02d}"
    return str(d / ".discovery_cache")


def test_set_then_get_from_fresh_instance_long_path(tmp_path, caplog):
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
    cache = DiscoveryCache(str(tmp_path / "dc"))
    with caplog.at_level(logging.DEBUG):
        assert cache.get("ab" * 16, default="MISS") == "MISS"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], [r.getMessage() for r in caplog.records]


def test_del_is_silent_when_logging_torn_down(tmp_path, monkeypatch, capsys):
    import mlframe.training.composite.cache_store as cs

    cache = DiscoveryCache(str(tmp_path / "dc"))
    cache.set(KEY, PAYLOAD)
    cache._lru_dirty = True

    class _BrokenLogger:
        def debug(self, *a, **k):
            raise TypeError("'NoneType' object is not callable")

    def _boom(*a, **k):
        raise OSError("disk gone")

    monkeypatch.setattr(cs, "logger", _BrokenLogger())
    monkeypatch.setattr(cache, "_flush_lru", _boom)
    cache.__del__()  # must not raise
