"""Audit 2026-11-20 finding 29-F11: an entry whose pickled class no longer exists is a miss, not an exception out of get().

A refactor that renames a pickled class makes the unpickler raise AttributeError or ImportError. The test forces the loader to raise
AttributeError and checks get() returns None and drops the entry.
"""

from mlframe.utils import disk_cache as dc


def test_stale_class_entry_is_a_miss_and_is_dropped(tmp_path, monkeypatch):
    """get() on an entry that fails to unpickle returns None, and the entry file is removed."""
    cache = dc.DiskCache(tmp_path)
    key = "entry"
    path = cache._key_path(key)
    path.write_bytes(b"payload")

    def stale_loader(_path):
        """Simulate unpickling an instance of a class that was renamed away."""
        raise AttributeError("Can't get attribute 'OldName' on module")

    monkeypatch.setattr(dc, "safe_load", stale_loader)
    assert cache.get(key) is None
    assert not path.exists()
