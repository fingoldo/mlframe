"""Audit 2026-11-20 finding 29-F10: a value that cannot be pickled is not cached and raises nothing, and leaves no temp file behind.

A value holding a thread lock raised TypeError out of put() and left a tmp_ file that eviction never reclaims. The test puts such a value,
checks put returns without raising, and checks the cache directory has no leftover temp file.
"""

import threading

from mlframe.utils import disk_cache as dc


def test_unpicklable_value_is_not_cached_and_leaves_no_temp_file(tmp_path):
    """put of a lock-holding value returns normally, nothing is retrievable, and no tmp_ file remains."""
    cache = dc.DiskCache(tmp_path)
    cache.put("locked", {"lock": threading.Lock()})
    assert cache.get("locked") is None
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("tmp_")]
