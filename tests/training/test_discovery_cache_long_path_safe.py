"""DiscoveryCache must apply long_path_safe so deep cache trees survive
on Windows. The 260-char MAX_PATH ceiling otherwise crashes ``os.replace``
in ``set()`` even though ``LocalDiskBackend`` (same FS, same operation)
worked because IT already wraps the root through ``long_path_safe``.

These tests pin both the ``cache_dir`` attribute (must be the wrapped
path on Windows) and the end-to-end set/get cycle on a deeply nested
path so a regression that re-introduces the bare ``str(cache_dir)``
construction is caught.
"""

from __future__ import annotations

import sys



def test_discovery_cache_cache_dir_uses_long_path_prefix_on_windows(tmp_path):
    r"""The ``cache_dir`` attribute must carry the ``\\?\`` prefix on
    Windows. ``LocalDiskBackend`` does this; ``DiscoveryCache`` did not
    until now.
    """
    from mlframe.training.composite.cache import DiscoveryCache

    cache = DiscoveryCache(str(tmp_path / "lp_root"))
    if sys.platform == "win32":
        assert cache.cache_dir.startswith("\\\\?\\"), f"cache_dir should carry long-path prefix on Windows; got: {cache.cache_dir!r}"
    else:
        # On POSIX long_path_safe is a no-op - the cache_dir is just the
        # absolute path.
        import os

        assert cache.cache_dir == os.path.abspath(str(tmp_path / "lp_root"))


def test_discovery_cache_set_get_on_deeply_nested_path(tmp_path):
    """End-to-end: set/get survives a path that easily exceeds 260 chars.

    Without ``long_path_safe`` the inner ``os.replace`` would raise
    ``FileNotFoundError`` / ``OSError`` on Windows. POSIX systems have
    no MAX_PATH so this just exercises the happy path there.
    """
    from mlframe.training.composite.cache import DiscoveryCache

    # Build a nested chain whose total path length exceeds 260 chars.
    nested = tmp_path
    for i in range(8):
        nested = nested / f"segment_{i:02d}_padding_{'x' * 30}"

    cache = DiscoveryCache(str(nested))
    cache.set("longpathkey", {"payload": "ok"})
    assert cache.get("longpathkey") == {"payload": "ok"}
