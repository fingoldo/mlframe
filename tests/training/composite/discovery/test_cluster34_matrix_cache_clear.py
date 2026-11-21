"""Audit 2026-11-20 finding 34-F29: MatrixKeyedCache owns its clear() instead of borrowing ContentMemo's.

The clear must empty the cache, leave it usable, and not depend on ContentMemo at all.
"""

import numpy as np

from mlframe.training.composite.discovery import _matrix_keyed_cache as mkc


def test_clear_empties_and_leaves_the_cache_usable():
    """After clear the cache has no entries, and a new put/get pair still works."""
    cache = mkc.MatrixKeyedCache(max_entries=4)
    m = np.zeros((3, 2))
    cache.put("k", m, 42)
    assert cache.get("k", m) == 42
    cache.clear()
    assert len(cache) == 0
    assert cache.get("k", m) is None
    cache.put("k2", m, 7)
    assert cache.get("k2", m) == 7


def test_clear_does_not_borrow_content_memo():
    """The class defines its own clear: it is not the ContentMemo function object."""
    assert "ContentMemo" not in vars(mkc)
    assert mkc.MatrixKeyedCache.clear.__qualname__.startswith("MatrixKeyedCache.")
