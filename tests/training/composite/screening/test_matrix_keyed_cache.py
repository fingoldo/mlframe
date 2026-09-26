"""MatrixKeyedCache: the shared LRU behind the LightGBM and ridge shared-fold caches."""

from __future__ import annotations

import gc
import pickle

import numpy as np

from mlframe.training.composite.discovery._matrix_keyed_cache import MatrixKeyedCache


def test_a_hit_needs_the_same_matrix_object():
    """An equal but different matrix is a miss: entries are keyed on the object, not its contents."""
    cache = MatrixKeyedCache(4)
    x = np.ones(3)
    cache.put("k", x, "value")
    assert cache.get("k", x) == "value"
    assert cache.get("k", np.ones(3)) is None  # an equal but different matrix is a miss


def test_entries_of_a_freed_matrix_are_dropped():
    """Once the matrix is garbage-collected, pruning removes its entries."""
    cache = MatrixKeyedCache(4)
    x = np.ones(3)
    cache.put("k", x, "value")
    del x
    gc.collect()
    cache.prune_dead()
    assert len(cache) == 0


def test_the_oldest_entry_goes_over_the_cap():
    """Over capacity, the oldest entry is evicted."""
    cache = MatrixKeyedCache(2)
    xs = [np.ones(2) for _ in range(3)]
    for i, x in enumerate(xs):
        cache.put(i, x, i)
    assert len(cache) == 2 and cache.get(0, xs[0]) is None and cache.get(2, xs[2]) == 2


def test_it_pickles_as_an_empty_cache_of_the_same_size():
    """A pickled cache comes back empty with the same capacity, and still works."""
    cache = MatrixKeyedCache(5)
    x = np.ones(2)
    cache.put("k", x, 1)
    restored = pickle.loads(pickle.dumps(cache))  # nosec B301 - round-trips an object this test just pickled
    assert restored.max_entries == 5 and len(restored) == 0
    restored.put("k", x, 2)
    assert restored.get("k", x) == 2
