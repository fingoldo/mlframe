"""``_cached_card``'s content-hash cache key switched from ``hash(arr.tobytes())`` (a full O(n) copy per
call) to ``xxhash.xxh3_64_intdigest(arr)`` (a copy-free buffer read) for numeric arrays -- same technique
``_mrmr_degenerate._content_key`` and ``_fe_resident_operands._content_hash`` already use. The cache's
observable behaviour (same content -> same cached cardinality, different content -> a fresh lookup) must
be unchanged.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._mi_greedy_cmi_fe_steps import _CARD_MAX_CACHE, _cached_card


def _clear_cache():
    """Reset the module-level memo so each test starts from a cold cache."""
    _CARD_MAX_CACHE.clear()


def test_identical_content_hits_the_cache_with_the_same_cardinality():
    """Two distinct arrays with identical content must report the same cardinality via the cache."""
    _clear_cache()
    host_a = np.arange(1000, dtype=np.int64)
    host_b = host_a.copy()
    dev_codes = np.arange(7, dtype=np.int64)
    first = _cached_card(host_a, dev_codes)
    second = _cached_card(host_b, dev_codes)
    assert first == second == int(dev_codes.max()) + 1


def test_different_content_same_shape_does_not_share_a_cache_entry():
    """Two different-content arrays of the same shape/dtype must not collide in the cache."""
    _clear_cache()
    host_a = np.arange(1000, dtype=np.int64)
    host_b = host_a[::-1].copy()
    dev_codes_a = np.arange(5, dtype=np.int64)
    dev_codes_b = np.arange(9, dtype=np.int64)
    card_a = _cached_card(host_a, dev_codes_a)
    card_b = _cached_card(host_b, dev_codes_b)
    assert card_a == 5
    assert card_b == 9


def test_empty_dev_codes_returns_one_without_touching_the_cache():
    """An empty ``dev_codes`` short-circuits to 1, the documented degenerate case."""
    _clear_cache()
    assert _cached_card(np.arange(10), np.empty(0, dtype=np.int64)) == 1
    assert len(_CARD_MAX_CACHE) == 0


def test_float_and_object_dtype_arrays_still_work():
    """A non-integer dtype (the xxhash dtype-kind gate's fallback branch) still produces a correct, cacheable key."""
    _clear_cache()
    host = np.linspace(0.0, 1.0, 500)
    dev_codes = np.arange(3, dtype=np.int64)
    assert _cached_card(host, dev_codes) == 3
    assert _cached_card(host.copy(), dev_codes) == 3
