"""The packed permuted labels are computed once per label array and never returned for another array."""

import numpy as np

from mlframe.feature_selection.filters import _honest_null_device as H


def _labels(seed, b=6, n=500):
    """A (permutations, n) int8 0/1 label matrix."""
    return (np.random.default_rng(seed).random((b, n)) < 0.4).astype(np.int8)


def test_same_array_is_packed_once(monkeypatch):
    """Two calls with the same array object run the packing once and return the same bits."""
    calls = []
    real = H._pack_bits
    monkeypatch.setattr(H, "_pack_bits", lambda y: calls.append(1) or real(y))
    monkeypatch.setattr(H, "_PACKED_LAST", None)
    y = _labels(0)
    a, b = H._pack_bits_cached(y), H._pack_bits_cached(y)
    assert a is b and len(calls) == 1
    np.testing.assert_array_equal(a, real(y))


def test_another_array_gets_its_own_bits():
    """A different label array (even of the same shape) never receives the cached bits of the previous one."""
    y1, y2 = _labels(1), _labels(2)
    p1 = H._pack_bits_cached(y1).copy()
    p2 = H._pack_bits_cached(y2)
    assert not np.array_equal(p1, p2)
    np.testing.assert_array_equal(p2, H._pack_bits(y2))


def test_recycled_array_object_is_not_confused_with_the_old_one():
    """After the first array is deleted and a new one is built, the cache holds only a dead weak reference and repacks."""
    y1 = _labels(3)
    H._pack_bits_cached(y1)
    del y1
    y2 = _labels(4)
    np.testing.assert_array_equal(H._pack_bits_cached(y2), H._pack_bits(y2))
