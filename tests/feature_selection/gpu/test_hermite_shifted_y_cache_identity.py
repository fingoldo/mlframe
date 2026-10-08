"""The shifted-y memo must serve an entry only to the very array it was built from, never to a new array that reuses a freed array's id."""

from __future__ import annotations

import gc

import numpy as np

from mlframe.feature_selection.filters import _hermite_fe_mi as hfmi


def _force_same_id(monkeypatch):
    """Make every array look like it lives at one address, the way an allocator hands a freed address back."""
    monkeypatch.setattr(hfmi, "id", lambda _obj: 1234, raising=False)


def test_reused_id_of_a_freed_array_does_not_serve_the_stale_shift(monkeypatch):
    """Array A is cached then freed; array B of the same shape at the same id gets its own shifted values."""
    _force_same_id(monkeypatch)
    hfmi._SHIFTED_Y_CACHE.clear()
    a = np.array([5, 6, 7, 5], dtype=np.int64)
    assert np.array_equal(hfmi._shifted_y_cached(a, 5), [0, 1, 2, 0])
    del a
    gc.collect()
    b = np.array([9, 9, 10, 11], dtype=np.int64)
    shifted_b = hfmi._shifted_y_cached(b, 5)
    assert np.array_equal(shifted_b, [4, 4, 5, 6])


def test_reused_id_while_the_first_array_is_alive_is_a_miss(monkeypatch):
    """A second live array colliding on id and shape is a different object: it must not read the first array's entry."""
    _force_same_id(monkeypatch)
    hfmi._SHIFTED_Y_CACHE.clear()
    a = np.array([3, 4, 5], dtype=np.int64)
    b = np.array([7, 7, 8], dtype=np.int64)
    assert np.array_equal(hfmi._shifted_y_cached(a, 3), [0, 1, 2])
    assert np.array_equal(hfmi._shifted_y_cached(b, 3), [4, 4, 5])


def test_same_array_and_shift_is_a_cache_hit_returning_the_same_object():
    """The memo still works for its purpose: the same live array and y_min return the one stored shifted vector."""
    hfmi._SHIFTED_Y_CACHE.clear()
    a = np.array([2, 3, 4], dtype=np.int64)
    first = hfmi._shifted_y_cached(a, 2)
    assert hfmi._shifted_y_cached(a, 2) is first
    assert len(hfmi._SHIFTED_Y_CACHE) == 1
