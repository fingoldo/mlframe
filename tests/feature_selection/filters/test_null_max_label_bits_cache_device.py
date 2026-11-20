"""The packed-label cache of the family-wise null is keyed per device and rejects a block of another shape."""

from __future__ import annotations

import types

import numpy as np

from mlframe.feature_selection.filters import _null_max_perm_gpu as mod


def _fake_cp(device_id: int) -> types.SimpleNamespace:
    """A cupy stand-in whose current device has the given id."""
    return types.SimpleNamespace(cuda=types.SimpleNamespace(Device=lambda: types.SimpleNamespace(id=device_id)))


def test_label_bits_cache_key_differs_between_devices():
    """The same labels key on two devices must not share a cache entry."""
    k0 = mod._device_scoped_key(_fake_cp(0), ("labels", 1))
    k1 = mod._device_scoped_key(_fake_cp(1), ("labels", 1))
    assert k0 != k1
    assert mod._device_scoped_key(_fake_cp(0), None) is None
    mod._store_label_bits(k0, 0, np.zeros((4, 2), dtype=np.uint32))
    assert mod._cached_label_bits(k0, 0) is not None
    assert mod._cached_label_bits(k1, 0) is None


def test_cached_label_bits_of_another_shape_is_a_miss():
    """A cached block with another permutation count or word count is not reused."""
    bits = np.zeros((4, 2), dtype=np.uint32)
    assert mod._label_bits_if_shaped(bits, 4, 2) is bits
    assert mod._label_bits_if_shaped(bits, 5, 2) is None
    assert mod._label_bits_if_shaped(bits, 4, 3) is None
    assert mod._label_bits_if_shaped(None, 4, 2) is None
