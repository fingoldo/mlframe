"""The cached bin-boundary array is kept per GPU device, so a device array is never handed to another device.

The cache key had the array size but no device id, so after a switch of the current device the first device's array came back. Using it on
the other device fails or makes an implicit peer copy.
"""

from __future__ import annotations

import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _gpu_resident_rank_bin as rb


def test_each_device_gets_its_own_boundary_array(monkeypatch):
    """Two different current devices with the same (n, n_bins) get two distinct arrays, each built for its own device."""
    rb._BIN_BOUNDARIES_CACHE.clear()
    device = {"id": 0}
    monkeypatch.setattr(cp.cuda, "get_device_id", lambda: device["id"])
    first = rb._bin_boundaries_dev(cp, 1000, 8)
    device["id"] = 1
    second = rb._bin_boundaries_dev(cp, 1000, 8)
    assert first is not second
    assert len(rb._BIN_BOUNDARIES_CACHE) == 2


def test_same_device_reuses_the_cached_array(monkeypatch):
    """Within one device the array is still reused, as before."""
    rb._BIN_BOUNDARIES_CACHE.clear()
    monkeypatch.setattr(cp.cuda, "get_device_id", lambda: 0)
    a = rb._bin_boundaries_dev(cp, 1000, 8)
    b = rb._bin_boundaries_dev(cp, 1000, 8)
    assert a is b
