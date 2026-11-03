"""Capability queries ask about the device the launch runs on, not a hard-coded device 0."""

from __future__ import annotations

import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters import _gpu_hw_launch as hw


def test_launch_device_id_is_the_current_device():
    """The id equals cupy's current device, and switching the device context switches it."""
    assert hw.launch_device_id() == int(cp.cuda.Device().id)
    with cp.cuda.Device(0):
        assert hw.launch_device_id() == 0


def test_capability_queries_ask_about_the_launch_device(monkeypatch):
    """The shared-memory budget probes pass the launch device's id to the capability query (a stand-in id shows it is not a constant 0)."""
    import pyutilz.system.gpu_dispatch as gd

    from mlframe.feature_selection.filters import _batch_pair_mi_cuda_shared_fused

    asked: list = []
    real = gd.gpu_capability_summary

    def recording(device_id=0, *a, **k):
        """Record the device the probe asks about, then answer for device 0 (the only one present)."""
        asked.append(device_id)
        return real(0, *a, **k)

    monkeypatch.setattr(gd, "gpu_capability_summary", recording)
    monkeypatch.setattr(hw, "launch_device_id", lambda: 7)
    _batch_pair_mi_cuda_shared_fused._opt_in_shared_mem_budget()
    assert asked == [7]
