"""``gpu_globally_disabled`` reads ``CUDA_VISIBLE_DEVICES`` as CUDA does: no device when the list is empty or starts with an invalid entry such as ``-1``."""

from __future__ import annotations

import pytest

from mlframe.feature_selection.filters._gpu_policy import gpu_globally_disabled


@pytest.mark.parametrize("value,disabled", [("", True), ("-1", True), (" -1 ", True), ("-1,0", True), ("none", True), ("0", False), ("1,0", False), ("GPU-8a1b", False), ("MIG-3c", False)])
def test_visible_devices(monkeypatch, value, disabled):
    """Each value against the device list CUDA would expose."""
    monkeypatch.delenv("MLFRAME_DISABLE_GPU", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
    assert gpu_globally_disabled() is disabled


def test_unset_and_flag(monkeypatch):
    """Unset leaves the GPU on; ``MLFRAME_DISABLE_GPU=1`` turns it off whatever the device list."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("MLFRAME_DISABLE_GPU", raising=False)
    assert gpu_globally_disabled() is False
    monkeypatch.setenv("MLFRAME_DISABLE_GPU", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert gpu_globally_disabled() is True


def test_device_gates_follow_the_policy(monkeypatch):
    """The modular-codes and tied-bootstrap device gates say no on a run without a GPU, even when their own switch forces the device."""
    from mlframe.feature_selection.filters._bootstrap_rank_mi_tied_gpu import use_device_tied
    from mlframe.feature_selection.filters._modular_codes_gpu import _use_device

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    monkeypatch.setenv("MLFRAME_MODULAR_CODES_DEVICE", "1")
    monkeypatch.setenv("MLFRAME_BOOTSTRAP_RANK_TIED_DEVICE", "1")
    assert _use_device(10**9) is False
    assert use_device_tied(10**6) is False
