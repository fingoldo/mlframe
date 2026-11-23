"""Cluster 27 regression: the GPU probe selects the smi backend only when nvidia-smi is on an absolute PATH entry (audit 2026-11-20, finding 27-F25)."""

import sys

import pytest

from mlframe.training import _gpu_state_probe as probe


@pytest.fixture
def _no_nvml_fresh(monkeypatch):
    """Force the nvml import to fail and reset the cached backend for an isolated resolve."""
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(probe, "_BACKEND", None)
    monkeypatch.setattr(probe, "_NVML", None)


def test_relative_only_path_gives_no_smi_backend(_no_nvml_fresh, monkeypatch):
    """When nvidia-smi is not on an absolute PATH entry the backend is 'none', matching the probe's own lookup (27-F25)."""

    def _refuse():
        """Simulate nvidia-smi not reachable through an absolute PATH entry."""
        raise FileNotFoundError("nvidia-smi is not on an absolute PATH entry")

    monkeypatch.setattr(probe, "_resolve_smi", _refuse)
    assert probe._resolve_backend() == "none"


def test_absolute_path_gives_smi_backend(_no_nvml_fresh, monkeypatch):
    """Control: when the resolver finds nvidia-smi the backend is 'smi' (27-F25)."""
    monkeypatch.setattr(probe, "_resolve_smi", lambda: "/usr/bin/nvidia-smi")
    assert probe._resolve_backend() == "smi"
