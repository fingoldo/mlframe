"""Cluster 27 regression: importing the profiling module does not change CUDA visibility in the process (audit 2026-11-20, finding 27-F24)."""

import importlib.util
import os
from pathlib import Path

import mlframe.training._profile_fs_suite_integration as mod


def test_import_does_not_set_cuda_visible_devices(monkeypatch):
    """Executing the module (as an import would) leaves CUDA_VISIBLE_DEVICES untouched (27-F24)."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    spec = importlib.util.spec_from_file_location("_profile_probe_import", Path(mod.__file__))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    leaked = "CUDA_VISIBLE_DEVICES" in os.environ
    assert not leaked
