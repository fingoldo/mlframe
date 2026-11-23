"""Cluster 39 regression: a CUDA failure while sizing the provider cache falls back to the default, not an exception (audit 2026-11-20, finding 39-F39)."""

import logging
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from mlframe.training.feature_handling import registry as registry_mod


def test_cuda_mem_info_failure_falls_back_to_two(monkeypatch, caplog):
    """A RuntimeError from torch.cuda.mem_get_info returns the default 2 and logs a WARNING (39-F39)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def _broken():
        """Simulate a broken CUDA context."""
        raise RuntimeError("CUDA error: context destroyed")

    monkeypatch.setattr(torch.cuda, "mem_get_info", _broken)
    cfg = SimpleNamespace(keep_n_providers="auto")
    with caplog.at_level(logging.WARNING, logger=registry_mod.logger.name):
        assert registry_mod._resolve_keep_n(cfg) == 2
    assert any("fell back to 2" in r.getMessage() for r in caplog.records)
