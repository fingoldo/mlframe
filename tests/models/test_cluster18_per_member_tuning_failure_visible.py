"""Cluster 18 regression: a failed per-member auto-tune is logged as a WARNING, not at DEBUG (audit 2026-11-20, 18-F13)."""

import logging

import pytest

from mlframe.models.ensembling import per_member_tuning as pmt


def test_failed_sweep_logs_a_warning(monkeypatch, caplog):
    """When the sweep raises, ensure_per_member_tuning swallows it but a WARNING names the error (18-F13)."""
    pytest.importorskip("pyutilz.performance.kernel_tuning.cache")

    def boom(**kwargs):
        """Fail like a broken benchmark."""
        raise RuntimeError("sweep exploded")

    monkeypatch.setattr(pmt, "run_per_member_sweep", boom)
    with caplog.at_level(logging.WARNING, logger=pmt.logger.name):
        pmt.ensure_per_member_tuning(force=True, observed_elements=1000, repeats=1)
    assert any("auto-tune failed" in r.getMessage() and "sweep exploded" in r.getMessage() for r in caplog.records)
