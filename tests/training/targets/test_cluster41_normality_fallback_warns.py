"""Cluster 41 regression: a failing normality test is reported at WARNING, not debug (audit 2026-11-20, finding 41-F13)."""

import logging

import numpy as np
import pytest

pyutilz_normality = pytest.importorskip("pyutilz.stats.normality")
from mlframe.training.targets.regression_residual_audit import audit_residuals


def test_normality_failure_is_logged_at_warning(monkeypatch, caplog):
    """When the formal normality test raises, the fallback is visible at WARNING (41-F13)."""

    def _boom(_residuals):
        """Stand in for a normality test that fails at runtime."""
        raise RuntimeError("normality backend broke")

    monkeypatch.setattr(pyutilz_normality, "normality_verdict", _boom)
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    with caplog.at_level(logging.WARNING, logger="mlframe.training.targets.regression_residual_audit"):
        audit_residuals(y, y + 0.1 * rng.normal(size=200), sample_size=None)
    assert any(r.levelno == logging.WARNING and "normality_verdict" in r.getMessage() for r in caplog.records)
