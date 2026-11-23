"""Cluster 11 regression: a non-positive baseline error is logged as the reason for a NaN relative improvement (audit 2026-11-20, finding 11-F18)."""

import logging

import numpy as np
import pytest

from mlframe.feature_selection._benchmarks.fe_operator_factory.common import downstream
from mlframe.feature_selection._benchmarks.fe_operator_factory.common.downstream import rel_improvement


def test_zero_baseline_returns_nan_and_logs_why(caplog):
    """A perfect baseline (error 0) still gives NaN, and a WARNING names the undefined improvement (11-F18)."""
    with caplog.at_level(logging.WARNING, logger=downstream.logger.name):
        out = rel_improvement(0.5, 0.0)
    assert np.isnan(out)
    assert any(r.levelno == logging.WARNING and "not positive" in r.getMessage() for r in caplog.records)


def test_positive_baseline_is_silent(caplog):
    """Control: a positive baseline gives the ratio and no warning (11-F18)."""
    with caplog.at_level(logging.WARNING, logger=downstream.logger.name):
        assert rel_improvement(0.8, 1.0) == pytest.approx(0.2)
    assert not any("not positive" in r.getMessage() for r in caplog.records)
