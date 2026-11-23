"""Cluster 22 regression: a cupy measurement that fails is a WARNING, not a debug line (audit 2026-11-20, finding 22-F42)."""

import logging
import sys
import types

import pytest

from mlframe.calibration import _ktc_dispatch as kd


def test_cupy_measurement_failure_is_logged_as_warning(monkeypatch, caplog):
    """With cupy importable but its transfer raising, the tuner logs a WARNING that cupy is excluded (22-F42)."""
    pytest.importorskip("numba")
    fake_cp = types.ModuleType("cupy")

    def _asarray(_x):
        """Simulate a broken device transfer."""
        raise RuntimeError("device lost")

    fake_cp.asarray = _asarray
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    with caplog.at_level(logging.WARNING, logger=kd.logger.name):
        kd._make_tuner(64, 3)()
    assert any(r.levelno == logging.WARNING and "cupy measurement failed" in r.getMessage() for r in caplog.records)
