"""Cluster 41 regression: a failing collapse sensor is logged at WARNING, not debug (audit 2026-11-20, finding 41-F27)."""

import logging

import numpy as np

from mlframe.training.reporting._reporting_regression import _sensors

_MODULE = "mlframe.training.reporting._reporting_regression._sensors"


def test_sensor_failure_is_logged_at_warning(monkeypatch, caplog):
    """A collapse trip whose recording raises is visible at WARNING, so a broken sensor is not mistaken for a healthy run (41-F27)."""

    def _boom(*_args, **_kwargs):
        """Stand in for the trip recorder failing at runtime."""
        raise RuntimeError("trip recorder broke")

    monkeypatch.setattr(_sensors, "record_sensor_trip", _boom)
    rng = np.random.default_rng(0)
    targets = rng.normal(size=200)
    preds = np.zeros(200)  # constant predictions against a spread target: collapse branch
    with caplog.at_level(logging.WARNING, logger=_MODULE):
        _sensors.run_collapse_sensor(preds, targets, -1.0, model_name="m")
    assert any(r.levelno == logging.WARNING and "collapse-sensor probe failed" in r.getMessage() for r in caplog.records)
