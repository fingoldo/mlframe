"""Cluster 26 regression: a failing GPU DTW path falls back to CPU with a WARNING, not a debug line (audit 2026-11-20, finding 26-F29)."""

import logging

import numpy as np

from mlframe.signal import dtw


def test_cupy_failure_falls_back_with_warning(monkeypatch, caplog):
    """A cupy kernel that raises gives the CPU answer and a WARNING saying the path is slower (26-F29)."""
    monkeypatch.setattr(dtw, "_HAS_CUPY", True)
    monkeypatch.setattr(dtw._DTW_SPEC, "choose", lambda **k: "cupy")

    def _boom(*a, **k):
        """Simulate a device failure."""
        raise RuntimeError("device lost")

    monkeypatch.setattr(dtw, "dtw_cupy", _boom)
    x = np.sin(np.linspace(0, 3, 30))
    y = np.cos(np.linspace(0, 3, 30))
    with caplog.at_level(logging.WARNING, logger=dtw.logger.name):
        out = dtw.dtw_dispatch(x, y)
    distance, _path = out  # dtw_cpu returns (distance, path)
    assert np.isfinite(distance)
    assert any(r.levelno == logging.WARNING and "dtw_cupy failed" in r.getMessage() for r in caplog.records)
