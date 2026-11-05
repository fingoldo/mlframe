"""A failed GPU odds-combine falls back to the CPU and says so at WARNING level.

The fallback used to log at DEBUG, so a run that silently lost its GPU path looked the same as a run that never had one.
"""

from __future__ import annotations

import logging

import numpy as np

from mlframe.calibration import _ktc_dispatch
from mlframe.calibration import ensembling


def _broken_cupy(*args, **kwargs):
    """Stand in for a GPU kernel that fails at runtime."""
    raise RuntimeError("simulated device fault")


def test_failed_cupy_odds_combine_falls_back_and_logs_a_warning(monkeypatch, caplog):
    """The unweighted dispatch logs a WARNING naming the failure, then still returns the CPU result."""
    monkeypatch.setattr(_ktc_dispatch, "choose_odds_combine_backend", lambda *a, **k: "cupy")
    monkeypatch.setattr(ensembling, "_odds_combine_cupy", _broken_cupy)
    p = np.random.default_rng(0).uniform(0.05, 0.95, size=(50, 3))
    with caplog.at_level(logging.WARNING, logger=ensembling.logger.name):
        out = ensembling._dispatch(p, 1e-6)
    assert out.shape == (50,)
    assert np.all(np.isfinite(out))
    assert any(r.levelno == logging.WARNING and "falling back to CPU" in r.getMessage() for r in caplog.records)


def test_failed_cupy_weighted_odds_combine_logs_a_warning(monkeypatch, caplog):
    """The weighted dispatch reports its GPU fallback at WARNING level too."""
    monkeypatch.setattr(_ktc_dispatch, "choose_odds_combine_backend", lambda *a, **k: "cupy")
    monkeypatch.setattr(ensembling, "_odds_combine_cupy_weighted", _broken_cupy)
    p = np.random.default_rng(1).uniform(0.05, 0.95, size=(40, 3))
    w = np.array([1.0, 2.0, 0.5])
    with caplog.at_level(logging.WARNING, logger=ensembling.logger.name):
        out = ensembling._dispatch_weighted(p, w, 1e-6)
    assert out.shape == (40,)
    assert any(r.levelno == logging.WARNING and "falling back to CPU" in r.getMessage() for r in caplog.records)
