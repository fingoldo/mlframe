"""A repeated GPU fallback in the odds-combine dispatcher is throttled per call site: a few warnings, then one suppression notice."""

import logging

import numpy as np

import mlframe.calibration.ensembling as ens


def test_repeated_cupy_failures_warn_once(monkeypatch, caplog):
    """Eight dispatches that all fall back from cupy log five warnings and then one suppression notice."""
    monkeypatch.setattr("mlframe.calibration._ktc_dispatch.choose_odds_combine_backend", lambda *a, **k: "cupy")

    def failing(p, clip):
        """Simulate a GPU failure."""
        raise RuntimeError("device out of memory")

    monkeypatch.setattr(ens, "_odds_combine_cupy", failing)
    p = np.clip(np.random.default_rng(0).uniform(0.05, 0.95, size=(50, 3)), 0.01, 0.99)
    with caplog.at_level(logging.WARNING):
        for _ in range(8):
            ens._dispatch(p, 1e-6)
    falls = [r for r in caplog.records if "falling back to CPU" in r.getMessage()]
    notices = [r for r in caplog.records if "suppressed" in r.getMessage()]
    assert len(falls) == 5
    assert len(notices) == 1
