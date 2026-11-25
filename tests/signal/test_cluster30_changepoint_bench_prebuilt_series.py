"""Cluster 30 regression: the changepoint benchmark runs its timed call on a prebuilt series (audit 2026-11-20, finding 30-F5)."""

import numpy as np

from mlframe.signal._benchmarks import bench_changepoint_detection as bench


def test_run_uses_a_passed_series_without_regenerating(monkeypatch):
    """With y supplied the timed path never calls the generator (30-F5)."""

    generated = []
    detected = []

    def _record_generate(*a, **k):
        """Record any call to the generator on the timed path."""
        generated.append(True)
        return np.zeros(300)

    def _record_detect(*a, **k):
        """Record that the timed call reached the detector, so the run is not vacuous."""
        detected.append(True)

    monkeypatch.setattr(bench, "_make_series", _record_generate)
    monkeypatch.setattr(bench, "detect_regime_changepoints", _record_detect)
    y = np.random.default_rng(0).normal(size=300)
    bench._run(300, 3, "njit", y=y)
    assert generated == []
    assert detected == [True]
