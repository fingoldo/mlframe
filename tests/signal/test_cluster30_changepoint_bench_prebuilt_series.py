"""Cluster 30 regression: the changepoint benchmark runs its timed call on a prebuilt series (audit 2026-11-20, finding 30-F5)."""

import numpy as np

from mlframe.signal._benchmarks import bench_changepoint_detection as bench


def test_run_uses_a_passed_series_without_regenerating(monkeypatch):
    """With y supplied the timed path never calls the generator (30-F5)."""

    def _no_generate(*a, **k):
        """Fail if the generator is reached on the timed path."""
        raise AssertionError("series regenerated inside the timed call")

    monkeypatch.setattr(bench, "_make_series", _no_generate)
    y = np.random.default_rng(0).normal(size=300)
    monkeypatch.setattr(bench, "detect_regime_changepoints", lambda *a, **k: None)
    bench._run(300, 3, "njit", y=y)
