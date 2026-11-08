"""A measured odds-combine backend choice covers the measured shape and its near neighbours, not every smaller shape."""

from __future__ import annotations

from pyutilz.performance.kernel_tuning.cache.region_matching import _region_matches

from mlframe.calibration import _ktc_dispatch as kd


def _region(monkeypatch, n=100_000, k=20):
    """The region the tuner emits when the GPU measures fastest."""
    monkeypatch.setattr(kd, "measure_backend", lambda fn: 1.0 if "gpu" in getattr(fn, "__name__", "") else 5.0)
    return kd._make_tuner(n, k)()[0]


def test_region_has_a_lower_row_bound(monkeypatch):
    """The region carries n_samples_min so that tiny calls do not reuse a large-shape GPU win."""
    region = _region(monkeypatch)
    assert region["n_samples_max"] == 100_000
    assert region["n_samples_min"] == 100_000 // 4


def test_tiny_query_misses_and_near_query_hits(monkeypatch):
    """A 100-row query does not match the 100k-row region; a 90k-row query does."""
    region = _region(monkeypatch)
    assert not _region_matches(region, {"n_samples": 100, "n_members": 5})
    assert _region_matches(region, {"n_samples": 90_000, "n_members": 10})
