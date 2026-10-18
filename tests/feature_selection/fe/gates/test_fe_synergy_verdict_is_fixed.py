"""The auto exhaustive-sweep verdict is a function of the problem size, not of a measured throughput.

A measured pairs-per-second (cold cache, machine load, GPU or CPU host) changed the verdict, and with it the selected features. The verdict must hold for any measured rate.
"""

import pytest

import mlframe.feature_selection.filters._fe_synergy_exhaustive as ex


class _Knobs:
    """Minimal estimator stand-in carrying the synergy knobs the decision reads."""

    def __init__(self, mode, budget):
        self.fe_synergy_exhaustive = mode
        self.fe_synergy_exhaustive_max_seconds = budget
        self.max_runtime_mins = None


@pytest.mark.parametrize("rate", [1.0, 5.0e4, 1.0e9])
@pytest.mark.parametrize("n_raw", [60, 2000])
def test_verdict_does_not_follow_the_measured_rate(monkeypatch, rate, n_raw):
    """The same problem gets the same verdict whatever throughput the cache reports."""
    monkeypatch.setattr(ex, "_CUDA_AVAIL", True, raising=False)
    monkeypatch.setattr(ex, "warm_exhaustive_throughput_cache", lambda: None)
    monkeypatch.setattr(ex, "warm_exhaustive_cpu_throughput_cache", lambda: None)
    monkeypatch.setattr(ex, "measured_pairs_per_second", lambda n_samples, n_pairs: (1.0, "reference"))
    monkeypatch.setattr(ex, "measured_cpu_pairs_per_second", lambda n_samples, n_pairs: (1.0, "reference"))
    base, _ = ex.decide_exhaustive_sweep(_Knobs("auto", 300.0), n_samples=20_000, n_raw=n_raw, verbose=0)
    monkeypatch.setattr(ex, "measured_pairs_per_second", lambda n_samples, n_pairs: (rate, "test"))
    monkeypatch.setattr(ex, "measured_cpu_pairs_per_second", lambda n_samples, n_pairs: (rate, "test"))
    use, _ = ex.decide_exhaustive_sweep(_Knobs("auto", 300.0), n_samples=20_000, n_raw=n_raw, verbose=0)
    assert use == base
