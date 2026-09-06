"""A persistent GPU fallback must keep saying so, not fall silent after the first message.

`_shap_proxy_subsetrank` latched `_fallback_logged` to True on the first GPU-unavailable fallback, so the
warning was emitted once per process. Nothing about the DISPATCH was cached -- `gpu_available()` is consulted
on every call -- so this never degraded behaviour, only what an operator could see: a run that lost the
device after the first message emitted nothing further, and the log understated how long the CPU kernel had
been carrying the fit.

Rate-limited by time instead, so a long fallback re-reports periodically while a burst of calls still does
not spam.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_subsetrank as sr


class _FakeClock:
    """A monotonic clock the test advances by hand."""

    def __init__(self, start: float = 1000.0):
        """Start at an arbitrary non-zero point."""
        self.t = start

    def monotonic(self) -> float:
        """The current fake time."""
        return self.t


@pytest.fixture(autouse=True)
def _clock(monkeypatch):
    """Give the module a clock this test controls, and reset the limiter between tests."""
    clock = _FakeClock()
    monkeypatch.setattr(sr, "time", clock)
    monkeypatch.setattr(sr, "_last_fallback_log_ts", 0.0)
    return clock


def test_a_burst_of_fallbacks_logs_once(_clock):
    """The behaviour worth keeping from the latch: repeated calls in quick succession do not spam."""
    assert sr._should_log_fallback() is True, "the first fallback in a run must report"
    for _ in range(50):
        assert sr._should_log_fallback() is False, "a burst of fallbacks re-reported inside the rate-limit window"


def test_a_fallback_that_persists_reports_again_later(_clock):
    """The bug: after the first message, a run that stayed on the CPU kernel said nothing more, ever."""
    assert sr._should_log_fallback() is True
    _clock.t += sr._FALLBACK_LOG_INTERVAL_S * 1.1
    assert sr._should_log_fallback() is True, "a fallback still in effect a full interval later never re-reported"


def test_the_window_restarts_from_the_last_message(_clock):
    """Two intervals of silence must not bank up two messages."""
    assert sr._should_log_fallback() is True
    _clock.t += sr._FALLBACK_LOG_INTERVAL_S * 1.1
    assert sr._should_log_fallback() is True
    assert sr._should_log_fallback() is False, "the limiter emitted twice in a row after one long gap"


def test_the_dispatch_itself_was_never_gated_by_the_flag(caplog, monkeypatch):
    """What made this diagnostic-only: the GPU is re-attempted every call regardless of what was logged.

    Pinned so a future change cannot quietly turn the log limiter into an availability cache.
    """
    rng = np.random.default_rng(0)
    phi = rng.normal(size=(60, 5))
    base = np.full(60, 0.1)
    y = (rng.random(60) > 0.5).astype(float)

    probes = []

    def _counting_gpu_available():
        """Record each availability probe."""
        probes.append(1)
        return False

    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_gpu as gpu

    # The route also requires the subset count to clear the tuned crossover (250k by default), which this
    # deliberately small fixture never does -- without dropping it the GPU branch is not entered at all and
    # the probe count below is zero for a reason that has nothing to do with caching.
    monkeypatch.setattr(sr, "_gpu_min_subsets", lambda: 0)
    monkeypatch.setattr(gpu, "gpu_available", _counting_gpu_available)

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            sr.brute_force_top_n_dispatch(phi, base, y, classification=True, metric="brier", max_card=3, top_n=5, prefer_gpu=True)

    assert len(probes) == 3, f"the GPU was probed {len(probes)} times across 3 calls; availability is being cached, which it never was"
    assert len([r for r in caplog.records if "using CPU kernel" in r.getMessage()]) == 1, "the burst of fallbacks should log once, not once per call"
