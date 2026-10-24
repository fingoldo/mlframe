"""Load-aware thread cap for HistGradientBoosting fits. The CPU-load probe is substituted; no real load is generated."""

from __future__ import annotations

import pytest
import threadpoolctl

from mlframe.training import _hgb_thread_cap as cap


class HistGradientBoostingClassifier:
    """Name-only stand-in: the cap keys on the class name, so no sklearn fit is needed."""


_FakeHGBClassifier = HistGradientBoostingClassifier


class _Other:
    """Non-HGB stand-in that must never trigger a probe."""


class _LimitRecorder:
    """Replaces threadpoolctl.threadpool_limits and records each entry and exit."""

    calls: list = []
    exits: int = 0

    def __init__(self, limits=None, **kwargs):
        """Store the requested limit."""
        self.limits = limits

    def __enter__(self):
        """Record the requested limit on entry."""
        _LimitRecorder.calls.append(self.limits)
        return self

    def __exit__(self, *exc):
        """Count the scope exit and let exceptions propagate."""
        _LimitRecorder.exits += 1
        return False


@pytest.fixture
def recorder(monkeypatch):
    """Patch threadpool_limits with the recorder and reset its counters."""
    _LimitRecorder.calls = []
    _LimitRecorder.exits = 0
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", _LimitRecorder)
    return _LimitRecorder


@pytest.fixture
def cap_enabled(monkeypatch):
    """Turn the cap on for the test; it ships disabled by default."""
    monkeypatch.setattr(cap, "HGB_THREAD_CAP_ENABLED", True)


def test_shipped_default_is_disabled():
    """The cap ships off until an A/B on a controlled host shows a win."""
    assert cap.HGB_THREAD_CAP_ENABLED is False


@pytest.fixture
def probe(monkeypatch):
    """Patch the CPU-load probe; the test sets state["value"] to the load it should report."""
    state = {"value": 0.0, "calls": 0}

    def fake_probe(interval):
        """Return the scripted load and check the probe interval."""
        state["calls"] += 1
        assert interval == cap.HGB_THREAD_CAP_PROBE_SECONDS
        return state["value"]

    monkeypatch.setattr(cap, "_probe_cpu_percent", fake_probe)
    return state


def test_high_load_caps_hgb_threads(cap_enabled, probe, recorder):
    """Load above the threshold limits HGB to 4 threads and reports the cap as active."""
    probe["value"] = 90.0
    with cap.hgb_fit_thread_cap(_FakeHGBClassifier()) as active:
        assert active is True
    assert recorder.calls == [4]


def test_low_load_keeps_default_threads(probe, recorder):
    """Load below the threshold leaves the library thread count untouched."""
    probe["value"] = 5.0
    with cap.hgb_fit_thread_cap(_FakeHGBClassifier()) as active:
        assert active is False
    assert recorder.calls == []


def test_threshold_is_strict_greater_than(probe, recorder):
    """Load exactly at the threshold does not cap, since the comparison is strict."""
    probe["value"] = cap.HGB_THREAD_CAP_LOAD_THRESHOLD_PCT
    with cap.hgb_fit_thread_cap(_FakeHGBClassifier()) as active:
        assert active is False
    assert recorder.calls == []


def test_non_hgb_model_never_probes_or_caps(probe, recorder):
    """A non-HGB model neither probes the load nor caps threads, even at 100% load."""
    probe["value"] = 100.0
    with cap.hgb_fit_thread_cap(_Other()) as active:
        assert active is False
    assert probe["calls"] == 0
    assert recorder.calls == []


def test_disabled_never_probes_or_caps(probe, recorder):
    """enabled=False skips both the probe and the cap."""
    probe["value"] = 100.0
    with cap.hgb_fit_thread_cap(_FakeHGBClassifier(), enabled=False) as active:
        assert active is False
    assert probe["calls"] == 0
    assert recorder.calls == []


def test_probe_failure_keeps_default_threads(monkeypatch, recorder):
    """A failing load probe keeps the default thread count instead of raising."""

    def broken_probe(interval):
        """Simulate a psutil failure."""
        raise RuntimeError("psutil unavailable")

    monkeypatch.setattr(cap, "_probe_cpu_percent", broken_probe)
    with cap.hgb_fit_thread_cap(_FakeHGBClassifier()) as active:
        assert active is False
    assert recorder.calls == []


def test_probe_runs_once_per_context_and_cap_scope_exits(cap_enabled, probe, recorder):
    """One probe per context entry, and the limit scope is exited on normal completion."""
    probe["value"] = 90.0
    with cap.hgb_fit_thread_cap(_FakeHGBClassifier()):
        pass
    assert probe["calls"] == 1
    assert recorder.calls == [4]
    assert recorder.exits == 1


def test_cap_scope_exits_when_fit_raises(cap_enabled, probe, recorder):
    """The limit scope is exited even when the fit raises, and the exception propagates."""
    probe["value"] = 90.0
    with pytest.raises(RuntimeError, match="boom"):
        with cap.hgb_fit_thread_cap(_FakeHGBClassifier()):
            raise RuntimeError("boom")
    assert recorder.calls == [4]
    assert recorder.exits == 1


def test_module_switch_read_at_call_time_disables_cap(monkeypatch, probe, recorder):
    """Flipping HGB_THREAD_CAP_ENABLED at runtime disables the cap for later calls."""
    monkeypatch.setattr(cap, "HGB_THREAD_CAP_ENABLED", False)
    probe["value"] = 100.0
    with cap.hgb_fit_thread_cap(_FakeHGBClassifier()) as active:
        assert active is False
    assert probe["calls"] == 0
    assert recorder.calls == []


def test_threshold_read_at_call_time(monkeypatch, cap_enabled, probe, recorder):
    """The threshold is read at call time, so a runtime change takes effect immediately."""
    monkeypatch.setattr(cap, "HGB_THREAD_CAP_LOAD_THRESHOLD_PCT", 80.0)
    probe["value"] = 90.0
    with cap.hgb_fit_thread_cap(_FakeHGBClassifier()) as active:
        assert active is True
    assert recorder.calls == [4]
