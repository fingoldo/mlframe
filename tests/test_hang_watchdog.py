"""Unit tests for tests/_hang_watchdog.py -- the per-test stall-diagnostics timer."""

from tests import _hang_watchdog


def test_arm_schedules_dump_traceback_later_with_configured_seconds(monkeypatch):
    """arm() forwards HANG_WATCHDOG_SECONDS to faulthandler with exit=False."""
    monkeypatch.setattr(_hang_watchdog, "HANG_WATCHDOG_SECONDS", 42)
    calls = []
    monkeypatch.setattr(_hang_watchdog.faulthandler, "dump_traceback_later", lambda seconds, **kwargs: calls.append((seconds, kwargs.get("exit"))))
    _hang_watchdog.arm("some::test_id")
    assert calls == [(42, False)]


def test_disarm_cancels_the_timer(monkeypatch):
    """disarm() calls faulthandler.cancel_dump_traceback_later()."""
    monkeypatch.setattr(_hang_watchdog, "HANG_WATCHDOG_SECONDS", 42)
    cancelled = []
    monkeypatch.setattr(_hang_watchdog.faulthandler, "cancel_dump_traceback_later", lambda: cancelled.append(True))
    _hang_watchdog.disarm()
    assert cancelled == [True]


def test_zero_seconds_disables_arm_and_disarm(monkeypatch):
    """HANG_WATCHDOG_SECONDS=0 is the documented opt-out; neither call reaches faulthandler."""
    monkeypatch.setattr(_hang_watchdog, "HANG_WATCHDOG_SECONDS", 0)
    armed = []
    disarmed = []
    monkeypatch.setattr(_hang_watchdog.faulthandler, "dump_traceback_later", lambda *a, **k: armed.append(True))
    monkeypatch.setattr(_hang_watchdog.faulthandler, "cancel_dump_traceback_later", lambda: disarmed.append(True))
    _hang_watchdog.arm("some::test_id")
    _hang_watchdog.disarm()
    assert armed == []
    assert disarmed == []


def test_default_seconds_is_positive_and_below_the_per_test_timeout():
    """The real env-derived default must fire before pytest-timeout's own 900s per-test cap."""
    # --timeout=900 is the suite's per-test cap (pyproject.toml addopts); the watchdog must fire
    # BEFORE that so a genuine hang is diagnosed even though pytest-timeout's thread method can't
    # interrupt it (see the module docstring). Reads the real env-derived default, not a reloaded copy.
    assert 0 < _hang_watchdog.HANG_WATCHDOG_SECONDS < 900
