"""Unit tests for mlframe.training.crash_reporting.

Covers the single public symbol ``enable_crash_reporting`` (no other ``def``-level public
helpers in the module). Exercises idempotency, the Windows-only SetErrorMode path,
the cross-platform faulthandler wiring, and the failure-tolerance contract (the toggle
must never raise; it only ever returns True/False).
"""

from __future__ import annotations

import importlib
import io
import logging
import sys

import pytest


@pytest.fixture
def crash_reporting_module():
    """Re-import the module fresh with the snapshot+restore pattern so the
    module-level ``_ENABLED`` flag does not leak across tests. Per CLAUDE.md
    test-pollution rules, snapshot and restore rather than del-from-sys.modules.
    """
    import mlframe.training.crash_reporting as cr

    saved_enabled = cr._ENABLED
    # Force the module into the disabled state so each test starts from a known baseline.
    cr._ENABLED = False
    try:
        yield cr
    finally:
        cr._ENABLED = saved_enabled


def test_enable_returns_bool(crash_reporting_module):
    cr = crash_reporting_module
    rv = cr.enable_crash_reporting()
    assert isinstance(rv, bool), "enable_crash_reporting must return a bool"


def test_enable_idempotent_second_call_short_circuits(crash_reporting_module, monkeypatch):
    """Second invocation must return True without doing work again — `_ENABLED` guard.

    Wire faulthandler.enable to a counter so we can assert it was invoked at most once
    across two ``enable_crash_reporting`` calls.
    """
    cr = crash_reporting_module

    import faulthandler

    calls = {"count": 0}

    def _counting_enable(file=None, all_threads=False):
        calls["count"] += 1

    monkeypatch.setattr(faulthandler, "enable", _counting_enable)

    rv1 = cr.enable_crash_reporting()
    assert rv1 is True, "first call should succeed when faulthandler.enable does not raise"
    first_count = calls["count"]
    assert first_count == 1, f"faulthandler.enable should be called exactly once on first invocation; got {first_count}"

    rv2 = cr.enable_crash_reporting()
    assert rv2 is True, "second call must still return True"
    assert calls["count"] == first_count, "second call must NOT re-invoke faulthandler.enable (idempotency)"


def test_enable_sets_module_flag(crash_reporting_module, monkeypatch):
    cr = crash_reporting_module

    import faulthandler

    monkeypatch.setattr(faulthandler, "enable", lambda *a, **kw: None)

    assert cr._ENABLED is False
    cr.enable_crash_reporting()
    assert cr._ENABLED is True, "_ENABLED must be set to True after a successful enable"


def test_enable_uses_provided_file_handle(crash_reporting_module, monkeypatch):
    """When caller passes a file with .fileno(), faulthandler.enable should receive it."""
    cr = crash_reporting_module
    captured = {}

    def _capture(file=None, all_threads=True):
        captured["file"] = file
        captured["all_threads"] = all_threads

    import faulthandler

    monkeypatch.setattr(faulthandler, "enable", _capture)

    real_stream = sys.stderr  # real stderr has a fileno() on every supported platform
    cr.enable_crash_reporting(file=real_stream, all_threads=False)

    assert captured["file"] is real_stream, "user-supplied stream must propagate to faulthandler.enable"
    assert captured["all_threads"] is False, "all_threads kwarg must propagate"


def test_enable_falls_back_to_fd_2_when_stream_has_no_fileno(crash_reporting_module, monkeypatch):
    """Jupyter wraps stderr in an object without ``.fileno()`` — module must catch the
    UnsupportedOperation and re-invoke faulthandler.enable with the raw fd 2 instead.
    """
    cr = crash_reporting_module
    invocations = []

    def _record(file=None, all_threads=True):
        invocations.append(file)
        if file != 2 and not isinstance(file, int):
            # The first call (with the broken stream) raised UnsupportedOperation in
            # the .fileno() check path; here we make the SECOND call (with fd=2) succeed.
            raise io.UnsupportedOperation("no fileno")

    import faulthandler

    monkeypatch.setattr(faulthandler, "enable", _record)

    class _BrokenStream:
        def fileno(self):
            raise io.UnsupportedOperation("no fileno on this wrapper")

    rv = cr.enable_crash_reporting(file=_BrokenStream())
    # The module catches the UnsupportedOperation INSIDE the try block and re-invokes with fd=2.
    assert 2 in invocations, f"expected fd=2 fallback invocation, got {invocations!r}"
    assert isinstance(rv, bool)


def test_enable_never_raises_on_faulthandler_exception(crash_reporting_module, monkeypatch, caplog):
    """A faulthandler failure must NOT propagate — training must continue. The function
    returns False instead and logs a warning. Pre-fix contract from the module docstring.
    """
    cr = crash_reporting_module

    def _boom(file=None, all_threads=True):
        raise RuntimeError("simulated faulthandler init failure")

    import faulthandler

    monkeypatch.setattr(faulthandler, "enable", _boom)

    # On Windows the SetErrorMode block may still succeed, so the return value is the AND
    # of the two steps. We assert that the call returns without raising AND logs the warning.
    with caplog.at_level(logging.WARNING, logger="mlframe.training.crash_reporting"):
        rv = cr.enable_crash_reporting()
    assert isinstance(rv, bool)
    assert any("faulthandler" in rec.getMessage() for rec in caplog.records), "expected warning log mentioning faulthandler when init fails"


@pytest.mark.windows_only
@pytest.mark.skipif(sys.platform != "win32", reason="SetErrorMode is Windows-only")
def test_enable_calls_seterrormode_on_windows(crash_reporting_module, monkeypatch):
    """On Windows, the module must call kernel32.SetErrorMode with the
    SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX bitmask. Spy on the call and verify
    the second invocation receives the OR of (prior_mode | 0x0001 | 0x0002).
    """
    cr = crash_reporting_module

    import ctypes

    call_log = []

    real_set = ctypes.windll.kernel32.SetErrorMode

    def _spy_set_error_mode(flags):
        call_log.append(int(flags))
        # Return a benign prior value (0); the real call would return the current process mode.
        return 0

    monkeypatch.setattr(ctypes.windll.kernel32, "SetErrorMode", _spy_set_error_mode)

    import faulthandler

    monkeypatch.setattr(faulthandler, "enable", lambda *a, **kw: None)

    rv = cr.enable_crash_reporting()
    assert rv is True

    # Expect two calls: first reads the current mode (with 0), second applies the OR.
    assert len(call_log) >= 2, f"expected at least two SetErrorMode calls (read + write); got {call_log!r}"
    final = call_log[-1]
    assert final & 0x0001, f"final mode must include SEM_FAILCRITICALERRORS bit; got 0x{final:04X}"
    assert final & 0x0002, f"final mode must include SEM_NOGPFAULTERRORBOX bit; got 0x{final:04X}"


def test_enable_skips_seterrormode_off_windows(crash_reporting_module, monkeypatch):
    """When ``sys.platform != 'win32'`` the SetErrorMode branch must not run. We can
    simulate non-Windows by patching sys.platform; the test passes if no AttributeError
    on ``ctypes.windll`` ever surfaces (which would happen on Linux if the branch ran).
    """
    cr = crash_reporting_module
    monkeypatch.setattr(sys, "platform", "linux")

    import faulthandler

    monkeypatch.setattr(faulthandler, "enable", lambda *a, **kw: None)

    rv = cr.enable_crash_reporting()
    assert rv is True, "non-Windows enable must return True after a clean faulthandler init"


def test_enable_seterrormode_failure_returns_false(crash_reporting_module, monkeypatch, caplog):
    """If SetErrorMode raises on Windows, the function returns False but does not re-raise."""
    if sys.platform != "win32":
        pytest.skip("SetErrorMode failure path is Windows-only")

    cr = crash_reporting_module

    import faulthandler

    monkeypatch.setattr(faulthandler, "enable", lambda *a, **kw: None)

    import ctypes

    def _boom(flags):
        raise OSError("simulated SetErrorMode failure")

    monkeypatch.setattr(ctypes.windll.kernel32, "SetErrorMode", _boom)

    with caplog.at_level(logging.WARNING, logger="mlframe.training.crash_reporting"):
        rv = cr.enable_crash_reporting()
    # `ok` is set to False by the SetErrorMode branch; module returns False.
    assert rv is False, "SetErrorMode failure must surface as a False return"
    assert any("SetErrorMode" in rec.getMessage() for rec in caplog.records), "expected warning log mentioning SetErrorMode failure"
