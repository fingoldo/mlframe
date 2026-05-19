"""Tests for mlframe.training.crash_reporting.enable_crash_reporting().

Behavioural-only: never asserts source text. Covers:
  - idempotency of the module-level _ENABLED guard
  - faulthandler.is_enabled() flips True after first call
  - SetErrorMode is invoked on win32 with the WER-suppress flag

Uses real platform branches (no monkeypatching of sys.platform) and
restores the prior ErrorMode mask in a finalizer to keep the suite
process clean for sibling tests.
"""
from __future__ import annotations

import faulthandler
import sys

import pytest

from mlframe.training import crash_reporting


@pytest.fixture
def reset_enabled():
    """Reset the module-level guard around each test."""
    prev = crash_reporting._ENABLED
    crash_reporting._ENABLED = False
    try:
        yield
    finally:
        crash_reporting._ENABLED = prev


@pytest.mark.fast
def test_enable_returns_true_and_is_idempotent(reset_enabled):
    """First call enables; second call short-circuits and returns True."""
    first = crash_reporting.enable_crash_reporting()
    assert first is True
    # Second call must not re-run side effects but still returns True.
    second = crash_reporting.enable_crash_reporting()
    assert second is True
    assert crash_reporting._ENABLED is True


@pytest.mark.fast
def test_enable_turns_on_faulthandler(reset_enabled):
    """faulthandler.is_enabled() must be True after enable_crash_reporting()."""
    # If a sibling test already enabled it, disable first to verify the
    # *call* is what flips it - not pre-existing state.
    if faulthandler.is_enabled():
        faulthandler.disable()
    assert not faulthandler.is_enabled()
    crash_reporting.enable_crash_reporting()
    assert faulthandler.is_enabled()


@pytest.mark.skipif(sys.platform != "win32", reason="WER suppression is Windows-only")
def test_enable_sets_wer_suppress_flag_on_windows(reset_enabled):
    """On win32 the kernel32 ErrorMode mask must include SEM_NOGPFAULTERRORBOX."""
    import ctypes

    # Snapshot prior mask so we can restore it after.
    prev = ctypes.windll.kernel32.SetErrorMode(0)
    try:
        crash_reporting.enable_crash_reporting()
        current = ctypes.windll.kernel32.SetErrorMode(0)
        # Re-set immediately because reading via SetErrorMode(0) clears it.
        ctypes.windll.kernel32.SetErrorMode(current)
        SEM_NOGPFAULTERRORBOX = 0x0002
        assert current & SEM_NOGPFAULTERRORBOX, (
            f"WER-suppress flag missing from ErrorMode mask: 0x{current:04x}"
        )
    finally:
        ctypes.windll.kernel32.SetErrorMode(prev)
