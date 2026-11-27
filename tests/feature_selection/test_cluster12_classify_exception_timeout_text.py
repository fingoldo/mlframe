"""Cluster 12 regression: a bad-argument error that mentions a timeout is recorded as an error, not a timeout (audit 2026-11-20, 12-F17)."""

import subprocess

from mlframe.feature_selection._benchmarks.fs_hybrid._protocol_types import classify_exception


def test_argument_errors_mentioning_timeout_stay_errors():
    """ValueError / TypeError / KeyError with the word timeout in the message are plain errors (12-F17)."""
    assert classify_exception(ValueError("timeout parameter invalid")) == "error"
    assert classify_exception(TypeError("got an unexpected keyword argument 'timeout'")) == "error"
    assert classify_exception(KeyError("timed out")) == "error"


def test_real_timeouts_are_still_timeouts():
    """TimeoutError, a runtime error that says it timed out, and a subprocess timeout keep the timeout status (12-F17)."""
    assert classify_exception(TimeoutError()) == "timeout"
    assert classify_exception(RuntimeError("worker timed out after 60s")) == "timeout"
    assert classify_exception(subprocess.TimeoutExpired("cmd", 5)) == "timeout"
