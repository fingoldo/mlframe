"""Fallbacks that substitute a value say so at WARNING.

A host-block estimate that fails runs the call with no memory reservation, and a pid whose start time cannot be read
falls back to the pid alone. Both used to do this at DEBUG, which an operator never sees.
"""

from __future__ import annotations

import logging

import psutil

from mlframe.system.kernel_tuning_cache import _ensure
from mlframe.utils.host_block_budget import reserves_host_block_budget


def _warnings_mentioning(caplog, text: str) -> list[logging.LogRecord]:
    """The WARNING records whose message contains ``text``."""
    return [r for r in caplog.records if r.levelno == logging.WARNING and text in r.getMessage()]


def test_failed_estimate_warns_and_runs_without_reservation(caplog):
    """An estimate that raises warns at WARNING, and the wrapped call still runs."""

    def _broken_estimate(*_a, **_k):
        """An estimate that fails."""
        raise RuntimeError("bad shape")

    @reserves_host_block_budget(_broken_estimate)
    def _work(x):
        """The call under the budget."""
        return x + 1

    with caplog.at_level(logging.WARNING):
        assert _work(1) == 2
    assert _warnings_mentioning(caplog, "estimate failed"), [r.getMessage() for r in caplog.records]


def test_unreadable_start_time_warns(monkeypatch, caplog):
    """A process whose start time cannot be read reports no token at WARNING."""

    def _broken_process(*_a, **_k):
        """A process lookup that fails."""
        raise OSError("access denied")

    monkeypatch.setattr(psutil, "Process", _broken_process)
    with caplog.at_level(logging.WARNING):
        assert _ensure._pid_start_token(12345) == ""
    assert _warnings_mentioning(caplog, "no process start token"), [r.getMessage() for r in caplog.records]


def test_vanished_process_warns_that_the_pid_alone_decides(monkeypatch, caplog):
    """A process that no longer exists substitutes an empty token, and that substitution is visible at WARNING."""

    def _gone(*_a, **_k):
        """A lookup for a process that has exited."""
        raise psutil.NoSuchProcess(12345)

    monkeypatch.setattr(psutil, "Process", _gone)
    with caplog.at_level(logging.WARNING):
        assert _ensure._pid_start_token(12345) == ""
    assert _warnings_mentioning(caplog, "has exited"), [r.getMessage() for r in caplog.records]
