"""Fallback paths in the MRMR filters log a warning when they substitute a value.

A handler that returns a default without a visible log hides the failure it was written for: the caller carries on with a value nobody
can see was substituted. Each of these probes now reports its fallback at WARNING.
"""

from __future__ import annotations

import logging

import pandas as pd
import psutil

from mlframe.feature_selection.filters import _adaptive_nbins as adaptive
from mlframe.feature_selection.filters import _fe_family_timing as timing
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _copy_on_write_active


def _warnings_mentioning(caplog, text: str) -> list[logging.LogRecord]:
    """The WARNING records whose message contains text."""
    return [r for r in caplog.records if r.levelno == logging.WARNING and text in r.getMessage()]


def test_unreadable_rss_warns_and_reports_zero(monkeypatch, caplog):
    """A process whose memory cannot be read reports 0 bytes and says so at WARNING."""

    def _broken_process(*_a, **_k):
        """A psutil process lookup that fails."""
        raise psutil.Error("no access to the process table")

    monkeypatch.setattr(psutil, "Process", _broken_process)
    with caplog.at_level(logging.WARNING):
        assert timing._process_rss() == 0.0
    assert _warnings_mentioning(caplog, "fe family RSS unreadable"), [r.getMessage() for r in caplog.records]


def test_unreadable_copy_on_write_warns_and_assumes_deep_copy(monkeypatch, caplog):
    """A pandas version string that cannot be parsed falls back to the deep copy and says so at WARNING."""
    monkeypatch.setattr(pd, "__version__", "unparseable")
    with caplog.at_level(logging.WARNING):
        assert _copy_on_write_active() is False
    assert _warnings_mentioning(caplog, "copy-on-write probe failed"), [r.getMessage() for r in caplog.records]


def test_edge_pool_cpu_probe_failure_warns(monkeypatch, caplog):
    """A failed logical-core probe sizes the edge pool to one worker and says so at WARNING."""

    def _broken_count(*_a, **_k):
        """A logical-core probe that fails."""
        raise OSError("cpu topology unavailable")

    monkeypatch.setattr(psutil, "cpu_count", _broken_count)
    with caplog.at_level(logging.WARNING):
        adaptive._edge_pool(1)
    assert _warnings_mentioning(caplog, "edge pool: cpu count probe failed"), [r.getMessage() for r in caplog.records]
