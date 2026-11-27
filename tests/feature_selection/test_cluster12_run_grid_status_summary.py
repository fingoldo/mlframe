"""Cluster 12 regression: run_grid ends with one summary of the cells that did not finish ok (audit 2026-11-20, 12-F74)."""

import logging
from collections import Counter

from mlframe.feature_selection._benchmarks.fs_hybrid import run_experiment as re_


def test_all_ok_gives_no_summary(caplog):
    """A run whose cells all finished ok logs nothing and returns None (12-F74)."""
    with caplog.at_level(logging.WARNING, logger=re_.logger.name):
        assert re_._log_status_summary(Counter({"ok": 5})) is None
    assert not caplog.records


def test_failures_are_counted_by_status_with_the_resume_hint(caplog):
    """Error, timeout and oom counts are named with the total, and the retry-failed hint is included (12-F74)."""
    with caplog.at_level(logging.WARNING, logger=re_.logger.name):
        msg = re_._log_status_summary(Counter({"ok": 7, "error": 2, "timeout": 1}))
    assert msg is not None and msg.startswith("3 of 10 cells")
    assert "error=2" in msg and "timeout=1" in msg and "--retry-failed" in msg
    assert any(r.levelno == logging.WARNING and "run_grid" in r.getMessage() for r in caplog.records)
