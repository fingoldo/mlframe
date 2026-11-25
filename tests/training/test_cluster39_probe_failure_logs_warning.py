"""Cluster 39 regression: a failing degenerate-init probe is logged at WARNING, not DEBUG (audit 2026-11-20, finding 39-F62).

When the probe raises, the degenerate-weight check is skipped. That is non-fatal, but it must be visible. The source-level pin checks that the
handler logs at WARNING and names the skipped check.
"""

from pathlib import Path

from mlframe.training.neural import _flat_generate_helpers as fg


def test_probe_failure_handler_logs_warning():
    """The probe's except branch logs at WARNING and names the skipped check (39-F62)."""
    src = Path(fg.__file__).resolve().read_text(encoding="utf-8")
    start = src.index("except Exception as _rank_err:")
    handler = src[start : start + 400]
    assert "logger.warning(" in handler
    assert "degenerate-weight check skipped" in handler
