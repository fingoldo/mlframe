"""Cluster 26 and 34 regression: two documented choices are stated where the code makes them (audit 2026-11-20, findings 26-F26 and 34-F20).

26-F26: a queued figure is recorded as saved at submit; the worker's later failure is not reconciled into the chart status.
34-F20: the warm-start signature is computed on the full frame, not a sample, so a changed row outside a sample cannot reuse stale specs.
"""

from pathlib import Path

from mlframe.reporting import diagnostics_dispatch
from mlframe.training.composite.discovery import _incremental


def test_queued_figure_choice_is_documented():
    """The saved-at-submit recording and its reconciliation gap are stated at the submit site (26-F26)."""
    src = Path(diagnostics_dispatch.__file__).read_text(encoding="utf-8")
    assert "documented choice" in src
    assert "charts['failed']" in src


def test_warm_start_full_frame_signature_choice_is_documented():
    """The warm-start check states that the signature is computed on the full frame (34-F20)."""
    src = Path(_incremental.__file__).read_text(encoding="utf-8")
    assert "audit 34-F20" in src
    assert "FULL frame" in src
