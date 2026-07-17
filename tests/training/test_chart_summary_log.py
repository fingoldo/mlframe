"""Regression: suite-end chart-count summary independent of verbose (INV-14).

A default suite run saves nothing and renders nothing with no trace, so the operator cannot
tell "skipped by design" from "lost to a bug". log_chart_summary reads the metadata["charts"]
accounting and always logs the count + destination (or the data_dir hint, or the failure count).
"""

from __future__ import annotations

import logging

from mlframe.training.core._setup_helpers import log_chart_summary


def test_zero_charts_no_destination_emits_hint(caplog):
    with caplog.at_level(logging.INFO):
        msg = log_chart_summary({}, save_charts=False, data_dir="")
    assert "0 charts saved" in msg
    assert "output_config.data_dir" in msg
    assert any("0 charts saved" in r.message for r in caplog.records)


def test_saved_charts_reports_count_and_destination(caplog):
    metadata = {"charts": {"saved": ["val_panels", "test_panels", "target_dist"], "failed": []}}
    with caplog.at_level(logging.INFO):
        msg = log_chart_summary(metadata, save_charts=True, data_dir="/tmp/run")  # nosec B108 -- placeholder path string only, never touches the real filesystem
    assert "3 chart(s) saved" in msg
    assert "/tmp/run/charts" in msg  # nosec B108 -- placeholder path string only, never touches the real filesystem


def test_failed_renders_surfaced(caplog):
    metadata = {"charts": {"saved": ["val_panels"], "failed": ["test_panels"]}}
    with caplog.at_level(logging.INFO):
        msg = log_chart_summary(metadata, save_charts=True, data_dir="/tmp/run")  # nosec B108 -- placeholder path string only, never touches the real filesystem
    assert "1 chart(s) saved" in msg
    assert "1 render(s) failed" in msg


def test_missing_charts_key_is_safe(caplog):
    # No metadata["charts"] at all (e.g. a run that never reached the chart path).
    with caplog.at_level(logging.INFO):
        msg = log_chart_summary({"other": 1}, save_charts=True, data_dir="/tmp/run")  # nosec B108 -- placeholder path string only, never touches the real filesystem
    assert "0 chart(s) saved" in msg or "0 charts saved" in msg


def test_none_metadata_does_not_crash():
    # Must not raise on a None metadata (defensive).
    log_chart_summary(None, save_charts=False, data_dir="")
