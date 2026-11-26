"""Cluster 41 regression: a non-numeric extra metric is printed as text and does not drop the report (audit 2026-11-20, finding 41-F34).

The report formatted each extra metric with ``:+.4f``. A metric that is not a number raised inside the block and the whole report was lost.
Such a metric is now printed with ``str``, and the numeric metrics are still formatted.
"""

import logging

import numpy as np

from mlframe.training import metrics_registry
from mlframe.training.reporting._reporting_regression import _mtr


def _render(monkeypatch, caplog, extras):
    """Render the MTR report for a fixed prediction matrix with the given extra metrics and return the INFO messages it logged."""
    monkeypatch.setattr(metrics_registry, "iter_extra_metrics", lambda *_args, **_kwargs: iter(extras))
    targets = np.arange(12, dtype=float).reshape(6, 2)
    preds = targets + 0.1
    with caplog.at_level(logging.INFO, logger=_mtr.logger.name):
        _mtr.render_mtr_report(
            targets,
            preds,
            "model",
            metrics={},
            print_report=True,
            plot_outputs=None,
            plot_file="",
            figsize=(4, 3),
            plot_sample_size=10,
            plot_dpi=None,
            report_title="report",
            verbose=False,
        )
    return [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]


def test_non_numeric_metric_is_printed_as_text_and_numeric_ones_still_format(monkeypatch, caplog):
    """A text metric appears as ``label = n/a`` and a numeric one as a signed four-decimal value, in the same report (41-F34)."""
    messages = _render(monkeypatch, caplog, [("ok_metric", 0.5), ("label", "n/a")])
    report = "\n".join(messages)
    assert "ok_metric = +0.5000" in report
    assert "label = n/a" in report


def test_numeric_metrics_alone_keep_the_signed_format(monkeypatch, caplog):
    """Control: with only numeric extra metrics the report prints them as signed four-decimal values (41-F34)."""
    messages = _render(monkeypatch, caplog, [("ok_metric", 0.25)])
    assert "ok_metric = +0.2500" in "\n".join(messages)
