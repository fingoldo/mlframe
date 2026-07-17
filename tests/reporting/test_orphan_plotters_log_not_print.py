"""Regression: orphan binary plotters log via logging, not print to stdout (INV-41).

INV-41: plot_pr_curve previously print()'d a classification_report to stdout (cp1251-unsafe,
bypasses log handlers). It now routes through logger.info. The canonical suite binary curves
live in reporting/charts/binary.py; these helpers remain as test-only / notebook shims.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mlframe.evaluation.reports import plot_pr_curve


def test_plot_pr_curve_logs_classification_report_not_stdout(capsys, caplog):
    """Plot pr curve logs classification report not stdout."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, size=n)
    preds = np.clip(y * 0.6 + rng.uniform(0, 0.4, size=n), 0, 1)
    with caplog.at_level(logging.INFO, logger="mlframe.evaluation.reports"):
        fig = plot_pr_curve(y, preds, thresh=0.5)
    plt.close(fig)

    captured = capsys.readouterr()
    # The classification report must NOT be printed to stdout.
    assert "precision" not in captured.out.lower(), "plot_pr_curve must not print the classification report to stdout (INV-41)"
    # It must reach the logging system instead.
    assert any("classification report" in r.message.lower() for r in caplog.records), "plot_pr_curve should log the classification report via logger.info"
