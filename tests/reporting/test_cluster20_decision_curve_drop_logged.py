"""Cluster 20 regression: the decision-curve helper logs the rows it drops (audit 2026-11-20, finding 20-F37).

A multiclass target or NaN scores used to shrink the curve to a silent subset. The helper now logs a WARNING with the dropped count; clean
binary input is silent and returns the same arrays.
"""

import logging

import numpy as np

from mlframe.reporting.charts import decision_curve as dc


def test_off_binary_labels_are_reported(caplog):
    """Labels outside {0, 1} are dropped with a WARNING naming the count (20-F37)."""
    y = np.array([0, 1, 2, 1, 0], dtype=float)
    s = np.array([0.1, 0.9, 0.5, 0.7, 0.2])
    with caplog.at_level(logging.WARNING, logger=dc.logger.name):
        yt, _ys = dc._finite_binary(y, s)
    assert yt.size == 4
    assert any(r.levelno == logging.WARNING and "dropped 1 of 5" in r.getMessage() for r in caplog.records)


def test_clean_binary_input_is_silent(caplog):
    """Control: clean binary input passes through with no warning (20-F37)."""
    y = np.array([0, 1, 1, 0], dtype=float)
    s = np.array([0.1, 0.9, 0.7, 0.2])
    with caplog.at_level(logging.WARNING, logger=dc.logger.name):
        yt, ys = dc._finite_binary(y, s)
    assert yt.size == 4 and ys.size == 4
    assert not any("dropped" in r.getMessage() for r in caplog.records)
