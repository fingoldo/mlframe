"""Cluster 41 regression: confusion-derived metrics use the same threshold as precision/recall/F1 (audit 2026-11-20, finding 41-F17)."""

import numpy as np

from mlframe.training.reporting._reporting_probabilistic_helpers import _report_probabilist_dropping_new_fields


def _run(threshold):
    """Run the derived-metrics block at a threshold and return the accuracy it stamped."""
    y_score = np.array([0.3, 0.7, 0.35, 0.8])
    y_true = np.array([0, 1, 1, 1])
    class_metrics = {}
    _report_probabilist_dropping_new_fields(y_score, y_true, class_metrics, 0.5, "pos", threshold=threshold)
    return class_metrics


def test_accuracy_follows_the_tuned_threshold():
    """At threshold 0.75 only the 0.8 row is positive (accuracy 2/4); at 0.5 it is 3/4 (41-F17)."""
    assert _run(0.75)["accuracy"] == 0.5
    assert _run(0.5)["accuracy"] == 0.75


def test_default_threshold_is_half_when_not_tuned():
    """Control: omitting the threshold keeps the historical 0.5 cut (41-F17)."""
    assert _run_default()["accuracy"] == 0.75


def _run_default():
    """Call the helper without a threshold argument."""
    y_score = np.array([0.3, 0.7, 0.35, 0.8])
    y_true = np.array([0, 1, 1, 1])
    class_metrics = {}
    _report_probabilist_dropping_new_fields(y_score, y_true, class_metrics, 0.5, "pos")
    return class_metrics
