"""Binary confusion metrics: a single-class y_true averages recalls over the present class; an empty input is undefined, not 0."""

from __future__ import annotations

import numpy as np

from mlframe.metrics.classification._classification_extras_blocks import fast_binary_confusion_metrics_block


def test_perfect_classifier_on_single_class_y_true_scores_one():
    """y_true = y_pred = [1, 1, 1]: the negative recall is undefined, so balanced accuracy and G-mean are the positive recall, 1.0 (they were 0.5 and 0.0)."""
    out = fast_binary_confusion_metrics_block(np.array([1, 1, 1]), np.array([1, 1, 1]))
    assert out["balanced_accuracy"] == 1.0 and out["G_mean"] == 1.0
    out0 = fast_binary_confusion_metrics_block(np.array([0, 0, 0]), np.array([0, 0, 0]))
    assert out0["balanced_accuracy"] == 1.0 and out0["G_mean"] == 1.0


def test_two_class_values_are_unchanged():
    """With both classes present the mean and the geometric mean of the two recalls are as before."""
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    p = np.array([1, 1, 0, 0, 0, 0, 1])
    out = fast_binary_confusion_metrics_block(y, p)
    tpr, tnr = 2 / 3, 3 / 4
    assert abs(out["balanced_accuracy"] - 0.5 * (tpr + tnr)) < 1e-12
    assert abs(out["G_mean"] - (tpr * tnr) ** 0.5) < 1e-12


def test_empty_input_is_nan_everywhere():
    """No rows: every rate, F-score and MCC is NaN rather than 0.0, while the integer counts stay 0."""
    out = fast_binary_confusion_metrics_block(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    assert all(np.isnan(v) for k, v in out.items() if not k.startswith("_"))
    assert out["_TP"] == 0 and out["_FN"] == 0
