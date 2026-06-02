"""Regression: multiclass reporting must treat class labels by POSITION.

Multiclass targets are not label-encoded to 0..K-1 anywhere in mlframe, so the
raw labels reaching the reporting layer may be e.g. [1, 2, 3] or [10, 20, 30].

#2  reporting/charts/multiclass.py: every panel builder indexes a K-sized
    structure positionally (``matrix[int(t)]``, ``y_true == k``,
    ``labels=range(K)``). With raw non-0..K-1 labels this IndexErrors (the whole
    multiclass figure is then silently dropped by the dispatcher) or builds a
    wrong confusion matrix / zero top-k. compose_multiclass_figure now remaps
    y_true to positions once before dispatching to the panels.

#3  _reporting_probabilistic.py: the weighted-metric supports counted
    ``_yt_all == cid`` where ``cid`` is the per-class ENUMERATE position, not the
    class label -> for non-0-indexed integer labels the supports shift and
    ``weighted_*`` aggregates are silently wrong. Now counts against the actual
    class label ``classes[cid]``.
"""
from __future__ import annotations

import numpy as np


def test_compose_multiclass_figure_handles_non_contiguous_labels():
    from mlframe.reporting.charts.multiclass import compose_multiclass_figure

    classes = [10, 20, 30]  # non-0..K-1 labels: pre-fix matrix[int(10)] -> IndexError
    y_true = np.array([10, 10, 20, 20, 30, 30])
    # Perfect predictions: one-hot on each row's true-class POSITION.
    y_proba = np.eye(3)[[0, 0, 1, 1, 2, 2]].astype(float)

    fig = compose_multiclass_figure(
        y_true, y_proba, classes,
        panels_template="CONFUSION PR_F1 ROC PR_CURVES CALIB_GRID PROB_DIST TOP_K_ACC",
    )
    assert fig is not None

    mats = [p for row in fig.panels for p in row if p is not None and hasattr(p, "matrix")]
    assert mats, "confusion heatmap panel missing"
    conf = np.asarray(mats[0].matrix)
    # Perfect predictions -> row-normalised confusion is the identity.
    assert np.allclose(np.diag(conf), 1.0), f"confusion diagonal not all 1.0: {np.diag(conf)}"


def test_weighted_metric_supports_use_class_label_not_enumerate_index():
    from mlframe.training._reporting_probabilistic import report_probabilistic_model_perf

    classes = [1, 2, 3]  # non-0-indexed integer multiclass labels
    # Skewed supports: 60 of class 1, 30 of class 2, 10 of class 3.
    targets = np.array([1] * 60 + [2] * 30 + [3] * 10)
    # Predict class 1 (position 0) for everyone -> class-1 recall=1.0, others 0.0.
    probs = np.zeros((100, 3))
    probs[:, 0] = 1.0
    preds = np.full(100, classes[0])  # predicted label = 1
    metrics: dict = {}

    report_probabilistic_model_perf(
        targets=targets, columns=["f"], model_name="m", model=None,
        classes=classes, preds=preds, probs=probs, metrics=metrics,
        print_report=False, show_perf_chart=False, verbose=False,
    )

    # Correct support-weighted recall = (1.0*60 + 0*30 + 0*10)/100 = 0.6.
    # Pre-fix the class-1 block (enumerate position 0) got support
    # count(targets==0)=0 -> dropped -> weighted_recall collapsed to 0.0.
    assert "weighted_recall" in metrics, f"weighted_recall absent; keys={sorted(metrics)}"
    assert abs(float(metrics["weighted_recall"]) - 0.6) < 0.05, (
        f"weighted_recall={metrics['weighted_recall']} (correct=0.6, pre-fix buggy=0.0)"
    )
