"""Regression tests for plot_pr_curve single-class guard + figure-leak fix (EDGE22 / LEAK-P2).

Pre-fix: a single-class y produced a degenerate PR/ROC + misleading baseline silently.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from mlframe.evaluation.reports import plot_pr_curve


def test_plot_pr_curve_single_class_raises():
    """Plot pr curve single class raises."""
    y = np.zeros(20, dtype=int)  # all one class
    preds = np.random.RandomState(0).rand(20)
    with pytest.raises(ValueError, match="single class"):
        plot_pr_curve(y, preds)


def test_plot_pr_curve_two_classes_returns_figure():
    """Plot pr curve two classes returns figure."""
    rng = np.random.RandomState(0)
    y = np.array([0] * 10 + [1] * 10)
    preds = rng.rand(20)
    fig = plot_pr_curve(y, preds)
    try:
        ax = fig.axes[0]
        assert ax.get_title() == f"PRC/ROC, BrierLoss={brier_score_loss(y, preds):.3f}"
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert any(lbl.startswith("PR AUC=") for lbl in labels)
        assert f"ROC AUC={roc_auc_score(y, preds):.3f}" in labels
    finally:
        plt.close(fig)
