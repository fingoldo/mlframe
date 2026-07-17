"""Regression tests for plot_pr_curve single-class guard + figure-leak fix (EDGE22 / LEAK-P2).

Pre-fix: a single-class y produced a degenerate PR/ROC + misleading baseline silently.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlframe.evaluation.reports import plot_pr_curve


def test_plot_pr_curve_single_class_raises():
    y = np.zeros(20, dtype=int)  # all one class
    preds = np.random.RandomState(0).rand(20)
    with pytest.raises(ValueError, match="single class"):
        plot_pr_curve(y, preds)


def test_plot_pr_curve_two_classes_returns_figure():
    rng = np.random.RandomState(0)
    y = np.array([0] * 10 + [1] * 10)
    preds = rng.rand(20)
    fig = plot_pr_curve(y, preds)
    assert fig is not None
    plt.close(fig)
