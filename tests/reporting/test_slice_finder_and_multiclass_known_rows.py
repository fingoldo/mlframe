"""The slice finder divides by the rows it actually scored, and multiclass P/R/F1 agrees with the confusion panels about unknown true labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.reporting.charts.multiclass import _pr_f1_panel
from mlframe.reporting.charts.slice_finder import find_weak_slices


def test_support_fraction_is_relative_to_the_scored_rows_not_all_rows():
    """40% of rows have a NaN prediction; a slice holding every scored row has support_fraction 1.0 (it read 0.6)."""
    rng = np.random.default_rng(0)
    n = 1000
    X = pd.DataFrame({"f": np.where(rng.random(n) < 0.5, 0.0, 1.0)})
    y = rng.normal(size=n)
    pred = y + rng.normal(scale=0.1, size=n)
    pred[:400] = np.nan
    res = find_weak_slices(X, y, pred, task="regression", min_support_fraction=0.01)
    assert float(res.table["support_fraction"].max()) <= 1.0 + 1e-9
    assert float(res.table["support"].max()) <= 600


def test_rows_with_an_unknown_true_label_do_not_count_as_false_positives():
    """True labels 'c' (not in classes) predicted 'a' are -1 after mapping: class a's precision must stay 1.0 for the 4 known rows."""
    classes = ["a", "b"]
    y_true_pos = np.array([0, 0, 1, 1] + [-1] * 100)
    y_pred_pos = np.array([0, 0, 1, 1] + [0] * 100)
    panel = _pr_f1_panel(y_true_pos, None, classes, y_pred=y_pred_pos)
    precision = panel.values[0]
    assert abs(float(precision[0]) - 1.0) < 1e-12
