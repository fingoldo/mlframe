"""Cluster 32 regression: the baseline diagnostic skips below 50 rows instead of scoring in-sample (audit 2026-11-20, finding 32-F12).

With fewer than 50 rows the quick model has no holdout, so its headline metric and ablation deltas are computed on the training rows. The
diagnostic now returns a skipped report that names the reason.
"""

import numpy as np
import pandas as pd

from mlframe.training.baselines.diagnostics import BaselineDiagnostics
from mlframe.training.configs import BaselineDiagnosticsConfig


def _frame(n: int, seed: int = 0):
    """A small frame with two numeric features and a binary target."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = (X["a"] > 0).astype(int).to_numpy()
    return X, y


def test_below_fifty_rows_is_skipped_with_reason():
    """A 40-row frame returns a skipped report naming the in-sample reason (32-F12)."""
    X, y = _frame(40)
    rep = BaselineDiagnostics(BaselineDiagnosticsConfig()).fit_and_report(X, y, ["a", "b"], "binary_classification", "t")
    assert rep.skipped is True
    assert "in-sample" in rep.skip_reason


def test_fifty_rows_is_not_skipped_for_the_minimum_check():
    """Control: at the minimum row count the minimum-row skip does not fire (32-F12)."""
    X, y = _frame(60)
    rep = BaselineDiagnostics(BaselineDiagnosticsConfig()).fit_and_report(X, y, ["a", "b"], "binary_classification", "t")
    assert "in-sample" not in (rep.skip_reason or "")
