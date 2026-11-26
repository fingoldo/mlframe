"""Cluster 21 regression: split metrics report the full split's row counts, not the subsample cap (audit 2026-11-20, 21-F11)."""

import numpy as np

from mlframe.reporting.charts.split_comparison import _metrics_for_split


def test_binary_counts_come_from_the_full_split():
    """With 5,000 rows and a 500-row cap, n_rows/n_pos/n_neg describe all 5,000 rows (21-F11)."""
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=5000) < 0.3).astype(int)
    s = np.clip(y * 0.3 + rng.uniform(size=5000) * 0.7, 0, 1)
    m, note = _metrics_for_split("binary", {"y_true": y, "y_score": s}, 0.5, 500, 0)
    assert note is None
    assert m["n_rows"] == 5000.0
    assert m["n_pos"] == float(y.sum()) and m["n_neg"] == float((y == 0).sum())


def test_regression_row_count_comes_from_the_full_split():
    """The regression n_rows is the full split size too (21-F11)."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=4000)
    m, _ = _metrics_for_split("regression", {"y_true": y, "y_pred": y + rng.normal(size=4000)}, 0.5, 300, 0)
    assert m["n_rows"] == 4000.0
