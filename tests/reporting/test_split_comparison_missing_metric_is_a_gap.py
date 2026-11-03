"""A missing classification metric in the split-comparison bars is a gap (NaN), not a bar at quality 0."""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.split_comparison import _grouped_bar_panel


def test_missing_ece_on_test_is_nan_not_zero():
    """Test has no ECE: the bar value must be NaN. It was 0.0, which reads as 'perfectly miscalibrated'."""
    per = {
        "train": {"ROC_AUC": 0.9, "ECE": 0.02, "n_rows": 100},
        "test": {"ROC_AUC": 0.85, "n_rows": 100},
    }
    panel = _grouped_bar_panel("binary", ["train", "test"], per, "green")
    cats = list(panel.categories)
    assert "ECE" in cats
    test_series = panel.values[1]
    assert np.isnan(test_series[cats.index("ECE")])
    assert np.isfinite(panel.values[0][cats.index("ECE")])


def test_capitalised_split_names_give_the_same_verdict_as_lowercase():
    """{'Test','Val','Train'} in that insertion order: case-sensitive lookups paired Test with Train in the wrong direction and could turn overfit into 'generalises'."""
    from mlframe.reporting.charts.split_comparison import overfit_verdict

    metrics = {"train": {"ROC_AUC": 0.99, "n_rows": 1000}, "val": {"ROC_AUC": 0.80, "n_rows": 500}, "test": {"ROC_AUC": 0.78, "n_rows": 500}}
    lower = overfit_verdict(task="binary", per_split={k: {"metrics": v} for k, v in metrics.items()})
    shuffled_caps = {"Test": {"metrics": metrics["test"]}, "Val": {"metrics": metrics["val"]}, "Train": {"metrics": metrics["train"]}}
    caps = overfit_verdict(task="binary", per_split=shuffled_caps)
    assert caps == lower
