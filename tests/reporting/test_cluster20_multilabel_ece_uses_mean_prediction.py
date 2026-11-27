"""Cluster 20 regression: the multilabel reliability ECE uses each bin's mean prediction, not the bin centre (audit 2026-11-20, 20-F8)."""

import numpy as np

from mlframe.reporting.charts.multilabel import _calib_grid_panel


def test_perfectly_calibrated_off_centre_bin_has_zero_ece():
    """Every score is 0.11 and 11% of rows are positive: the model is calibrated, so ECE is ~0 (it was ~0.04 against the 0.15 centre) (20-F8)."""
    n = 10000
    y = np.zeros((n, 1))
    y[: int(n * 0.11), 0] = 1
    proba = np.full((n, 1), 0.11)
    panel = _calib_grid_panel(y, proba, ["a"])
    label = panel.series_labels[1]
    ece = float(label.split("ECE=")[1].split(",")[0])
    assert ece < 0.002
    assert panel.xlabel == "Mean predicted P(label)"
