"""Cluster 22 regression: non-finite rows are dropped from the group bias fit, with a warning (audit 2026-11-20, finding 22-F15).

One NaN prediction made the global mean NaN, so with shrinkage every group's ratio came out NaN and silently fell back to 1.0 at apply time. The rows with a
non-finite y_true or y_pred are now excluded, so the ratios come from the finite rows and stay close to the clean ones.
"""

import logging

import numpy as np

from mlframe.calibration.group_bias_correction import fit_group_bias_correction


def _data():
    """Two groups whose true mean is twice the prediction mean in group a and equal in group b."""
    group = np.array(["a"] * 20 + ["b"] * 20)
    y_pred = np.where(group == "a", 0.2, 0.4).astype(float)
    y_true = np.where(group == "a", 0.4, 0.4).astype(float)
    return y_true, y_pred, group


def test_nan_prediction_does_not_disable_shrunk_ratios(caplog):
    """With shrinkage and one NaN prediction the ratios equal those of the clean data, and a warning names the dropped row (22-F15)."""
    y_true, y_pred, group = _data()
    clean = fit_group_bias_correction(y_true, y_pred, group, shrinkage_k=5.0)
    y_nan = y_pred.copy()
    y_nan[0] = np.nan
    with caplog.at_level(logging.WARNING, logger="mlframe.calibration.group_bias_correction"):
        got = fit_group_bias_correction(y_true, y_nan, group, shrinkage_k=5.0)
    assert all(np.isfinite(v) for v in got.values())
    assert abs(got["b"] - clean["b"]) < 0.01  # one row fewer shifts the global mean a little; the correction is not disabled
    assert got["a"] > 1.0
    assert any("non-finite" in r.getMessage() for r in caplog.records)
