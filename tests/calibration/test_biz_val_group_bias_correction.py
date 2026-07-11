"""biz_value test for ``calibration.group_bias_correction`` (``fit_group_bias_correction``, ``apply_group_bias_correction``).

The win (5th_m5-forecasting-accuracy.md): a model can be well-calibrated ON AVERAGE while systematically
over/under-predicting for specific segments (opposite-direction biases that cancel in the global mean). A
single global correction factor can't fix this (the average bias is ~1.0), but a per-group correction
recovers accuracy for every segment.
"""
from __future__ import annotations

import numpy as np

from mlframe.calibration.group_bias_correction import apply_group_bias_correction, fit_group_bias_correction


def _make_opposite_bias_dataset(n_per_group: int, seed: int):
    rng = np.random.default_rng(seed)
    groups = np.repeat(["store_A", "store_B", "store_C", "store_D"], n_per_group)
    n = len(groups)

    true_demand = rng.uniform(50, 150, n)
    # model systematically UNDER-predicts store_A/B and OVER-predicts store_C/D -- opposite biases that
    # cancel in the GLOBAL mean(true)/mean(pred) ratio (~1.0), invisible to a single global correction.
    bias_factor = np.select([groups == "store_A", groups == "store_B", groups == "store_C", groups == "store_D"], [0.7, 0.8, 1.3, 1.2])
    y_pred = true_demand * bias_factor + rng.normal(scale=2.0, size=n)
    return true_demand, y_pred, groups


def test_biz_val_group_bias_correction_fixes_opposite_direction_biases():
    y_true, y_pred, groups = _make_opposite_bias_dataset(n_per_group=200, seed=0)

    global_ratio = float(y_true.mean() / y_pred.mean())
    assert abs(global_ratio - 1.0) < 0.05, f"sanity check: expected the opposite-direction biases to roughly cancel in the global mean ratio, got {global_ratio:.4f}"

    global_corrected = y_pred * global_ratio
    mae_global_correction = float(np.mean(np.abs(y_true - global_corrected)))

    ratios = fit_group_bias_correction(y_true, y_pred, groups)
    group_corrected = apply_group_bias_correction(y_pred, groups, ratios)
    mae_group_correction = float(np.mean(np.abs(y_true - group_corrected)))

    mae_uncorrected = float(np.mean(np.abs(y_true - y_pred)))

    assert mae_group_correction < mae_global_correction * 0.5, f"expected per-group correction to far outperform a global correction (which can't see the canceling biases), got group={mae_group_correction:.2f} global={mae_global_correction:.2f}"
    assert mae_group_correction < mae_uncorrected * 0.5, f"expected per-group correction to materially reduce error vs uncorrected predictions, got corrected={mae_group_correction:.2f} uncorrected={mae_uncorrected:.2f}"


def test_fit_group_bias_correction_falls_back_for_small_groups():
    y_true = np.array([10.0, 10.0, 10.0, 100.0, 100.0])
    y_pred = np.array([5.0, 5.0, 5.0, 50.0, 50.0])
    groups = np.array(["big", "big", "big", "tiny", "tiny"])

    ratios = fit_group_bias_correction(y_true, y_pred, groups, min_group_size=3)
    assert ratios["big"] == 2.0
    assert ratios["tiny"] == 1.0  # too few rows (2 < min_group_size=3) -> no correction


def test_fit_group_bias_correction_respects_clip_range():
    y_true = np.array([1000.0, 1000.0, 1000.0])
    y_pred = np.array([0.01, 0.01, 0.01])
    groups = np.array(["g", "g", "g"])

    ratios = fit_group_bias_correction(y_true, y_pred, groups, min_group_size=1, clip_range=(0.5, 2.0))
    assert ratios["g"] == 2.0  # clipped from the raw ~100000x ratio


def test_apply_group_bias_correction_unseen_group_uses_default():
    y_pred = np.array([10.0, 20.0])
    groups = np.array(["unseen1", "unseen2"])
    out = apply_group_bias_correction(y_pred, groups, ratios={}, default_ratio=1.5)
    np.testing.assert_allclose(out, [15.0, 30.0])
