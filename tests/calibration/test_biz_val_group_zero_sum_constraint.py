"""biz_value test for ``calibration.group_zero_sum_constraint.apply_group_zero_sum_constraint``.

Source: 9th_optiver-trading-at-the-close.md -- predictions subtracted their `groupby(date_id,
seconds_in_bucket)` weighted mean so the weighted sum is exactly zero within each group, matching a known
property of the (cross-sectional, relative) target. When a model's per-group bias drifts around the true
zero-sum constraint, re-centering each group's predictions to satisfy the KNOWN constraint removes that drift
entirely -- this test confirms it materially reduces RMSE versus the raw (uncorrected) predictions, and that
the corrected predictions satisfy the constraint exactly.
"""
from __future__ import annotations

import numpy as np

from mlframe.calibration.group_zero_sum_constraint import apply_group_zero_sum_constraint


def _make_zero_sum_target_with_biased_predictions(n_groups: int, group_size: int, seed: int):
    rng = np.random.default_rng(seed)
    n = n_groups * group_size
    group = np.repeat(np.arange(n_groups), group_size)

    raw_signal = rng.normal(size=n)
    group_means = np.array([raw_signal[group == g].mean() for g in range(n_groups)])
    true_y = raw_signal - np.repeat(group_means, group_size)  # exact zero-sum within each group, by construction.

    group_bias = np.repeat(rng.normal(scale=0.3, size=n_groups), group_size)  # model drifts off the constraint per group.
    pred = true_y + rng.normal(scale=0.2, size=n) + group_bias
    return pred, true_y, group


def test_biz_val_group_zero_sum_constraint_reduces_rmse():
    pred, true_y, group = _make_zero_sum_target_with_biased_predictions(n_groups=200, group_size=20, seed=0)

    corrected = apply_group_zero_sum_constraint(pred, group)

    rmse_raw = float(np.sqrt(np.mean((pred - true_y) ** 2)))
    rmse_corrected = float(np.sqrt(np.mean((corrected - true_y) ** 2)))

    assert rmse_corrected < rmse_raw * 0.7, f"expected zero-sum re-centering to cut RMSE by >=30%, got corrected={rmse_corrected:.4f} raw={rmse_raw:.4f}"


def test_group_zero_sum_constraint_satisfies_exact_constraint():
    pred, _, group = _make_zero_sum_target_with_biased_predictions(n_groups=50, group_size=10, seed=1)
    corrected = apply_group_zero_sum_constraint(pred, group)

    for g in range(5):
        group_sum = float(corrected[group == g].sum())
        assert abs(group_sum) < 1e-8, f"expected group {g}'s corrected predictions to sum to ~0, got {group_sum}"


def test_group_zero_sum_constraint_respects_weights_and_target_sum():
    pred = np.array([1.0, 2.0, 3.0, 4.0])
    group = np.array([0, 0, 1, 1])
    weights = np.array([1.0, 1.0, 2.0, 1.0])

    corrected = apply_group_zero_sum_constraint(pred, group, weights=weights, target_sum=6.0)

    weighted_sum_g0 = float((corrected[:2] * weights[:2]).sum())
    weighted_sum_g1 = float((corrected[2:] * weights[2:]).sum())
    assert abs(weighted_sum_g0 - 6.0) < 1e-8
    assert abs(weighted_sum_g1 - 6.0) < 1e-8
