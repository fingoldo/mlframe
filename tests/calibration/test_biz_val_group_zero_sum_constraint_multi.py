"""biz_value test for ``calibration.group_zero_sum_constraint.apply_group_zero_sum_constraint_multi``.

Extends the single-constraint Optiver zero-sum trick to the realistic case where TWO independent groupings
must each sum (weighted) to zero simultaneously -- e.g. a target that is zero-sum both across a time grouping
AND across an orthogonal entity grouping (a double-centered residual, as in two-way ANOVA-style targets).
Applying the single-constraint correction on only ONE grouping re-satisfies that grouping but re-breaks the
other (the two partitions overlap), so this test demonstrates that concretely with numbers, then confirms the
multi-constraint (Dykstra alternating-projection) helper converges BOTH constraints simultaneously while still
cutting RMSE versus the raw uncorrected predictions.
"""
from __future__ import annotations

import numpy as np

from mlframe.calibration.group_zero_sum_constraint import apply_group_zero_sum_constraint, apply_group_zero_sum_constraint_multi


def _make_double_zero_sum_target_with_biased_predictions(n_a: int, n_b: int, seed: int):
    rng = np.random.default_rng(seed)
    n = n_a * n_b
    group_a = np.repeat(np.arange(n_a), n_b)  # e.g. date_id
    group_b = np.tile(np.arange(n_b), n_a)  # e.g. an orthogonal entity key

    raw = rng.normal(size=(n_a, n_b))
    doubly_centered = raw - raw.mean(axis=1, keepdims=True) - raw.mean(axis=0, keepdims=True) + raw.mean()
    true_y = doubly_centered.ravel()  # exact zero-sum along BOTH axes, by construction.

    bias_a = np.repeat(rng.normal(scale=0.3, size=n_a), n_b)
    bias_b = np.tile(rng.normal(scale=0.3, size=n_b), n_a)
    pred = true_y + rng.normal(scale=0.15, size=n) + bias_a + bias_b
    return pred, true_y, group_a, group_b


def _max_abs_group_sum(values: np.ndarray, group: np.ndarray, n_groups: int) -> float:
    return float(max(abs(values[group == g].sum()) for g in range(n_groups)))


def test_single_constraint_leaves_other_grouping_violated():
    pred, _, group_a, group_b = _make_double_zero_sum_target_with_biased_predictions(n_a=30, n_b=25, seed=0)

    corrected_a_only = apply_group_zero_sum_constraint(pred, group_a)

    residual_a = _max_abs_group_sum(corrected_a_only, group_a, n_groups=30)
    residual_b = _max_abs_group_sum(corrected_a_only, group_b, n_groups=25)

    assert residual_a < 1e-8, f"expected the fixed grouping to be satisfied, got residual={residual_a}"
    assert residual_b > 0.5, f"expected the OTHER grouping to remain badly violated after single-constraint apply, got residual={residual_b}"


def test_biz_val_group_zero_sum_constraint_multi_satisfies_both_and_reduces_rmse():
    pred, true_y, group_a, group_b = _make_double_zero_sum_target_with_biased_predictions(n_a=30, n_b=25, seed=0)

    corrected = apply_group_zero_sum_constraint_multi(pred, groups=[group_a, group_b])

    residual_a = _max_abs_group_sum(corrected, group_a, n_groups=30)
    residual_b = _max_abs_group_sum(corrected, group_b, n_groups=25)
    assert residual_a < 1e-6, f"expected grouping A constraint satisfied within 1e-6, got {residual_a}"
    assert residual_b < 1e-6, f"expected grouping B constraint satisfied within 1e-6, got {residual_b}"

    rmse_raw = float(np.sqrt(np.mean((pred - true_y) ** 2)))
    rmse_corrected = float(np.sqrt(np.mean((corrected - true_y) ** 2)))
    assert rmse_corrected < rmse_raw * 0.7, f"expected multi-constraint correction to cut RMSE by >=30%, got corrected={rmse_corrected:.4f} raw={rmse_raw:.4f}"


def test_group_zero_sum_constraint_multi_no_groups_is_identity():
    pred = np.array([1.0, 2.0, 3.0])
    corrected = apply_group_zero_sum_constraint_multi(pred, groups=[])
    assert np.allclose(corrected, pred)
