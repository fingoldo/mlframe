"""biz_value test for ``inference.apply_group_zero_sum_constraint``.

The win: when the target has a TRUE known zero-sum (or known-constant-sum) identity within groups but
independent per-row prediction noise violates it, enforcing the constraint removes the noise's
constraint-violating component and produces a strictly lower RMSE against the true target than the raw
(unconstrained) predictions -- exactly the effect the Optiver writeup exploited.
"""
from __future__ import annotations

import numpy as np

from mlframe.inference.group_zero_sum_constraint import apply_group_zero_sum_constraint


def test_biz_val_group_zero_sum_constraint_reduces_rmse_and_enforces_identity():
    rng = np.random.default_rng(0)
    n_groups = 500
    members_per_group = 10
    n = n_groups * members_per_group

    group_ids = np.repeat(np.arange(n_groups), members_per_group)
    weights = rng.uniform(0.5, 2.0, n)

    # true target satisfies the weighted zero-sum identity exactly within each group by construction.
    raw_signal = rng.normal(0, 1, n)
    group_weighted_mean_signal = (
        np.bincount(group_ids, weights=raw_signal * weights) / np.bincount(group_ids, weights=weights)
    )[group_ids]
    true_y = raw_signal - group_weighted_mean_signal  # exactly zero-sum per group now

    preds = true_y + rng.normal(0, 0.5, n)  # independent noise breaks the identity

    corrected = apply_group_zero_sum_constraint(preds, group_ids, weights=weights)

    raw_rmse = float(np.sqrt(np.mean((preds - true_y) ** 2)))
    corrected_rmse = float(np.sqrt(np.mean((corrected - true_y) ** 2)))
    assert corrected_rmse < raw_rmse, f"the constraint should reduce RMSE against the true target: corrected={corrected_rmse:.4f} raw={raw_rmse:.4f}"

    # the identity itself should now hold exactly (up to floating point) in every group.
    group_weighted_sums = np.bincount(group_ids, weights=corrected * weights)
    assert np.allclose(group_weighted_sums, 0.0, atol=1e-9)


def test_group_zero_sum_constraint_nonzero_target_sum():
    preds = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
    group_ids = np.array([0, 0, 0, 1, 1])
    corrected = apply_group_zero_sum_constraint(preds, group_ids, target_sum=6.0)
    assert np.isclose(corrected[:3].sum(), 6.0)
    assert np.isclose(corrected[3:].sum(), 6.0)


def test_group_zero_sum_constraint_shape_mismatch_raises():
    import pytest

    with pytest.raises(ValueError):
        apply_group_zero_sum_constraint(np.array([1.0, 2.0]), np.array([0, 0, 1]))
