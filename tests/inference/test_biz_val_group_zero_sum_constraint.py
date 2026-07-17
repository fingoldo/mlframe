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
    group_weighted_mean_signal = (np.bincount(group_ids, weights=raw_signal * weights) / np.bincount(group_ids, weights=weights))[group_ids]
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


def test_biz_val_apply_group_zero_sum_constraint_multi_constraint_projection():
    """A second known linear identity (e.g. a control-subset total) must hold simultaneously with zero-sum.

    Single-constraint correction re-satisfies the zero-sum identity but leaves the control-subset total badly
    violated (the two constraints overlap on the same rows); ``extra_constraint_coefs``/``extra_constraint_targets``
    project onto BOTH simultaneously via one constrained weighted-least-squares solve.
    """
    rng = np.random.default_rng(123)
    n_groups = 400
    members_per_group = 8
    n = n_groups * members_per_group

    group_ids = np.repeat(np.arange(n_groups), members_per_group)
    weights = rng.uniform(0.6, 1.4, n)
    # first 3 members of every group form a "control" subset with a known external total.
    core_coefs = np.tile(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), n_groups)
    control_target = 0.4

    preds = rng.normal(0, 1, n)

    single_corrected = apply_group_zero_sum_constraint(preds, group_ids, weights=weights, target_sum=0.0)
    multi_corrected = apply_group_zero_sum_constraint(
        preds, group_ids, weights=weights, target_sum=0.0, extra_constraint_coefs=[core_coefs], extra_constraint_targets=[control_target]
    )

    zero_sum_resid_single = np.bincount(group_ids, weights=single_corrected * weights, minlength=n_groups)
    core_resid_single = np.bincount(group_ids, weights=single_corrected * core_coefs, minlength=n_groups) - control_target
    zero_sum_resid_multi = np.bincount(group_ids, weights=multi_corrected * weights, minlength=n_groups)
    core_resid_multi = np.bincount(group_ids, weights=multi_corrected * core_coefs, minlength=n_groups) - control_target

    # single-constraint correction satisfies zero-sum but leaves the control-subset constraint badly violated.
    assert np.abs(zero_sum_resid_single).max() < 1e-8
    assert np.abs(core_resid_single).max() > 3.0, "single-constraint correction should badly violate the un-modeled control-subset identity"

    # the multi-constraint projection satisfies BOTH simultaneously, to float precision.
    assert np.abs(zero_sum_resid_multi).max() < 1e-8
    assert np.abs(core_resid_multi).max() < 1e-8


def test_biz_val_apply_group_zero_sum_constraint_preserve_rank_order_reduces_violation():
    """``preserve_rank_order`` cuts total within-group rank-inversion magnitude vs. the plain multi-constraint solve.

    A subset-total constraint necessarily shifts only the subset's rows, which can reorder them relative to
    non-subset rows -- ``preserve_rank_order=True`` pulls the result back towards ``preds``' rank order via an
    isotonic re-fit + exact re-projection while still satisfying both constraints exactly.
    """
    rng = np.random.default_rng(123)
    n_groups = 400
    members_per_group = 8
    n = n_groups * members_per_group

    group_ids = np.repeat(np.arange(n_groups), members_per_group)
    weights = rng.uniform(0.6, 1.4, n)
    core_coefs = np.tile(np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), n_groups)
    control_target = 0.4
    preds = rng.normal(0, 1, n)

    kwargs = dict(weights=weights, target_sum=0.0, extra_constraint_coefs=[core_coefs], extra_constraint_targets=[control_target])
    multi_corrected = apply_group_zero_sum_constraint(preds, group_ids, **kwargs)
    preserved_corrected = apply_group_zero_sum_constraint(preds, group_ids, preserve_rank_order=True, **kwargs)

    def total_violation_depth(corrected: np.ndarray) -> float:
        total = 0.0
        for g in range(n_groups):
            member_mask = group_ids == g
            order = np.argsort(preds[member_mask], kind="stable")
            diffs = np.diff(corrected[member_mask][order])
            total += float(np.sum(np.clip(-diffs, 0, None)))
        return total

    violation_multi = total_violation_depth(multi_corrected)
    violation_preserved = total_violation_depth(preserved_corrected)
    assert violation_preserved < violation_multi * 0.7, (
        f"preserve_rank_order should cut total inversion magnitude by >=30%: preserved={violation_preserved:.2f} multi={violation_multi:.2f}"
    )

    # both variants must still satisfy both constraints exactly -- rank preservation is best-effort, exactness is not.
    for corrected in (multi_corrected, preserved_corrected):
        zero_sum_resid = np.bincount(group_ids, weights=corrected * weights, minlength=n_groups)
        core_resid = np.bincount(group_ids, weights=corrected * core_coefs, minlength=n_groups) - control_target
        assert np.abs(zero_sum_resid).max() < 1e-8
        assert np.abs(core_resid).max() < 1e-8
