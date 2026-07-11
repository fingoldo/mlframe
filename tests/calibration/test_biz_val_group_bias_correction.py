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


def test_fit_group_bias_correction_shrinkage_k_none_is_bit_identical_to_prior_behavior():
    """Default (shrinkage_k=None) must reproduce the exact prior hard-cutoff ratios, unchanged."""
    rng = np.random.default_rng(1)
    n = 2000
    groups = rng.integers(0, 40, n).astype(str)
    y_true = rng.uniform(50, 150, n)
    y_pred = y_true * rng.uniform(0.6, 1.4, n)

    ratios_default = fit_group_bias_correction(y_true, y_pred, groups)
    ratios_explicit_none = fit_group_bias_correction(y_true, y_pred, groups, shrinkage_k=None)
    assert ratios_default == ratios_explicit_none

    # cross-check against the hand-rolled hard-cutoff formula for a few groups
    for g in ["0", "1", "2"]:
        mask = groups == g
        if mask.sum() >= 5:
            expected = float(np.clip(y_true[mask].mean() / y_pred[mask].mean(), 0.5, 2.0))
        else:
            expected = 1.0
        assert abs(ratios_default[g] - expected) < 1e-9


def _make_mixed_group_size_dataset(seed: int):
    """Large groups with REAL, stable bias; tiny groups with only NOISE (no true bias) that a naive
    per-group fit will overfit. Ground truth: large groups need correction; tiny groups should NOT be
    corrected away from 1.0 because their "bias" is pure validation noise.
    """
    rng = np.random.default_rng(seed)

    # 4 large groups, n=5000 each, with a real, stable +30%/-25% bias.
    large_groups = np.repeat(["big_A", "big_B", "big_C", "big_D"], 5000)
    large_bias = np.select([large_groups == "big_A", large_groups == "big_B", large_groups == "big_C", large_groups == "big_D"], [1.3, 0.75, 1.25, 0.8])
    large_true = rng.uniform(50, 150, len(large_groups))
    large_pred_val = large_true / large_bias + rng.normal(scale=1.0, size=len(large_groups))  # bias applied to true->pred

    # 60 tiny groups, n=4 each, model is UNBIASED for these (ratio should be ~1.0), but small-sample noise
    # produces spurious per-group ratios far from 1.0 on the validation slice.
    n_tiny_groups = 60
    tiny_n_each = 4
    tiny_groups = np.repeat([f"tiny_{i}" for i in range(n_tiny_groups)], tiny_n_each)
    tiny_true_val = rng.uniform(50, 150, len(tiny_groups))
    tiny_pred_val = tiny_true_val + rng.normal(scale=25.0, size=len(tiny_groups))  # noisy but UNBIASED model

    y_true_val = np.concatenate([large_true, tiny_true_val])
    y_pred_val = np.concatenate([large_pred_val, tiny_pred_val])
    group_val = np.concatenate([large_groups, tiny_groups])

    # independent held-out set drawn from the SAME generative process, to score generalization.
    large_true_ho = rng.uniform(50, 150, len(large_groups))
    large_pred_ho = large_true_ho / large_bias + rng.normal(scale=1.0, size=len(large_groups))
    tiny_true_ho = rng.uniform(50, 150, len(tiny_groups))
    tiny_pred_ho = tiny_true_ho + rng.normal(scale=25.0, size=len(tiny_groups))  # still unbiased on held-out

    y_true_ho = np.concatenate([large_true_ho, tiny_true_ho])
    y_pred_ho = np.concatenate([large_pred_ho, tiny_pred_ho])
    group_ho = np.concatenate([large_groups, tiny_groups])

    is_large_ho = np.concatenate([np.ones(len(large_groups), dtype=bool), np.zeros(len(tiny_groups), dtype=bool)])
    return (y_true_val, y_pred_val, group_val), (y_true_ho, y_pred_ho, group_ho, is_large_ho)


def test_biz_val_group_bias_correction_shrinkage_beats_naive_on_small_noisy_groups():
    (y_true_val, y_pred_val, group_val), (y_true_ho, y_pred_ho, group_ho, is_large_ho) = _make_mixed_group_size_dataset(seed=42)

    # naive: min_group_size=1 so every tiny group (n=4) still gets its own noisy raw ratio -- the failure mode.
    ratios_naive = fit_group_bias_correction(y_true_val, y_pred_val, group_val, min_group_size=1)
    corrected_naive = apply_group_bias_correction(y_pred_ho, group_ho, ratios_naive)

    ratios_shrunk = fit_group_bias_correction(y_true_val, y_pred_val, group_val, shrinkage_k=20.0)
    corrected_shrunk = apply_group_bias_correction(y_pred_ho, group_ho, ratios_shrunk)

    mae_naive_tiny = float(np.mean(np.abs(y_true_ho[~is_large_ho] - corrected_naive[~is_large_ho])))
    mae_shrunk_tiny = float(np.mean(np.abs(y_true_ho[~is_large_ho] - corrected_shrunk[~is_large_ho])))
    mae_naive_large = float(np.mean(np.abs(y_true_ho[is_large_ho] - corrected_naive[is_large_ho])))
    mae_shrunk_large = float(np.mean(np.abs(y_true_ho[is_large_ho] - corrected_shrunk[is_large_ho])))

    # shrinkage must meaningfully reduce held-out error on tiny/noisy groups (avoids overfitting validation noise)...
    assert mae_shrunk_tiny < mae_naive_tiny * 0.85, f"expected shrinkage to reduce held-out error on tiny noisy groups, got shrunk={mae_shrunk_tiny:.2f} naive={mae_naive_tiny:.2f}"
    # ...while matching (not materially degrading) accuracy on large groups with real, well-estimated bias.
    assert mae_shrunk_large < mae_naive_large * 1.1, f"expected shrinkage to match naive correction on large groups with real bias, got shrunk={mae_shrunk_large:.2f} naive={mae_naive_large:.2f}"
