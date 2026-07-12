"""biz_value test for ``ordered_target_encode``'s ``noise_count_halflife`` count-decayed noise schedule.

Source: bestpractice_coursera-how-to-win-notes.md regularized-target-encoding entry. The pre-existing
``noise_std`` applies the SAME relative noise magnitude to every training row regardless of how many prior
observations that row's category has already accumulated -- but a category's expanding-mean statistic is a
noisy point estimate only while its running count is small; once a category has many prior observations the
expanding mean is already stable and constant-magnitude noise over-regularizes it for no reason. This test
proves a count-decayed noise schedule (heavy noise on low-count rows, tapering as running count grows) beats
constant noise on a synthetic mixing very-low-count and very-high-count categories: it should shrink the
overfitting gap on the low-count categories (like constant noise does) WITHOUT hurting the high-count
categories' predictive fit as much as constant noise does.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode


def _make_mixed_cardinality_data(n_low: int, low_categories: int, n_high: int, high_categories: int, seed: int):
    """Two category pools sharing one y-generating process: low-count (noisy stats) and high-count (stable stats).

    Both pools carry a REAL category effect (category id maps to a mean shift) so that a high-count category's
    expanding-mean estimate genuinely converges to a useful, low-variance signal that constant noise degrades.
    """
    rng = np.random.default_rng(seed)

    low_ids = rng.integers(0, low_categories, n_low)
    low_effect = rng.normal(scale=1.0, size=low_categories)[low_ids]
    low_y = low_effect + rng.normal(scale=0.3, size=n_low)

    high_ids = rng.integers(0, high_categories, n_high) + low_categories  # disjoint id space from the low pool.
    high_effect = rng.normal(scale=1.0, size=high_categories)[high_ids - low_categories]
    high_y = high_effect + rng.normal(scale=0.3, size=n_high)

    cats = np.concatenate([low_ids, high_ids])
    y = np.concatenate([low_y, high_y])
    order = np.arange(n_low + n_high)

    is_low = np.concatenate([np.ones(n_low, dtype=bool), np.zeros(n_high, dtype=bool)])
    return cats, order, y, is_low


def test_biz_val_ordered_target_encode_noise_count_halflife_beats_constant_noise_on_mixed_cardinality():
    cats, order, y, is_low = _make_mixed_cardinality_data(n_low=800, low_categories=700, n_high=4000, high_categories=40, seed=7)
    train_idx = np.arange(0, int(0.75 * len(y)))
    test_idx = np.arange(int(0.75 * len(y)), len(y))
    test_is_low = is_low[test_idx]

    def _fit_and_high_count_test_rmse(noise_std: float, noise_count_halflife) -> float:
        enc = ordered_target_encode(
            cats, y, order=order, smoothing=1.0, noise_std=noise_std, noise_count_halflife=noise_count_halflife, random_state=3
        ).reshape(-1, 1)
        model = LGBMRegressor(n_estimators=150, num_leaves=31, min_child_samples=5, random_state=0, verbose=-1)
        model.fit(enc[train_idx], y[train_idx])
        preds = model.predict(enc[test_idx])
        # RMSE restricted to the HIGH-count categories in the test set: constant noise over-regularizes their
        # already-stable, genuinely-informative expanding-mean statistic; the decayed schedule should not.
        high_mask = ~test_is_low
        return float(mean_squared_error(y[test_idx][high_mask], preds[high_mask]) ** 0.5)

    rmse_constant = _fit_and_high_count_test_rmse(noise_std=0.9, noise_count_halflife=None)
    rmse_scheduled = _fit_and_high_count_test_rmse(noise_std=0.9, noise_count_halflife=5.0)
    rmse_no_noise = _fit_and_high_count_test_rmse(noise_std=0.0, noise_count_halflife=None)

    assert rmse_scheduled < rmse_constant * 0.90, (
        f"expected count-decayed noise to cut high-count-category test RMSE by >=10% vs constant noise, "
        f"got scheduled={rmse_scheduled:.4f} constant={rmse_constant:.4f}"
    )
    # the schedule should also stay close to the no-noise baseline on the high-count categories (it decayed
    # noise away for them), not just "less bad than constant" -- within 15% relative of the un-noised fit.
    assert rmse_scheduled < rmse_no_noise * 1.15, (
        f"expected scheduled noise to stay near the no-noise baseline on high-count categories, "
        f"got scheduled={rmse_scheduled:.4f} no_noise={rmse_no_noise:.4f}"
    )


def test_ordered_target_encode_noise_count_halflife_none_is_bit_identical_to_omitting_param():
    rng = np.random.default_rng(0)
    n, n_categories = 300, 60
    cats = rng.integers(0, n_categories, n)
    order = np.arange(n)
    y = rng.normal(size=n)

    omitted = ordered_target_encode(cats, y, order=order, smoothing=1.0, noise_std=0.4, random_state=11)
    explicit_none = ordered_target_encode(cats, y, order=order, smoothing=1.0, noise_std=0.4, noise_count_halflife=None, random_state=11)
    np.testing.assert_array_equal(omitted, explicit_none)


def test_ordered_target_encode_noise_count_halflife_decays_toward_zero_noise_for_high_count_rows():
    # A single category seen many times in a row with a constant NONZERO target: the unnoised expanding mean
    # converges to that constant quickly, so any residual spread in the noised encoding at high running counts
    # is entirely due to the multiplicative noise term -- which should shrink toward zero as the schedule decays.
    n = 500
    cats = np.zeros(n, dtype=int)
    order = np.arange(n)
    y = np.full(n, 5.0)

    enc_a = ordered_target_encode(cats, y, order=order, smoothing=1.0, noise_std=1.0, noise_count_halflife=2.0, random_state=1)
    enc_b = ordered_target_encode(cats, y, order=order, smoothing=1.0, noise_std=1.0, noise_count_halflife=2.0, random_state=2)

    # by the tail, running_count >> halflife -> effective noise std ~0 -> both seeds converge to the same value.
    tail_a, tail_b = enc_a[-50:], enc_b[-50:]
    np.testing.assert_allclose(tail_a, np.full_like(tail_a, 5.0), atol=1e-6)
    np.testing.assert_allclose(tail_b, np.full_like(tail_b, 5.0), atol=1e-6)

    # early rows (small running_count, noise_count_halflife=2.0) should show real seed-to-seed divergence --
    # confirms the schedule is actually applying non-trivial noise there, not just always ~0.
    early_a, early_b = enc_a[1:6], enc_b[1:6]
    assert np.abs(early_a - early_b).max() > 0.05
