"""biz_value test for ``ordered_target_encode``'s ``noise_std`` regularization parameter.

Source: bestpractice_coursera-how-to-win-notes.md -- "Mean (target) encoding with regularization (CV loops,
smoothing, expanding mean, adding noise) is a powerful technique." Even with a leak-free CAUSAL expanding-mean
encoding, a high-cardinality low-count category's encoding is a noisy point estimate a tree model can still
memorize as a near-unique fingerprint (e.g. a 2nd-occurrence row's encoding is literally the 1st occurrence's
raw target value) -- a spurious train-set correlation that doesn't generalize. Injecting relative Gaussian
noise into the encoding should shrink this train/test generalization gap on a synthetic where the category
has NO real relationship to the target (pure overfitting signal, isolated from any genuine category effect).
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode


def _make_noninformative_high_cardinality_data(n: int, n_categories: int, seed: int):
    rng = np.random.default_rng(seed)
    cats = rng.integers(0, n_categories, n)
    order = np.arange(n)
    y = rng.normal(scale=1.0, size=n)  # pure noise -- category carries no real signal about y.
    return cats, order, y


def test_biz_val_ordered_target_encode_noise_shrinks_overfitting_gap():
    cats, order, y = _make_noninformative_high_cardinality_data(n=1500, n_categories=750, seed=5)
    train_idx, test_idx = np.arange(0, 1100), np.arange(1100, 1500)

    def _fit_and_gap(noise_std: float) -> float:
        enc = ordered_target_encode(cats, y, order=order, smoothing=0.01, noise_std=noise_std, random_state=1).reshape(-1, 1)
        model = LGBMRegressor(n_estimators=100, num_leaves=63, min_child_samples=1, random_state=0, verbose=-1)
        model.fit(enc[train_idx], y[train_idx])
        train_rmse = float(mean_squared_error(y[train_idx], model.predict(enc[train_idx])) ** 0.5)
        test_rmse = float(mean_squared_error(y[test_idx], model.predict(enc[test_idx])) ** 0.5)
        return test_rmse - train_rmse

    gap_no_noise = _fit_and_gap(noise_std=0.0)
    gap_with_noise = _fit_and_gap(noise_std=0.8)

    assert gap_with_noise < gap_no_noise * 0.85, f"expected noise injection to shrink the train/test overfitting gap by >=15%, got with_noise={gap_with_noise:.4f} no_noise={gap_no_noise:.4f}"


def test_ordered_target_encode_noise_std_zero_is_deterministic_and_matches_baseline():
    cats, order, y = _make_noninformative_high_cardinality_data(n=200, n_categories=50, seed=0)
    baseline = ordered_target_encode(cats, y, order=order, smoothing=1.0)
    zero_noise = ordered_target_encode(cats, y, order=order, smoothing=1.0, noise_std=0.0, random_state=0)
    np.testing.assert_array_equal(baseline, zero_noise)


def test_ordered_target_encode_noise_is_reproducible_with_fixed_seed():
    cats, order, y = _make_noninformative_high_cardinality_data(n=200, n_categories=50, seed=0)
    enc_a = ordered_target_encode(cats, y, order=order, smoothing=1.0, noise_std=0.3, random_state=42)
    enc_b = ordered_target_encode(cats, y, order=order, smoothing=1.0, noise_std=0.3, random_state=42)
    np.testing.assert_array_equal(enc_a, enc_b)
