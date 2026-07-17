"""biz_value test for ``preprocessing.degradation_augment.augment_to_match_test_distribution``.

The win: training measurements are systematically cleaner (lower measurement noise) than the deployed/test
regime (common when training data comes from a well-instrumented pilot period and production sensors are
noisier). An ordinary-least-squares model fit on clean features is the classic "errors-in-variables" setup:
at test time, noisy inputs mean the OLS coefficients (fit assuming clean inputs) are no longer the
noise-aware optimum -- the theoretically correct predictor under measurement noise shrinks coefficients
toward zero in proportion to the noise variance (attenuation bias / regression-calibration / SIMEX
correction), which is exactly what training on noise-matched-to-test AUGMENTED copies induces automatically
(training with injected Gaussian input noise is a well-known exact equivalent of Tikhonov/L2 shrinkage on
the fitted weights, proportional to the injected noise variance). This makes the mechanism deterministic
enough to be robust seed-to-seed, unlike a noise-vs-decision-boundary interaction in a nonparametric model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mlframe.preprocessing.degradation_augment import augment_to_match_test_distribution, match_missingness_rate, match_noise_level


def _make_clean_train_noisy_test(seed: int):
    """Helper that make clean train noisy test."""
    rng = np.random.default_rng(seed)
    n_train, n_test = 500, 400

    def _make(n, noise_std):
        """Helper that make."""
        true_x1 = rng.normal(size=n)
        true_x2 = rng.normal(size=n)
        y = 2.0 * true_x1 + 1.5 * true_x2 + rng.normal(scale=0.3, size=n)
        x1 = true_x1 + rng.normal(scale=noise_std, size=n)
        x2 = true_x2 + rng.normal(scale=noise_std, size=n)
        df = pd.DataFrame({"x1": x1, "x2": x2})
        return df, y

    X_train, y_train = _make(n_train, noise_std=0.05)
    X_test, y_test = _make(n_test, noise_std=1.2)
    return X_train, y_train, X_test, y_test


def _fit_and_score(X_train, y_train, X_test, y_test) -> float:
    """Helper that fit and score."""
    reg = LinearRegression().fit(X_train, y_train)
    return float(mean_squared_error(y_test, reg.predict(X_test)))


def test_biz_val_degradation_augment_reduces_mse_under_measurement_noise_mismatch():
    """Degradation augment reduces mse under measurement noise mismatch."""
    rel_improvements = []
    for seed in range(5):
        X_train, y_train, X_test, y_test = _make_clean_train_noisy_test(seed=seed)
        mse_baseline = _fit_and_score(X_train, y_train, X_test, y_test)
        X_aug, y_aug = augment_to_match_test_distribution(X_train, y_train, X_test, degradation_fns=(match_noise_level,), n_augments=10, random_state=42 + seed)
        mse_augmented = _fit_and_score(X_aug, y_aug, X_test, y_test)
        rel_improvements.append((mse_baseline - mse_augmented) / mse_baseline)

    mean_rel_improvement = float(np.mean(rel_improvements))
    assert mean_rel_improvement > 0.45, (
        f"expected >45% mean MSE reduction from noise-matching augmentation across 5 seeds, got {mean_rel_improvement:.4f} (per-seed: {rel_improvements})"
    )
    assert min(rel_improvements) > 0, f"expected a positive MSE reduction on every seed, got {rel_improvements}"


def test_match_missingness_rate_raises_train_rate_toward_test_rate():
    """Match missingness rate raises train rate toward test rate."""
    rng = np.random.default_rng(1)
    X_train = pd.DataFrame(rng.normal(size=(500, 3)), columns=["a", "b", "c"])
    X_test = X_train.copy()
    test_mask = rng.random(X_test.shape) < 0.4
    X_test = X_test.mask(test_mask)

    degraded = match_missingness_rate(X_train, X_test, np.random.default_rng(2))
    for col in X_train.columns:
        train_rate = float(X_train[col].isna().mean())
        degraded_rate = float(degraded[col].isna().mean())
        test_rate = float(X_test[col].isna().mean())
        assert degraded_rate > train_rate
        assert abs(degraded_rate - test_rate) < 0.05


def test_match_noise_level_raises_train_std_toward_test_std():
    """Match noise level raises train std toward test std."""
    rng = np.random.default_rng(3)
    X_train = pd.DataFrame(rng.normal(scale=0.1, size=(2000, 2)), columns=["a", "b"])
    X_test = pd.DataFrame(rng.normal(scale=1.0, size=(2000, 2)), columns=["a", "b"])

    degraded = match_noise_level(X_train, X_test, np.random.default_rng(4))
    for col in X_train.columns:
        train_std = float(X_train[col].std())
        degraded_std = float(degraded[col].std())
        test_std = float(X_test[col].std())
        assert degraded_std > train_std
        assert abs(degraded_std - test_std) < 0.15
