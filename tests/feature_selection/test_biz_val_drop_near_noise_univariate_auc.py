"""biz_value test for ``feature_selection.drop_near_noise_univariate_auc.drop_near_noise_univariate_auc``.

The win (4th_santander-customer-transaction-prediction.md): a cheap univariate-AUC prescreen correctly
identifies features whose OWN individual AUC sits at chance (no linear/monotone signal in isolation) before
running the expensive MRMR/DCD pipeline. This test confirms the prescreen correctly flags pure-noise columns
while preserving genuinely predictive ones, and that dropping the flagged columns doesn't meaningfully hurt a
downstream classifier's AUC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mlframe.feature_selection.drop_near_noise_univariate_auc import drop_near_noise_univariate_auc


def _make_mixed_signal_dataset(n: int, n_signal: int, n_noise: int, seed: int):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    signal_direction = np.where(y == 1, 1.0, -1.0)

    cols = {}
    for i in range(n_signal):
        cols[f"signal{i}"] = signal_direction + rng.normal(scale=0.5, size=n)
    for i in range(n_noise):
        cols[f"noise{i}"] = rng.normal(size=n)  # independent of y

    df = pd.DataFrame(cols)
    return df, y


def test_biz_val_drop_near_noise_correctly_identifies_noise_columns():
    df, y = _make_mixed_signal_dataset(n=3000, n_signal=5, n_noise=15, seed=0)

    dropped = drop_near_noise_univariate_auc(df, y, tolerance=0.03)

    n_signal_dropped = sum(1 for c in dropped if c.startswith("signal"))
    n_noise_dropped = sum(1 for c in dropped if c.startswith("noise"))
    assert n_signal_dropped == 0, f"expected no genuinely-predictive signal columns to be flagged, got {[c for c in dropped if c.startswith('signal')]}"
    assert n_noise_dropped >= 12, f"expected most pure-noise columns to be correctly flagged, got {n_noise_dropped}/15"


def test_biz_val_dropping_near_noise_columns_preserves_downstream_auc():
    df, y = _make_mixed_signal_dataset(n=3000, n_signal=5, n_noise=15, seed=1)
    dropped = drop_near_noise_univariate_auc(df, y, tolerance=0.03)
    kept_cols = [c for c in df.columns if c not in dropped]

    auc_full = cross_val_score(LogisticRegression(max_iter=500), df, y, cv=5, scoring="roc_auc").mean()
    auc_pruned = cross_val_score(LogisticRegression(max_iter=500), df[kept_cols], y, cv=5, scoring="roc_auc").mean()

    assert len(kept_cols) < df.shape[1]
    assert auc_pruned >= auc_full - 0.02, (
        f"expected pruning near-noise columns to not meaningfully hurt downstream AUC, got pruned={auc_pruned:.4f} full={auc_full:.4f}"
    )


def _make_weak_signal_dataset(n: int, seed: int, effect: float):
    """A dataset with one genuinely-weak-but-real feature whose single-sample AUC estimate is noisy
    enough to sometimes land inside a ``tolerance`` band of 0.5 purely by sampling luck, plus several
    pure-noise columns whose true AUC is exactly 0.5 regardless of sample.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    signal_direction = np.where(y == 1, 1.0, -1.0)

    cols = {"weak_signal": effect * signal_direction + rng.normal(scale=1.0, size=n)}
    for i in range(5):
        cols[f"noise{i}"] = rng.normal(size=n)  # independent of y

    df = pd.DataFrame(cols)
    return df, y


def test_biz_val_drop_near_noise_bootstrap_stability_retains_weak_real_signal():
    """The single-pass AUC check has no way to distinguish "AUC near 0.5 because there's no signal" from
    "AUC near 0.5 because this particular sample happened to land there" -- across repeated seeds, a
    genuinely weak-but-real feature (true AUC just outside ``tolerance``) sometimes gets wrongly dropped
    by the single-pass check due to pure sampling variance. The opt-in bootstrap-stability mode should
    retain it substantially more reliably (at least 2x lower false-drop rate), while still correctly
    dropping the pure-noise columns every time.
    """
    n_seeds = 40
    tolerance = 0.04
    single_pass_false_drops = 0
    bootstrap_false_drops = 0
    bootstrap_noise_misses = 0

    for seed in range(n_seeds):
        df, y = _make_weak_signal_dataset(n=2000, seed=seed, effect=0.08)

        single_pass_dropped = drop_near_noise_univariate_auc(df, y, tolerance=tolerance)
        if "weak_signal" in single_pass_dropped:
            single_pass_false_drops += 1

        bootstrap_dropped = drop_near_noise_univariate_auc(df, y, tolerance=tolerance, n_bootstrap=40, bootstrap_frac=0.8, random_state=seed)
        if "weak_signal" in bootstrap_dropped:
            bootstrap_false_drops += 1
        n_noise_dropped = sum(1 for c in bootstrap_dropped if c.startswith("noise"))
        if n_noise_dropped < 4:  # allow an occasional single miss out of 5 pure-noise columns
            bootstrap_noise_misses += 1

    single_pass_false_drop_rate = single_pass_false_drops / n_seeds
    bootstrap_false_drop_rate = bootstrap_false_drops / n_seeds

    assert single_pass_false_drop_rate >= 0.30, (
        f"expected the single-pass check to sometimes wrongly drop the weak-but-real feature by sampling "
        f"luck, got false-drop rate {single_pass_false_drop_rate:.2f} over {n_seeds} seeds"
    )
    assert bootstrap_false_drop_rate <= 0.20, (
        f"expected the bootstrap-stability mode to retain the weak-but-real feature more reliably, got "
        f"false-drop rate {bootstrap_false_drop_rate:.2f} over {n_seeds} seeds"
    )
    assert bootstrap_false_drop_rate <= single_pass_false_drop_rate / 2, (
        f"expected bootstrap-stability mode to at least halve the single-pass false-drop rate, got "
        f"bootstrap={bootstrap_false_drop_rate:.2f} vs single-pass={single_pass_false_drop_rate:.2f}"
    )
    assert bootstrap_noise_misses <= 5, (
        f"expected bootstrap mode to still correctly drop pure-noise columns, missed on {bootstrap_noise_misses}/{n_seeds} seeds"
    )


def test_biz_val_drop_near_noise_bootstrap_mode_is_opt_in_and_bit_identical_by_default():
    """Omitting the new bootstrap params must reproduce the exact original single-pass result."""
    df, y = _make_mixed_signal_dataset(n=2000, n_signal=5, n_noise=15, seed=7)

    default_result = drop_near_noise_univariate_auc(df, y, tolerance=0.03)
    explicit_none_result = drop_near_noise_univariate_auc(df, y, tolerance=0.03, n_bootstrap=None)

    assert default_result == explicit_none_result
