"""biz_value test for ``feature_engineering.row_wise_summary_stats``.

The win: when the target depends on the DISPERSION (std) across a block of otherwise-uninformative feature
columns, a tree model trained on the raw columns alone must implicitly reconstruct cross-column spread from
individual per-column splits -- a genuinely harder learning problem than being handed the row-wise std
directly. Adding row-wise summary-statistic columns (mean/std/quantiles) should recover the true signal far
better, mirroring the Ubiquant Market Prediction 2nd place's per-row cross-sectional "macro" features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from mlframe.feature_engineering import row_wise_summary_stats


def _make_dispersion_dataset(n: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    y = X.to_numpy().std(axis=1) * 5.0 + rng.normal(scale=0.2, size=n)
    return X, y


def test_biz_val_row_wise_summary_stats_beats_raw_columns_alone_mse():
    X, y = _make_dispersion_dataset(n=400, d=30, seed=0)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(y))
    train_idx, test_idx = perm[:250], perm[250:]

    baseline = GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=3).fit(X.iloc[train_idx], y[train_idx])
    mse_baseline = mean_squared_error(y[test_idx], baseline.predict(X.iloc[test_idx]))

    summary = row_wise_summary_stats(X, stats=("mean", "std", "q10", "q50", "q90"))
    X_augmented = pd.concat([X, summary], axis=1)
    augmented = GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=3).fit(X_augmented.iloc[train_idx], y[train_idx])
    mse_augmented = mean_squared_error(y[test_idx], augmented.predict(X_augmented.iloc[test_idx]))

    improvement = 1.0 - mse_augmented / mse_baseline
    assert improvement > 0.7, f"expected >70% MSE reduction from row-wise summary features, got {improvement:.4f} (baseline={mse_baseline:.4f}, augmented={mse_augmented:.4f})"


def test_row_wise_summary_stats_output_shape_and_columns():
    X, _ = _make_dispersion_dataset(n=50, d=5, seed=2)
    result = row_wise_summary_stats(X, stats=("mean", "std", "min", "max", "median", "q10"))
    assert result.shape[0] == 50
    assert set(result.columns) == {"row_summary_mean", "row_summary_std", "row_summary_min", "row_summary_max", "row_summary_median", "row_summary_q10"}


def test_row_wise_summary_stats_ignores_nan_within_row():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [3.0, 2.0, 5.0], "c": [5.0, 4.0, 7.0]})
    result = row_wise_summary_stats(X, stats=("mean",))
    np.testing.assert_allclose(result["row_summary_mean"].to_numpy(), [3.0, 3.0, 5.0])
