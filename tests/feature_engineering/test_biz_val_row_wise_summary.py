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
    """Helper: Make dispersion dataset."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, d)), columns=[f"f{i}" for i in range(d)])
    y = X.to_numpy().std(axis=1) * 5.0 + rng.normal(scale=0.2, size=n)
    return X, y


def test_biz_val_row_wise_summary_stats_beats_raw_columns_alone_mse():
    """Biz val row wise summary stats beats raw columns alone mse."""
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
    assert (
        improvement > 0.7
    ), f"expected >70% MSE reduction from row-wise summary features, got {improvement:.4f} (baseline={mse_baseline:.4f}, augmented={mse_augmented:.4f})"


def test_row_wise_summary_stats_output_shape_and_columns():
    """Row wise summary stats output shape and columns."""
    X, _ = _make_dispersion_dataset(n=50, d=5, seed=2)
    result = row_wise_summary_stats(X, stats=("mean", "std", "min", "max", "median", "q10"))
    assert result.shape[0] == 50
    assert set(result.columns) == {"row_summary_mean", "row_summary_std", "row_summary_min", "row_summary_max", "row_summary_median", "row_summary_q10"}


def test_row_wise_summary_stats_ignores_nan_within_row():
    """Row wise summary stats ignores nan within row."""
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [3.0, 2.0, 5.0], "c": [5.0, 4.0, 7.0]})
    result = row_wise_summary_stats(X, stats=("mean",))
    np.testing.assert_allclose(result["row_summary_mean"].to_numpy(), [3.0, 3.0, 5.0])


def test_row_wise_summary_stats_groups_default_none_is_bit_identical_to_prior_behavior():
    """Opt-in ``groups`` param must not change the default (flat) code path at all."""
    X, _ = _make_dispersion_dataset(n=80, d=12, seed=3)
    result = row_wise_summary_stats(X, stats=("mean", "std", "q10", "q50", "q90"))
    result_explicit_none = row_wise_summary_stats(X, stats=("mean", "std", "q10", "q50", "q90"), groups=None)
    pd.testing.assert_frame_equal(result, result_explicit_none)


def _make_multi_scale_dataset(n: int, d_signal: int, d_noise_scale: int, seed: int):
    """Two feature families with very different scales; target depends only on the SMALL-scale family's dispersion.

    signal_* columns: N(0, 1) -- their row-wise std carries the target signal.
    bigscale_* columns: N(0, 200) -- irrelevant to the target, but their huge scale dominates any FLAT
    (all-columns-together) mean/std, drowning out the signal family's dispersion.
    """
    rng = np.random.default_rng(seed)
    signal = rng.normal(scale=1.0, size=(n, d_signal))
    bigscale = rng.normal(scale=200.0, size=(n, d_noise_scale))
    cols_signal = [f"signal_{i}" for i in range(d_signal)]
    cols_bigscale = [f"bigscale_{i}" for i in range(d_noise_scale)]
    X = pd.DataFrame(np.hstack([signal, bigscale]), columns=cols_signal + cols_bigscale)
    y = signal.std(axis=1) * 5.0 + rng.normal(scale=0.2, size=n)
    return X, y, cols_signal, cols_bigscale


def test_biz_val_row_wise_summary_stats_grouped_beats_flat_when_scales_differ_mse():
    """Biz val row wise summary stats grouped beats flat when scales differ mse."""
    X, y, cols_signal, cols_bigscale = _make_multi_scale_dataset(n=400, d_signal=15, d_noise_scale=15, seed=10)
    rng = np.random.default_rng(11)
    perm = rng.permutation(len(y))
    train_idx, test_idx = perm[:250], perm[250:]

    flat_summary = row_wise_summary_stats(X, stats=("mean", "std", "q10", "q50", "q90"))
    X_flat = pd.concat([X, flat_summary], axis=1)
    flat_model = GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=3).fit(X_flat.iloc[train_idx], y[train_idx])
    mse_flat = mean_squared_error(y[test_idx], flat_model.predict(X_flat.iloc[test_idx]))

    grouped_summary = row_wise_summary_stats(X, stats=("mean", "std", "q10", "q50", "q90"), groups={"signal": cols_signal, "bigscale": cols_bigscale})
    X_grouped = pd.concat([X, grouped_summary], axis=1)
    grouped_model = GradientBoostingRegressor(random_state=0, n_estimators=100, max_depth=3).fit(X_grouped.iloc[train_idx], y[train_idx])
    mse_grouped = mean_squared_error(y[test_idx], grouped_model.predict(X_grouped.iloc[test_idx]))

    improvement = 1.0 - mse_grouped / mse_flat
    assert improvement > 0.5, (
        f"expected per-group summary stats to beat a flat (all-columns) summary by >50% MSE when families "
        f"have very different scales, got {improvement:.4f} (flat={mse_flat:.4f}, grouped={mse_grouped:.4f})"
    )


def test_row_wise_summary_stats_grouped_matches_manual_per_group_calls():
    """Row wise summary stats grouped matches manual per group calls."""
    X, _, cols_signal, cols_bigscale = _make_multi_scale_dataset(n=60, d_signal=6, d_noise_scale=4, seed=12)
    groups = {"signal": cols_signal, "bigscale": cols_bigscale}
    grouped = row_wise_summary_stats(X, stats=("mean", "std", "q10"), groups=groups)

    manual_parts = []
    for name, cols in groups.items():
        part = row_wise_summary_stats(X, columns=cols, stats=("mean", "std", "q10"), column_prefix=f"row_summary_{name}")
        manual_parts.append(part)
    manual = pd.concat(manual_parts, axis=1)

    pd.testing.assert_frame_equal(grouped, manual)
    assert set(grouped.columns) == {
        "row_summary_signal_mean",
        "row_summary_signal_std",
        "row_summary_signal_q10",
        "row_summary_bigscale_mean",
        "row_summary_bigscale_std",
        "row_summary_bigscale_q10",
    }


def test_row_wise_summary_stats_nan_path_matches_numpy_nanquantile_nanmedian():
    """The NaN-present path routes quantiles (and median, as q=0.5) through an njit per-row kernel instead of
    ``np.nanquantile``/``np.nanmedian`` (apply_along_axis, ~7x slower at n=200k -- see bench_row_wise_summary.py).
    Pin the njit path to agree with the exact numpy reference to within float64 rounding-order noise, on a
    block with NaN present, varying row-completeness (some rows all-finite, some with several NaN, one
    all-NaN row) so every branch of the per-row kernel (m==0, m==1, m>1) is exercised.
    """
    rng = np.random.default_rng(3)
    n, d = 500, 12
    arr = rng.normal(size=(n, d))
    arr[rng.random((n, d)) < 0.15] = np.nan
    arr[0, :] = np.nan  # all-NaN row
    arr[1, 1:] = np.nan  # single finite value row
    X = pd.DataFrame(arr, columns=[f"f{i}" for i in range(d)])

    out = row_wise_summary_stats(X, stats=("median", "q10", "q50", "q90"))

    ref_q = np.nanquantile(arr, [0.1, 0.5, 0.9], axis=1)
    ref_median = np.nanmedian(arr, axis=1)

    np.testing.assert_allclose(out["row_summary_q10"].to_numpy(), ref_q[0], rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(out["row_summary_q50"].to_numpy(), ref_q[1], rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(out["row_summary_q90"].to_numpy(), ref_q[2], rtol=0, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(out["row_summary_median"].to_numpy(), ref_median, rtol=0, atol=1e-12, equal_nan=True)
    assert np.isnan(out["row_summary_median"].to_numpy()[0])  # the all-NaN row stays NaN
