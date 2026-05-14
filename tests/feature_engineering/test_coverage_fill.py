"""Coverage-fill tests for feature_engineering. Goal: push every module to >=80% by exercising
the public-API branches the audit-fix and existing tests don't already cover.

Generated 2026-05-14 to close gaps identified by `coverage run -m pytest --cov`. The tests are
intentionally short and synthetic - they touch a branch with the cheapest input that lights it
up, not exhaustive correctness. Real correctness lives in test_audit_fixes.py / test_<file>.py.
"""
from __future__ import annotations

import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import polars as pl
import pytest


# ============================================================================
# _numerical_stable.py (138 missing -> aim 80%)
# ============================================================================


class TestNumericalStableKernels:
    def test_welford_mean_var_seq_basic(self):
        from mlframe.feature_engineering._numerical_stable import welford_mean_var_seq
        rng = np.random.default_rng(0)
        arr = rng.standard_normal(500).astype(np.float64)
        mean, var, n = welford_mean_var_seq(arr)
        assert n == 500
        np.testing.assert_allclose(mean, float(arr.mean()), rtol=1e-10)
        np.testing.assert_allclose(var, float(arr.var()), rtol=1e-10)

    def test_welford_mean_var_seq_with_nans(self):
        from mlframe.feature_engineering._numerical_stable import welford_mean_var_seq
        arr = np.array([1.0, np.nan, 2.0, np.inf, 3.0], dtype=np.float64)
        mean, var, n = welford_mean_var_seq(arr)
        # Non-finite skipped; finite vals are [1, 2, 3].
        assert n == 3
        np.testing.assert_allclose(mean, 2.0, atol=1e-12)

    def test_welford_mean_var_seq_empty(self):
        from mlframe.feature_engineering._numerical_stable import welford_mean_var_seq
        mean, var, n = welford_mean_var_seq(np.empty(0, dtype=np.float64))
        assert n == 0 and mean == 0.0 and var == 0.0

    def test_welford_moments_seq_basic(self):
        from mlframe.feature_engineering._numerical_stable import welford_moments_seq
        rng = np.random.default_rng(1)
        arr = rng.standard_normal(1000).astype(np.float64)
        mean, var, skew, kurt, n = welford_moments_seq(arr)
        assert n == 1000
        np.testing.assert_allclose(mean, float(arr.mean()), rtol=1e-10)
        assert abs(skew) < 0.3 and abs(kurt) < 0.5  # near-Gaussian

    def test_welford_moments_seq_short(self):
        from mlframe.feature_engineering._numerical_stable import welford_moments_seq
        mean, var, skew, kurt, n = welford_moments_seq(np.array([1.0], dtype=np.float64))
        assert n == 1 and var == 0.0 and skew == 0.0 and kurt == 0.0

    def test_welford_moments_seq_zero_var(self):
        from mlframe.feature_engineering._numerical_stable import welford_moments_seq
        mean, var, skew, kurt, n = welford_moments_seq(np.full(50, 7.0, dtype=np.float64))
        assert mean == 7.0 and skew == 0.0 and kurt == 0.0

    def test_kahan_sum_seq(self):
        from mlframe.feature_engineering._numerical_stable import kahan_sum_seq
        arr = np.array([1.0, np.nan, 2.0, np.inf, 3.0], dtype=np.float64)
        s = kahan_sum_seq(arr)
        np.testing.assert_allclose(s, 6.0, atol=1e-12)

    def test_kahan_dot_seq_basic(self):
        from mlframe.feature_engineering._numerical_stable import kahan_dot_seq
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        s = kahan_dot_seq(a, b)
        np.testing.assert_allclose(s, 32.0, atol=1e-12)

    def test_kahan_dot_seq_skips_non_finite(self):
        from mlframe.feature_engineering._numerical_stable import kahan_dot_seq
        a = np.array([1.0, np.nan, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
        s = kahan_dot_seq(a, b)
        np.testing.assert_allclose(s, 22.0, atol=1e-12)  # skip middle (NaN product)

    def test_kahan_dot_seq_length_mismatch(self):
        from mlframe.feature_engineering._numerical_stable import kahan_dot_seq
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = np.array([4.0, 5.0], dtype=np.float64)
        with pytest.raises((ValueError, Exception)):
            kahan_dot_seq(a, b)

    def test_naive_mean_var_two_pass_seq(self):
        from mlframe.feature_engineering._numerical_stable import naive_mean_var_two_pass_seq
        rng = np.random.default_rng(2)
        arr = rng.standard_normal(200).astype(np.float64)
        mean, var, n = naive_mean_var_two_pass_seq(arr)
        np.testing.assert_allclose(mean, float(arr.mean()), rtol=1e-10)
        np.testing.assert_allclose(var, float(arr.var()), rtol=1e-10)
        assert n == 200

    def test_naive_mean_var_two_pass_seq_empty(self):
        from mlframe.feature_engineering._numerical_stable import naive_mean_var_two_pass_seq
        mean, var, n = naive_mean_var_two_pass_seq(np.empty(0, dtype=np.float64))
        assert n == 0

    def test_kahan_two_pass_var_seq(self):
        from mlframe.feature_engineering._numerical_stable import kahan_two_pass_var_seq
        rng = np.random.default_rng(3)
        arr = rng.standard_normal(500).astype(np.float64)
        mean, var, n = kahan_two_pass_var_seq(arr)
        np.testing.assert_allclose(mean, float(arr.mean()), rtol=1e-10)
        np.testing.assert_allclose(var, float(arr.var()), rtol=1e-10)
        assert n == 500

    def test_kahan_two_pass_var_seq_empty(self):
        from mlframe.feature_engineering._numerical_stable import kahan_two_pass_var_seq
        mean, var, n = kahan_two_pass_var_seq(np.empty(0, dtype=np.float64))
        assert n == 0

    def test_naive_moments_two_pass_seq(self):
        from mlframe.feature_engineering._numerical_stable import naive_moments_two_pass_seq
        rng = np.random.default_rng(4)
        arr = rng.standard_normal(500).astype(np.float64)
        mean, var, skew, kurt, n = naive_moments_two_pass_seq(arr)
        assert n == 500
        np.testing.assert_allclose(mean, float(arr.mean()), rtol=1e-10)

    def test_naive_moments_two_pass_seq_short(self):
        from mlframe.feature_engineering._numerical_stable import naive_moments_two_pass_seq
        mean, var, skew, kurt, n = naive_moments_two_pass_seq(np.array([3.0], dtype=np.float64))
        assert n == 1 and var == 0.0

    def test_naive_moments_two_pass_seq_zero_var(self):
        from mlframe.feature_engineering._numerical_stable import naive_moments_two_pass_seq
        mean, var, skew, kurt, n = naive_moments_two_pass_seq(np.full(20, 4.2, dtype=np.float64))
        # Constant input -> mean preserved, var near zero, skew is 0 (deviations all zero).
        # Kurtosis on constant input is convention-dependent (excess-kurt = -3, biased = 0):
        # both implementations exist in the wild; we only assert the call doesn't crash.
        np.testing.assert_allclose(mean, 4.2, atol=1e-10)
        assert n == 20
        assert np.isfinite(skew) and np.isfinite(kurt)


# ============================================================================
# mps.py (224 missing -> aim 80%)
# ============================================================================


class TestMpsCoverage:
    def test_compute_area_profits_all_flat(self):
        from mlframe.feature_engineering.mps import compute_area_profits
        prices = np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float64)
        positions = np.zeros(4, dtype=np.int8)
        out = compute_area_profits(prices, positions)
        assert (out == 0.0).all()

    def test_compute_area_profits_run_extends_to_end(self):
        from mlframe.feature_engineering.mps import compute_area_profits
        prices = np.array([100.0, 101.0, 102.0, 103.0], dtype=np.float64)
        positions = np.array([1, 1, 1, 1], dtype=np.int8)
        out = compute_area_profits(prices, positions)
        # Run extends to final bar -> no closing price -> tail profits 0.
        assert (out == 0.0).all()

    def test_compute_area_profits_zero_price_guarded(self):
        from mlframe.feature_engineering.mps import compute_area_profits
        prices = np.array([100.0, 0.0, 102.0, 103.0], dtype=np.float64)
        positions = np.array([1, 1, -1, -1], dtype=np.int8)
        out = compute_area_profits(prices, positions)
        # Zero-price bar contributes 0 (no NaN/inf leaks).
        assert np.isfinite(out).all()

    def test_find_best_mps_sequence_short_input(self):
        from mlframe.feature_engineering.mps import find_best_mps_sequence
        prices = np.array([100.0], dtype=np.float64)
        positions, profits = find_best_mps_sequence(
            prices=prices, raw_prices=prices, tc=0.0, tc_mode_is_fraction=True
        )
        assert positions.size == 0 and profits.size == 0

    def test_find_best_mps_sequence_with_shift(self):
        from mlframe.feature_engineering.mps import find_best_mps_sequence
        prices = np.linspace(100, 110, 20)
        positions, _ = find_best_mps_sequence(
            prices=prices, raw_prices=prices, tc=0.0, tc_mode_is_fraction=True, shift=2
        )
        assert positions.size == 19  # n-1

    def test_find_best_mps_sequence_fixed_tc_mode(self):
        from mlframe.feature_engineering.mps import find_best_mps_sequence
        prices = np.linspace(100, 110, 30)
        positions, profits = find_best_mps_sequence(
            prices=prices, raw_prices=prices, tc=0.5, tc_mode_is_fraction=False
        )
        assert positions.size == 29

    def test_find_best_mps_sequence_no_optimize_regions(self):
        from mlframe.feature_engineering.mps import find_best_mps_sequence
        prices = np.linspace(100, 110, 20)
        positions, _ = find_best_mps_sequence(
            prices=prices, raw_prices=prices, tc=0.0, tc_mode_is_fraction=True,
            optimize_consecutive_regions=False,
        )
        assert positions.size == 19

    def test_backfill_zeros_right(self):
        from mlframe.feature_engineering.mps import backfill_zeros
        arr = np.array([0, 0, 1, 0, 0, -1, 0, 0], dtype=np.int8)
        out = backfill_zeros(arr, direction="right")
        assert list(out) == [1, 1, 1, -1, -1, -1, 0, 0]

    def test_backfill_zeros_left(self):
        from mlframe.feature_engineering.mps import backfill_zeros
        arr = np.array([0, 0, 1, 0, 0, -1, 0, 0], dtype=np.int8)
        out = backfill_zeros(arr, direction="left")
        assert list(out) == [0, 0, 1, 1, 1, -1, -1, -1]

    def test_find_maximum_profit_system_bad_tc_mode(self):
        from mlframe.feature_engineering.mps import find_maximum_profit_system
        with pytest.raises(ValueError):
            find_maximum_profit_system(np.array([100.0, 101.0]), tc_mode="weird")

    def test_find_maximum_profit_system_fixed_mode(self):
        from mlframe.feature_engineering.mps import find_maximum_profit_system
        prices = np.linspace(100, 110, 50)
        r = find_maximum_profit_system(prices, tc=0.5, tc_mode="fixed")
        assert "positions" in r and "profits" in r

    def test_find_maximum_profit_system_with_raw_prices(self):
        from mlframe.feature_engineering.mps import find_maximum_profit_system
        prices = np.linspace(100, 110, 30)
        raw = prices + 0.5
        r = find_maximum_profit_system(prices, raw_prices=raw)
        assert r["positions"].size == 29

    def test_generate_market_price(self):
        from mlframe.feature_engineering.mps import generate_market_price
        dates, prices, volumes = generate_market_price(n_days=50, random_seed=0)
        assert len(dates) == 50 and prices.shape == (50,) and volumes.shape == (50,)

    def test_safely_compute_mps_missing_file(self):
        from mlframe.feature_engineering.mps import safely_compute_mps
        res = safely_compute_mps("/non/existent/path.parquet")
        assert res is None

    def test_safely_compute_mps_caught_exception(self, tmp_path):
        from mlframe.feature_engineering.mps import safely_compute_mps
        bad = tmp_path / "bad.parquet"
        bad.write_bytes(b"not a parquet")
        res = safely_compute_mps(str(bad))
        assert res is None

    def test_compute_mps_targets_from_df(self):
        from mlframe.feature_engineering.mps import compute_mps_targets
        rng = np.random.default_rng(0)
        n = 30
        df = pl.DataFrame({
            "ts": list(range(n)) + list(range(n)),
            "secid": ["AAPL"] * n + ["MSFT"] * n,
            "pr_close": list(rng.uniform(100, 110, n)) + list(rng.uniform(200, 210, n)),
        })
        res = compute_mps_targets(fo_df=df, sma_size=3)
        assert res is not None and res.height > 0

    def test_compute_mps_targets_ewm_path(self):
        from mlframe.feature_engineering.mps import compute_mps_targets
        rng = np.random.default_rng(1)
        df = pl.DataFrame({
            "ts": list(range(20)),
            "secid": ["A"] * 20,
            "pr_close": rng.uniform(100, 110, 20),
        })
        res = compute_mps_targets(fo_df=df, sma_size=0, ewm_alpha=0.3)
        assert res is not None

    def test_show_mps_regions_no_chart(self):
        from mlframe.feature_engineering.mps import show_mps_regions
        prices = np.linspace(100, 110, 30)
        r = show_mps_regions(prices=prices, show_chart=False, tc=1e-4)
        assert "positions" in r

    def test_plot_positions_matplotlib(self):
        from mlframe.feature_engineering.mps import plot_positions
        import matplotlib
        matplotlib.use("Agg")
        prices = np.linspace(100, 110, 30).tolist()
        positions = [1] * 30
        fig = plot_positions(
            prices=prices, positions=positions, use_plotly=False, figsize=(4, 3),
        )
        assert fig is not None


# ============================================================================
# hurst.py (48 missing -> aim 80%)
# ============================================================================


class TestHurstCoverage:
    def test_precompute_hurst_short_returns_empty(self):
        from mlframe.feature_engineering.hurst import precompute_hurst_exponent
        sizes, rs = precompute_hurst_exponent(np.array([1.0, 2.0]), min_window=5)
        # too short -> empty arrays.
        assert len(sizes) == 0 and len(rs) == 0

    def test_precompute_hurst_max_window_le_min(self):
        from mlframe.feature_engineering.hurst import precompute_hurst_exponent
        sizes, rs = precompute_hurst_exponent(
            np.random.default_rng(0).standard_normal(50), min_window=10, max_window=5
        )
        assert len(sizes) == 0 and len(rs) == 0

    def test_compute_hurst_rs_short_returns_nan(self):
        from mlframe.feature_engineering.hurst import compute_hurst_rs
        rs = compute_hurst_rs(np.array([5.0]))
        assert np.isnan(rs)

    def test_compute_hurst_rs_constant_returns_nan(self):
        from mlframe.feature_engineering.hurst import compute_hurst_rs
        rs = compute_hurst_rs(np.ones(50, dtype=np.float64))
        assert np.isnan(rs)

    def test_compute_hurst_rs_real_window(self):
        from mlframe.feature_engineering.hurst import compute_hurst_rs
        rng = np.random.default_rng(0)
        rs = compute_hurst_rs(rng.standard_normal(100).astype(np.float64))
        assert np.isfinite(rs) and rs > 0

    def test_compute_hurst_exponent_take_diffs(self):
        from mlframe.feature_engineering.hurst import compute_hurst_exponent
        rng = np.random.default_rng(0)
        path = np.cumsum(rng.choice([-1.0, 1.0], size=500))
        h, c = compute_hurst_exponent(path, take_diffs=True)
        assert 0.3 < h < 0.7 and c > 0


# ============================================================================
# basic.py (26 missing -> aim 80%)
# ============================================================================


class TestBasicCoverage:
    def test_create_date_features_empty_cols(self):
        from mlframe.feature_engineering.basic import create_date_features
        df = pd.DataFrame({"x": [1, 2, 3]})
        out = create_date_features(df, cols=[])
        assert out is df

    def test_create_date_features_pandas(self):
        from mlframe.feature_engineering.basic import create_date_features
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01", "2024-02-15", "2024-03-20"])})
        out = create_date_features(df, cols=["ts"], delete_original_cols=False)
        assert "ts_day" in out.columns and "ts_weekday" in out.columns

    def test_create_date_features_polars(self):
        from mlframe.feature_engineering.basic import create_date_features
        df = pl.DataFrame({"ts": [datetime(2024, 1, 1), datetime(2024, 2, 15)]})
        out = create_date_features(df, cols=["ts"], delete_original_cols=True)
        assert "ts" not in out.columns and "ts_day" in out.columns

    def test_create_date_features_unsupported_backend(self):
        from mlframe.feature_engineering.basic import create_date_features
        with pytest.raises(ValueError):
            create_date_features({"foo": "bar"}, cols=["ts"])

    def test_create_date_features_unknown_method_pandas(self):
        from mlframe.feature_engineering.basic import create_date_features
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-01"])})
        with pytest.raises(ValueError):
            create_date_features(df, cols=["ts"], methods={"nonexistent_method": np.int8})

    def test_create_date_features_unsupported_dtype_polars(self):
        from mlframe.feature_engineering.basic import create_date_features
        df = pl.DataFrame({"ts": [datetime(2024, 1, 1)]})
        with pytest.raises(ValueError):
            create_date_features(df, cols=["ts"], methods={"day": np.float32})

    def test_create_date_features_clash_warning(self, caplog):
        from mlframe.feature_engineering.basic import create_date_features
        import logging
        df = pd.DataFrame({
            "ts": pd.to_datetime(["2024-01-01", "2024-02-15"]),
            "ts_day": [99, 99],
        })
        with caplog.at_level(logging.WARNING):
            create_date_features(df, cols=["ts"])
        assert any("OVERWRITTEN" in r.message for r in caplog.records)


# ============================================================================
# numerical.py - exercise many compute_numerical_aggregates_numba branches
# ============================================================================


class TestNumericalCoverage:
    def test_compute_numaggs_short_input_returns_nans(self):
        from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names
        # Single element -> all NaN sentinels of correct length.
        res = compute_numaggs(np.array([1.0]))
        names = get_numaggs_names()
        assert len(res) == len(names)
        assert all(np.isnan(v) for v in res)

    def test_compute_numaggs_with_weights(self):
        from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names
        rng = np.random.default_rng(0)
        arr = rng.standard_normal(100)
        weights = np.abs(rng.standard_normal(100)) + 0.1
        res = compute_numaggs(arr, weights=weights)
        names = get_numaggs_names(weights=weights)
        assert len(res) == len(names)

    def test_compute_numaggs_directional_only(self):
        from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names
        arr = np.linspace(1, 100, 100)
        res = compute_numaggs(arr, directional_only=True)
        names = get_numaggs_names(directional_only=True)
        assert len(res) == len(names)

    def test_compute_numaggs_with_drawdown_stats(self):
        from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names
        rng = np.random.default_rng(0)
        arr = 100 + np.cumsum(rng.standard_normal(200))
        res = compute_numaggs(arr, return_drawdown_stats=True)
        names = get_numaggs_names(return_drawdown_stats=True)
        assert len(res) == len(names)

    def test_compute_numaggs_with_profit_factor(self):
        from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names
        rng = np.random.default_rng(0)
        arr = rng.standard_normal(100) * 5
        res = compute_numaggs(arr, return_profit_factor=True)
        names = get_numaggs_names(return_profit_factor=True)
        assert len(res) == len(names)

    def test_compute_numaggs_with_distributional(self):
        from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names
        rng = np.random.default_rng(0)
        arr = rng.standard_normal(200)
        res = compute_numaggs(arr, return_distributional=True, return_hurst=False, return_entropy=False)
        names = get_numaggs_names(return_distributional=True, return_hurst=False, return_entropy=False)
        assert len(res) == len(names)

    def test_compute_numaggs_with_lintrend_approx(self):
        from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names
        rng = np.random.default_rng(0)
        arr = np.linspace(0, 10, 50) + rng.standard_normal(50) * 0.1
        res = compute_numaggs(arr, return_lintrend_approx_stats=True)
        names = get_numaggs_names(return_lintrend_approx_stats=True)
        assert len(res) == len(names)

    def test_compute_numaggs_geomean_log_mode(self):
        from mlframe.feature_engineering.numerical import compute_numaggs
        arr = np.abs(np.random.default_rng(0).standard_normal(50)) + 0.1
        res = compute_numaggs(arr, geomean_log_mode=True)
        assert len(res) > 0

    def test_compute_numaggs_float32_return(self):
        from mlframe.feature_engineering.numerical import compute_numaggs
        arr = np.random.default_rng(0).standard_normal(50)
        res = compute_numaggs(arr, return_float32=True)
        assert isinstance(res, np.ndarray) and res.dtype == np.float32

    def test_compute_numaggs_no_entropy_no_hurst(self):
        from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names
        arr = np.random.default_rng(0).standard_normal(50)
        res = compute_numaggs(arr, return_entropy=False, return_hurst=False)
        names = get_numaggs_names(return_entropy=False, return_hurst=False)
        assert len(res) == len(names)

    def test_compute_numaggs_with_xvals(self):
        from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names
        arr = np.linspace(0, 10, 30)
        # xvals must be float32 to match the kernel's internal default branch type.
        xvals = np.arange(30, dtype=np.float32)
        res = compute_numaggs(arr, xvals=xvals)
        names = get_numaggs_names()
        assert len(res) == len(names)

    def test_compute_numerical_aggregates_numba_empty(self):
        from mlframe.feature_engineering.numerical import compute_numerical_aggregates_numba
        res = compute_numerical_aggregates_numba(np.empty(0, dtype=np.float64))
        assert isinstance(res, list) and len(res) > 0

    def test_compute_numerical_aggregates_numba_with_weights(self):
        from mlframe.feature_engineering.numerical import compute_numerical_aggregates_numba
        arr = np.linspace(1, 10, 20)
        w = np.abs(np.random.default_rng(0).standard_normal(20)) + 0.1
        res = compute_numerical_aggregates_numba(arr, weights=w, return_exotic_means=True)
        assert len(res) > 0

    def test_compute_numerical_aggregates_numba_drawdown(self):
        from mlframe.feature_engineering.numerical import compute_numerical_aggregates_numba
        rng = np.random.default_rng(0)
        arr = 100 + np.cumsum(rng.standard_normal(50))
        res = compute_numerical_aggregates_numba(arr, return_drawdown_stats=True, return_profit_factor=True)
        assert len(res) > 0

    def test_compute_nunique_modes_quantiles_numpy(self):
        from mlframe.feature_engineering.numerical import compute_nunique_modes_quantiles_numpy
        arr = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0])
        res = compute_nunique_modes_quantiles_numpy(arr)
        assert len(res) > 0

    def test_compute_ncrossings(self):
        from mlframe.feature_engineering.numerical import compute_ncrossings
        arr = np.array([1.0, 2.0, 1.5, 2.5, 1.8, 3.0])
        marks = np.array([2.0], dtype=np.float64)
        out = compute_ncrossings(arr, marks)
        assert out.shape == (1,)

    def test_numaggs_over_matrix_rows_basic(self):
        from mlframe.feature_engineering.numerical import numaggs_over_matrix_rows
        rng = np.random.default_rng(0)
        vals = rng.standard_normal((5, 30))
        out = numaggs_over_matrix_rows(vals, numagg_params={}, rolling_ma=0, use_diffs=False)
        assert out.shape[0] == 5

    def test_numaggs_over_matrix_rows_with_rolling(self):
        from mlframe.feature_engineering.numerical import numaggs_over_matrix_rows
        rng = np.random.default_rng(0)
        vals = rng.standard_normal((3, 50))
        out = numaggs_over_matrix_rows(vals, numagg_params={}, rolling_ma=5, use_diffs=True)
        assert out.shape[0] == 3

    def test_compute_numaggs_parallel_validates_inputs(self):
        from mlframe.feature_engineering.numerical import compute_numaggs_parallel
        with pytest.raises(ValueError):
            compute_numaggs_parallel()


# ============================================================================
# timeseries.py (185 missing -> exercise create_aggregated_features branches)
# ============================================================================


class TestTimeseriesCoverage:
    def test_find_next_cumsum_left_index_basic(self):
        from mlframe.feature_engineering.timeseries import find_next_cumsum_left_index
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        left, total = find_next_cumsum_left_index(arr, 6.0)
        assert left >= 0 and total >= 6.0

    def test_find_next_cumsum_right_index_basic(self):
        from mlframe.feature_engineering.timeseries import find_next_cumsum_right_index
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        right, total = find_next_cumsum_right_index(arr, 6.0, left_index=0)
        assert right > 0 and total >= 6.0

    def test_find_next_cumsum_left_index_use_abs(self):
        from mlframe.feature_engineering.timeseries import find_next_cumsum_left_index
        arr = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float64)
        left, total = find_next_cumsum_left_index(arr, 4.0, use_abs=True)
        assert left >= 0

    def test_get_nwindows_expected(self):
        from mlframe.feature_engineering.timeseries import get_nwindows_expected
        n = get_nwindows_expected({"a": [1, 2, 3], "b": [10]})
        assert n == 4

    def test_get_ts_window_name(self):
        from mlframe.feature_engineering.timeseries import get_ts_window_name
        assert get_ts_window_name("", 5, "D") == "5D"
        assert "vol" in get_ts_window_name("vol", 1000)

    def test_create_aggregated_features_with_weighting(self):
        from mlframe.feature_engineering.timeseries import create_aggregated_features
        df = pd.DataFrame({
            "price": np.arange(20, dtype=float),
            "vol": np.abs(np.random.default_rng(0).standard_normal(20)) + 0.1,
        })
        feats, names = [], []
        create_aggregated_features(
            window_df=df, row_features=feats, create_features_names=True,
            features_names=names, dataset_name="ds", weighting_vars=("vol",),
        )
        assert len(feats) > 0 and len(names) == len(feats)

    def test_create_aggregated_features_with_ewma(self):
        from mlframe.feature_engineering.timeseries import create_aggregated_features
        df = pd.DataFrame({"price": np.arange(30, dtype=float)})
        feats, names = [], []
        create_aggregated_features(
            window_df=df, row_features=feats, create_features_names=True,
            features_names=names, dataset_name="ds", ewma_alphas=(0.3, 0.6),
        )
        assert len(feats) > 0

    def test_create_aggregated_features_with_nonlinear(self):
        from mlframe.feature_engineering.timeseries import create_aggregated_features
        df = pd.DataFrame({"price": np.arange(1, 21, dtype=float)})
        feats, names = [], []
        create_aggregated_features(
            window_df=df, row_features=feats, create_features_names=True,
            features_names=names, dataset_name="ds",
            nonnormal_vars=("price",), nonlinear_transforms=[np.log, np.cbrt],
        )
        assert any("log" in n for n in names)
        assert any("cbrt" in n for n in names)

    def test_create_aggregated_features_with_groupby(self):
        from mlframe.feature_engineering.timeseries import create_aggregated_features
        df = pd.DataFrame({
            "group": pd.Series(["A", "B", "A", "B", "A"], dtype="category"),
            "value": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        feats, names = [], []
        create_aggregated_features(
            window_df=df, row_features=feats, create_features_names=True,
            features_names=names, dataset_name="ds",
            groupby_vars={"group": ["value"]},
        )
        assert any("grpby" in n for n in names)

    def test_create_aggregated_features_with_rolling(self):
        from mlframe.feature_engineering.timeseries import create_aggregated_features
        df = pd.DataFrame({"price": np.arange(30, dtype=float)})
        feats, names = [], []
        rolling = [({"window": 5, "min_periods": 1}, "mean", {})]
        create_aggregated_features(
            window_df=df, row_features=feats, create_features_names=True,
            features_names=names, dataset_name="ds", rolling=rolling,
        )
        assert any("rol" in n for n in names)

    def test_create_aggregated_features_with_splitting_vars(self):
        from mlframe.feature_engineering.timeseries import create_aggregated_features
        df = pd.DataFrame({
            "price": np.arange(30, dtype=float),
            "vol": np.arange(30, dtype=float) + 1.0,
        })
        feats, names = [], []
        create_aggregated_features(
            window_df=df, row_features=feats, create_features_names=True,
            features_names=names, dataset_name="ds",
            splitting_vars={"price": ["vol"]},
        )
        assert any("split" in n for n in names)

    def test_create_aggregated_features_with_counts_processing(self):
        import re
        from mlframe.feature_engineering.timeseries import create_aggregated_features
        df = pd.DataFrame({"counts_x": [1, 2, 1, 2, 1, 3, 3, 2, 1]})
        feats, names = [], []
        create_aggregated_features(
            window_df=df, row_features=feats, create_features_names=True,
            features_names=names, dataset_name="ds",
            counts_processing_mask_regexp=re.compile(r"^counts_"),
        )
        assert any("vlscnt" in n for n in names)

    def test_create_aggregated_features_with_subsets(self):
        from mlframe.feature_engineering.timeseries import create_aggregated_features
        df = pd.DataFrame({
            "side": pd.Series(["BUY"] * 5 + ["SELL"] * 5, dtype="category"),
            "px": np.arange(10, dtype=float),
        })
        feats, names = [], []
        create_aggregated_features(
            window_df=df, row_features=feats, create_features_names=True,
            features_names=names, dataset_name="ds",
            subsets={"side": ["BUY", "SELL"]},
        )
        assert any("side=BUY" in n for n in names)

    def test_create_aggregated_features_with_wavelets(self):
        from mlframe.feature_engineering.timeseries import create_aggregated_features
        df = pd.DataFrame({"price": np.arange(64, dtype=float)})
        feats, names = [], []
        create_aggregated_features(
            window_df=df, row_features=feats, create_features_names=True,
            features_names=names, dataset_name="ds",
            waveletnames=("haar",),
        )
        assert any("haar" in n for n in names)

    def test_compute_corr_with_corrcoef(self):
        from mlframe.feature_engineering.timeseries import compute_corr
        rng = np.random.default_rng(0)
        a = rng.standard_normal(100)
        b = rng.standard_normal(100)
        c = compute_corr(a, b, np.corrcoef)
        assert 0 <= c <= 1

    def test_general_acf_no_windows(self):
        from mlframe.feature_engineering.timeseries import general_acf
        rng = np.random.default_rng(0)
        y = rng.standard_normal(2000)
        res = general_acf(y, lag_len=5, min_samples=100)
        assert "fixed_offsets" in res

    def test_compute_splitting_stats_with_datetime(self):
        from mlframe.feature_engineering.timeseries import compute_splitting_stats
        df = pd.DataFrame({
            "weight": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"]),
        })
        feats, names = [], []
        compute_splitting_stats(
            window_df=df, dataset_name="ds", splitting_vars={"x": ["weight"]},
            var="x", numaggs_names=["minr"], numaggs_values=[0.5],
            row_features=feats, features_names=names, create_features_names=True,
        )
        assert len(feats) >= 1


# ============================================================================
# bruteforce.py - skip PySR-requiring code, test the helper functions
# ============================================================================


class TestBruteforceHelper:
    def test_kfold_target_encode_helper(self):
        try:
            import category_encoders  # noqa: F401
        except ImportError:
            pytest.skip("category_encoders not installed")
        from mlframe.feature_engineering.bruteforce import _kfold_target_encode

        rng = np.random.default_rng(0)
        n = 80
        df = pd.DataFrame({"cat": rng.choice(list("abcd"), size=n)})
        df["cat"] = df["cat"].astype("category")
        target = pd.Series(rng.standard_normal(n))
        out = _kfold_target_encode(df, cols=["cat"], target=target, n_splits=4, random_state=0)
        assert out.shape == (n, 1)
        assert "cat" in out.columns
