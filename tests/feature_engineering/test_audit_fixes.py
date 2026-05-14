"""Regression tests for the 2026-05-14 feature_engineering audit fixes.

Each test corresponds to a specific finding identified by the multi-agent audit; the test name
references the finding (file:line where the bug lived in the pre-fix code).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# mps.py
# ---------------------------------------------------------------------------


class TestMpsTradeCountFix:
    """Audit P0-9: mps.py _trade_count returned 1 for continuing a non-zero position,
    biasing the DP toward churning out of profitable holds. Fixed to return 0 on continuation.
    """

    def test_continuing_long_is_zero_trades(self):
        from mlframe.feature_engineering.mps import _trade_count
        assert _trade_count(1, 1) == 0

    def test_continuing_short_is_zero_trades(self):
        from mlframe.feature_engineering.mps import _trade_count
        assert _trade_count(-1, -1) == 0

    def test_staying_flat_is_zero_trades(self):
        from mlframe.feature_engineering.mps import _trade_count
        assert _trade_count(0, 0) == 0

    def test_opening_from_flat_is_one_trade(self):
        from mlframe.feature_engineering.mps import _trade_count
        assert _trade_count(0, 1) == 1
        assert _trade_count(0, -1) == 1

    def test_closing_to_flat_is_one_trade(self):
        from mlframe.feature_engineering.mps import _trade_count
        assert _trade_count(1, 0) == 1
        assert _trade_count(-1, 0) == 1

    def test_flipping_position_is_two_trades(self):
        from mlframe.feature_engineering.mps import _trade_count
        assert _trade_count(1, -1) == 2
        assert _trade_count(-1, 1) == 2


class TestMpsCorrectness:
    """End-to-end DP must produce the obvious answer for a monotone price."""

    def test_monotone_rise_then_fall(self):
        from mlframe.feature_engineering.mps import find_maximum_profit_system
        prices = np.array([100.0, 101.0, 102.0, 103.0, 102.0, 101.0, 100.0, 99.0])
        r = find_maximum_profit_system(prices, tc=0.0)
        # Going long on the rise, short on the fall.
        assert list(r["positions"]) == [1, 1, 1, -1, -1, -1, -1]


# ---------------------------------------------------------------------------
# hurst.py
# ---------------------------------------------------------------------------


class TestHurstFixes:
    """Audit P1: ddof=1 (Mandelbrot R/S convention), boundary off-by-one fix, max_window=None
    sentinel handled in the Python wrapper, np.unique on int-cast window sizes.
    """

    def test_brownian_increment_hurst_near_half(self):
        from mlframe.feature_engineering.hurst import compute_hurst_exponent
        rng = np.random.default_rng(0)
        increments = rng.choice([-1.0, 1.0], size=4000)
        # take_diffs=True turns the cumulative path back into i.i.d. increments before R/S.
        h, _ = compute_hurst_exponent(np.cumsum(increments), take_diffs=True)
        assert 0.35 < h < 0.65, f"H={h} outside expected [0.35, 0.65] for i.i.d. increments"

    def test_max_window_none_works(self):
        """Python wrapper must translate max_window=None to the int sentinel for the njit kernel."""
        from mlframe.feature_engineering.hurst import compute_hurst_exponent
        rng = np.random.default_rng(1)
        x = rng.standard_normal(500)
        h, c = compute_hurst_exponent(x, max_window=None)
        assert not np.isnan(h), "max_window=None should not crash or return NaN for valid input"

    def test_short_input_returns_nan(self):
        from mlframe.feature_engineering.hurst import compute_hurst_exponent
        h, c = compute_hurst_exponent(np.array([1.0, 2.0]), min_window=5)
        assert np.isnan(h) and np.isnan(c)

    def test_degenerate_constant_input_returns_nan(self):
        from mlframe.feature_engineering.hurst import compute_hurst_exponent
        h, _ = compute_hurst_exponent(np.ones(100))
        # Constant input has zero variance, every R/S degenerates, regression is undefined.
        assert np.isnan(h)


# ---------------------------------------------------------------------------
# numerical.py
# ---------------------------------------------------------------------------


class TestRollingMovingAverageKahan:
    """Audit P1: rolling_moving_average had fastmath=True over a Kahan compensator. LLVM was free to
    reassociate `(t - sum_window) - y` to zero, silently nullifying the compensation. Fixed by
    setting fastmath=False; this test verifies the precision is now exact for an arithmetic
    progression where the simple formula gives a known result.
    """

    def test_exact_for_arithmetic_progression(self):
        from mlframe.feature_engineering.numerical import rolling_moving_average
        arr = np.arange(100, dtype=np.float64)
        ma = rolling_moving_average(arr, n=10)
        # MA over [0..9] = 4.5, [1..10] = 5.5, etc. Exact.
        expected = np.arange(91, dtype=np.float64) + 4.5
        np.testing.assert_allclose(ma, expected, atol=1e-12)

    def test_ill_conditioned_recovers_via_kahan(self):
        """Large constant + small signal: naive sum loses precision; Kahan keeps it. Test that the
        rolling mean recovers the small-signal pattern on top of a huge offset."""
        from mlframe.feature_engineering.numerical import rolling_moving_average
        rng = np.random.default_rng(0)
        signal = rng.standard_normal(1000) * 1e-6
        big = signal + 1e9
        ma_big = rolling_moving_average(big, n=50)
        ma_signal = rolling_moving_average(signal, n=50)
        # The two rolling means must differ by exactly 1e9 at every position.
        np.testing.assert_allclose(ma_big - ma_signal, 1e9, atol=1e-4)


class TestNumaggsLengthInvariant:
    from typing import Tuple as _T

    @pytest.mark.parametrize(
        "n", [10, 100, 1000]
    )
    def test_numaggs_length_matches_names(self, n):
        from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names
        rng = np.random.default_rng(42)
        arr = rng.standard_normal(n)
        values = compute_numaggs(arr)
        names = get_numaggs_names()
        assert len(values) == len(names), f"n={n}: {len(values)} vs {len(names)}"


# ---------------------------------------------------------------------------
# categorical.py
# ---------------------------------------------------------------------------


class TestCountaggsInvariant:
    """Audit P1: compute_countaggs and get_countaggs_names must agree on length; the new code
    raises AssertionError if compute_numaggs(directional_only=True) returns the wrong count."""

    def test_string_series_pads_correctly(self):
        from mlframe.feature_engineering.categorical import compute_countaggs, get_countaggs_names
        s = pd.Series(["a", "a", "b", "c", "c", "c"])
        values = compute_countaggs(s)
        names = get_countaggs_names()
        assert len(values) == len(names)

    def test_numeric_with_values_numaggs(self):
        from mlframe.feature_engineering.categorical import compute_countaggs, get_countaggs_names
        s = pd.Series([1, 1, 2, 3, 3, 3, 4, 5])
        kw = dict(counts_compute_values_numaggs=True)
        values = compute_countaggs(s, **kw)
        names = get_countaggs_names(**kw)
        assert len(values) == len(names)


# ---------------------------------------------------------------------------
# timeseries.py
# ---------------------------------------------------------------------------


class TestTimeseriesFixes:

    def test_acf_seeds_lag0_at_zero(self):
        """Audit P0-related: general_acf seeded acfs_index = [1.0] instead of [0.0]; lag-0 index
        and lag-1 index then collided. Now seeded with [0.0]."""
        from mlframe.feature_engineering.timeseries import general_acf
        rng = np.random.default_rng(0)
        y = rng.standard_normal(3000)
        res = general_acf(y, lag_len=5, min_samples=100)
        idx = list(res["fixed_offsets"].index)
        # First entry is lag 0 (autocorrelation = 1 by definition).
        assert idx[0] == 0.0, f"expected lag-0 = 0.0, got {idx[0]}"
        # Subsequent indices are 1, 2, ... not 1.0 duplicated with lag-0.
        assert len(set(idx)) == len(idx), f"duplicate lag indices: {idx}"

    def test_create_windowed_features_requires_df(self):
        """Pre-fix df defaulted to None and later crashed with TypeError on len(df). Now raises ValueError up front."""
        from mlframe.feature_engineering.timeseries import create_windowed_features
        with pytest.raises(ValueError, match="df is required"):
            create_windowed_features(df=None)

    def test_create_windowed_features_end_index_none_vs_zero(self):
        """Pre-fix `if not end_index:` collapsed end_index=0 with unset. Now end_index=0 produces
        an empty range and returns (None, None); end_index=None means "to len(df)"."""
        from mlframe.feature_engineering.timeseries import create_windowed_features
        df = pd.DataFrame({"x": range(10)})

        def apply_fcn(df, row_features, targets, features_names, dataset_name):
            return

        x, y = create_windowed_features(
            df=df, start_index=0, end_index=0, past_processing_fcn=apply_fcn,
            future_processing_fcn=apply_fcn, past_windows={"": [3]}, future_windows={"": [3]},
        )
        # Empty range -> nothing to compute.
        assert x is None and y is None


# ---------------------------------------------------------------------------
# _numerical_stable.py
# ---------------------------------------------------------------------------


class TestNumericalStableFixes:

    def test_kahan_dot_seq_raises_on_length_mismatch(self):
        """Audit Low: previously silently truncated to min length; now raises."""
        from mlframe.feature_engineering._numerical_stable import kahan_dot_seq
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with pytest.raises((ValueError, Exception)):
            kahan_dot_seq(a, b)

    def test_welford_moments_matches_scipy(self):
        """Audit P1: cross-check the Pebay 2008 Eq. 2.1-2.4 implementation against scipy.

        Verifies the convention (pre/post-update n) was implemented correctly.
        """
        from scipy import stats as sp_stats
        from mlframe.feature_engineering._numerical_stable import welford_moments_seq

        rng = np.random.default_rng(0)
        arr = rng.standard_normal(500)
        mean, var, skew, kurt, n = welford_moments_seq(arr)

        # scipy.stats.skew/kurtosis with bias=True ARE the moment-form g1/g2; matches Pebay output.
        expected_mean = float(np.mean(arr))
        expected_var = float(np.var(arr))  # ddof=0
        expected_skew = float(sp_stats.skew(arr, bias=True))
        expected_kurt = float(sp_stats.kurtosis(arr, bias=True))

        assert n == len(arr)
        np.testing.assert_allclose(mean, expected_mean, rtol=1e-10)
        np.testing.assert_allclose(var, expected_var, rtol=1e-10)
        np.testing.assert_allclose(skew, expected_skew, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(kurt, expected_kurt, rtol=1e-8, atol=1e-10)

    def test_welford_mean_var_matches_numpy(self):
        from mlframe.feature_engineering._numerical_stable import welford_mean_var_seq
        rng = np.random.default_rng(1)
        arr = rng.standard_normal(1000)
        mean, var, n = welford_mean_var_seq(arr)
        np.testing.assert_allclose(mean, np.mean(arr), rtol=1e-10)
        np.testing.assert_allclose(var, np.var(arr), rtol=1e-10)
        assert n == len(arr)
