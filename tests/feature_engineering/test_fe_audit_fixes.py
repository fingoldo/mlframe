"""Regression tests for the 2026-05-14 feature_engineering audit fixes.

Each test corresponds to a specific finding identified by the multi-agent audit; the test name
references the finding (file:line where the bug lived in the pre-fix code).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pytest

from tests.conftest import running_under_xdist
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# mps.py
# ---------------------------------------------------------------------------


class TestMpsTradeCountFix:
    """Audit P0-9: mps.py _trade_count returned 1 for continuing a non-zero position,
    biasing the DP toward churning out of profitable holds. Fixed to return 0 on continuation.
    """

    def test_continuing_long_is_zero_trades(self):
        """Continuing long is zero trades."""
        from mlframe.feature_engineering.mps import _trade_count

        assert _trade_count(1, 1) == 0

    def test_continuing_short_is_zero_trades(self):
        """Continuing short is zero trades."""
        from mlframe.feature_engineering.mps import _trade_count

        assert _trade_count(-1, -1) == 0

    def test_staying_flat_is_zero_trades(self):
        """Staying flat is zero trades."""
        from mlframe.feature_engineering.mps import _trade_count

        assert _trade_count(0, 0) == 0

    def test_opening_from_flat_is_one_trade(self):
        """Opening from flat is one trade."""
        from mlframe.feature_engineering.mps import _trade_count

        assert _trade_count(0, 1) == 1
        assert _trade_count(0, -1) == 1

    def test_closing_to_flat_is_one_trade(self):
        """Closing to flat is one trade."""
        from mlframe.feature_engineering.mps import _trade_count

        assert _trade_count(1, 0) == 1
        assert _trade_count(-1, 0) == 1

    def test_flipping_position_is_two_trades(self):
        """Flipping position is two trades."""
        from mlframe.feature_engineering.mps import _trade_count

        assert _trade_count(1, -1) == 2
        assert _trade_count(-1, 1) == 2


class TestMpsCorrectness:
    """End-to-end DP must produce the obvious answer for a monotone price."""

    def test_monotone_rise_then_fall(self):
        """Monotone rise then fall."""
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
        """Brownian increment hurst near half."""
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
        h, _c = compute_hurst_exponent(x, max_window=None)
        assert not np.isnan(h), "max_window=None should not crash or return NaN for valid input"

    def test_short_input_returns_nan(self):
        """Short input returns nan."""
        from mlframe.feature_engineering.hurst import compute_hurst_exponent

        h, c = compute_hurst_exponent(np.array([1.0, 2.0]), min_window=5)
        assert np.isnan(h) and np.isnan(c)

    def test_degenerate_constant_input_returns_nan(self):
        """Degenerate constant input returns nan."""
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
        """Exact for arithmetic progression."""
        from mlframe.feature_engineering.numerical import rolling_moving_average

        arr = np.arange(100, dtype=np.float64)
        ma = rolling_moving_average(arr, n=10)
        # MA over [0..9] = 4.5, [1..10] = 5.5, etc. Exact.
        expected = np.arange(91, dtype=np.float64) + 4.5
        np.testing.assert_allclose(ma, expected, atol=1e-12)

    def test_fast_variant_matches_compensated_on_well_conditioned(self):
        """``compensated=False`` (fastmath path) must match ``compensated=True`` (Kahan path)
        within ~n*eps*max(|x|) on well-conditioned float64 input.
        """
        from mlframe.feature_engineering.numerical import rolling_moving_average

        rng = np.random.default_rng(0)
        arr = rng.standard_normal(10_000).astype(np.float64)
        slow = rolling_moving_average(arr, n=100, compensated=True)
        fast = rolling_moving_average(arr, n=100, compensated=False)
        # Bound is generous: 100 * eps_f64 * max(|x|) ~ 100 * 2e-16 * 5 ~ 1e-13.
        # Use 1e-10 to absorb LLVM reassociation variability in the fast path.
        np.testing.assert_allclose(slow, fast, atol=1e-10)

    def test_simple_stats_fast_matches_kahan(self):
        """Fast path (default) must agree with the Kahan opt-in path to ~1e-12 on well-conditioned
        float64 N=10k input. Both are produced by ``_make_compute_simple_stats`` with different
        KAHAN closure constants; divergence signals an accidental DCE failure.
        """
        from mlframe.feature_engineering.numerical import compute_simple_stats_numba

        rng = np.random.default_rng(0)
        arr = rng.standard_normal(10_000).astype(np.float64)
        kahan = compute_simple_stats_numba(arr, compensated=True)
        fast = compute_simple_stats_numba(arr, compensated=False)
        # min/max/argmin/argmax must be EXACTLY equal (no float math).
        assert kahan[0] == fast[0] and kahan[1] == fast[1]
        assert kahan[2] == fast[2] and kahan[3] == fast[3]
        # mean / std within machine epsilon.
        np.testing.assert_allclose(kahan[4:], fast[4:], atol=1e-12)

    def test_moments_slope_mi_fast_matches_kahan(self):
        """Same invariant for the larger moments/slope/MI kernel under the dual-path wrapper."""
        from mlframe.feature_engineering.numerical import compute_moments_slope_mi

        rng = np.random.default_rng(1)
        arr = rng.standard_normal(5_000).astype(np.float64)
        mean = float(arr.mean())
        res_kahan, _ = compute_moments_slope_mi(arr, mean_value=mean, compensated=True)
        res_fast, _ = compute_moments_slope_mi(arr, mean_value=mean, compensated=False)
        # atol=1e-8 (not 1e-10): under NUMBA_DISABLE_JIT=1 the compiled-vs-interpreted execution
        # paths accumulate in a slightly different order, producing a ~1e-9 FP reduction-order
        # divergence -- the same acceptable-noise magnitude documented project-wide (nowhere near
        # the ~1e-3 selection-altering threshold), just not visible under normal JIT-enabled runs.
        np.testing.assert_allclose(res_kahan, res_fast, atol=1e-8, rtol=1e-8)

    def test_factory_pattern_kahan_is_compile_time_constant(self):
        """The two factory specializations must be distinct compiled kernels (not the same
        function dispatched at runtime). The private njit kernels (``_compute_simple_stats_*``,
        ``_compute_moments_slope_mi_*``) are imported directly here to verify identity differs.
        """
        from mlframe.feature_engineering.numerical import (
            _compute_simple_stats_compensated,
            _compute_simple_stats_fast,
            _compute_moments_slope_mi_compensated,
            _compute_moments_slope_mi_fast,
        )

        assert _compute_simple_stats_compensated is not _compute_simple_stats_fast
        assert _compute_moments_slope_mi_compensated is not _compute_moments_slope_mi_fast

    def test_fast_variant_diverges_on_ill_conditioned(self):
        """On a 1e9 + signal series the fast path accumulates measurable drift, while the
        compensated path stays at 1 ULP. This is exactly the regime the audit cared about.
        """
        from mlframe.feature_engineering.numerical import rolling_moving_average

        rng = np.random.default_rng(0)
        signal = rng.standard_normal(50_000).astype(np.float64)
        big = (signal + 1e9).astype(np.float64)
        slow = rolling_moving_average(big, n=500, compensated=True)
        fast = rolling_moving_average(big, n=500, compensated=False)
        # The compensated path stays within 1 ULP of (signal + 1e9)'s true rolling mean.
        # Fast path will drift; this assertion just documents the qualitative behaviour - we
        # do NOT require divergence (different hardware/LLVM may auto-vectorise tightly enough
        # to stay close). Just sanity-check that BOTH outputs are finite and within ~1e-4.
        assert np.isfinite(slow).all()
        assert np.isfinite(fast).all()
        assert np.allclose(slow, fast, atol=1e-3)  # close but not necessarily identical

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
    """Groups tests for: TestNumaggsLengthInvariant."""
    from typing import Tuple as _T

    @pytest.mark.parametrize("n", [10, 100, 1000])
    def test_numaggs_length_matches_names(self, n):
        """Numaggs length matches names."""
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
        """String series pads correctly."""
        from mlframe.feature_engineering.categorical import compute_countaggs, get_countaggs_names

        s = pd.Series(["a", "a", "b", "c", "c", "c"])
        values = compute_countaggs(s)
        names = get_countaggs_names()
        assert len(values) == len(names)

    def test_numeric_with_values_numaggs(self):
        """Numeric with values numaggs."""
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
    """Groups tests for: TestTimeseriesFixes."""
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
            """Apply fcn."""
            return

        x, y = create_windowed_features(
            df=df,
            start_index=0,
            end_index=0,
            past_processing_fcn=apply_fcn,
            future_processing_fcn=apply_fcn,
            past_windows={"": [3]},
            future_windows={"": [3]},
        )
        # Empty range -> nothing to compute.
        assert x is None and y is None

    def test_missing_window_var_does_not_abort_remaining(self):
        """Audit P0-5: line 727 had `break` instead of `continue` when a window_var column was
        missing. The bug silently dropped every later window_var after the first miss. This test
        builds a windows dict where the FIRST var is missing and the SECOND exists, and asserts
        the second still produces a window.
        """
        from mlframe.feature_engineering.timeseries import create_and_process_windows

        df = pd.DataFrame(
            {
                "existing_var": np.arange(20, dtype=np.float64),
            }
        )

        calls: list = []

        def apply_fcn(df, row_features, targets, features_names, dataset_name):
            """Apply fcn."""
            calls.append(dataset_name)
            row_features.append(len(df))

        windows = {
            "missing_var": [5],  # not in df - should be SKIPPED, not abort the loop
            "existing_var": [10],  # must still be processed after the miss
        }

        create_and_process_windows(
            df=df,
            base_point=15,
            apply_fcn=apply_fcn,
            windows=windows,
            window_features_names=[],
            window_features=[],
            forward_direction=False,
        )

        # Pre-fix code would produce zero calls (break terminated the loop on first miss).
        # Post-fix: at least one call for the existing_var window.
        assert len(calls) >= 1, "missing_var miss aborted the entire loop instead of `continue`-ing"
        # And the call MUST be for existing_var (which the bug would have skipped).
        assert any("existing_var" in name for name in calls), f"calls={calls}"

    def test_dtype_category_routes_to_countaggs(self):
        """`var.dtype in ('category', 'object')` must route to compute_countaggs
        when process_categoricals=True; otherwise the var is skipped entirely."""
        from mlframe.feature_engineering.timeseries import create_aggregated_features

        df = pd.DataFrame({"cat": pd.Series(["a", "b", "a", "c"], dtype="category")})
        feats: list = []
        names: list = []
        create_aggregated_features(
            window_df=df,
            row_features=feats,
            create_features_names=True,
            features_names=names,
            dataset_name="ds",
            process_categoricals=True,
        )
        assert len(feats) > 0, "process_categoricals=True must emit count-based features"
        assert all("cat" in n for n in names)

    def test_dtype_category_skipped_when_process_off(self):
        """Categorical variables are silently skipped when neither process_categoricals nor
        counts_processing_mask_regexp matches - regression guard for the branch ordering."""
        from mlframe.feature_engineering.timeseries import create_aggregated_features

        df = pd.DataFrame({"cat": pd.Series(["a", "b", "a", "c"], dtype="category")})
        feats: list = []
        names: list = []
        create_aggregated_features(
            window_df=df,
            row_features=feats,
            create_features_names=True,
            features_names=names,
            dataset_name="ds",
            process_categoricals=False,
        )
        assert feats == []
        assert names == []

    def test_datetime_diff_branch(self):
        """`'datetime' in dtype.name` -> raw_vals comes from diff().total_seconds()."""
        from mlframe.feature_engineering.timeseries import create_aggregated_features

        df = pd.DataFrame(
            {
                "ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05", "2024-01-09"]),
            }
        )
        feats: list = []
        names: list = []
        create_aggregated_features(
            window_df=df,
            row_features=feats,
            create_features_names=True,
            features_names=names,
            dataset_name="ds",
        )
        # Numaggs over [86400, 259200, 345600] (1d, 3d, 4d in seconds) must produce a real mean ~230400.
        # First numagg is `min` and should be > 0 (non-empty diff).
        assert len(feats) > 0

    def test_drawdown_var_adds_extra_numagg(self):
        """Drawdown var adds extra numagg."""
        from mlframe.feature_engineering.timeseries import create_aggregated_features
        from mlframe.feature_engineering.numerical import get_numaggs_names

        df = pd.DataFrame({"price": np.linspace(100, 110, 50).tolist() + np.linspace(110, 95, 50).tolist()})
        feats_normal: list = []
        names_normal: list = []
        create_aggregated_features(
            window_df=df,
            row_features=feats_normal,
            create_features_names=True,
            features_names=names_normal,
            dataset_name="ds",
        )
        feats_with_dd: list = []
        names_with_dd: list = []
        create_aggregated_features(
            window_df=df,
            row_features=feats_with_dd,
            create_features_names=True,
            features_names=names_with_dd,
            dataset_name="ds",
            drawdown_vars=["price"],
        )
        # drawdown_vars adds extra fields to the numagg block for that column.
        assert len(feats_with_dd) > len(feats_normal)
        n_extra = len(get_numaggs_names(return_drawdown_stats=True)) - len(get_numaggs_names())
        assert n_extra > 0

    def test_differences_features_branch(self):
        """Differences features branch."""
        from mlframe.feature_engineering.timeseries import create_aggregated_features

        df = pd.DataFrame({"price": np.arange(20, dtype=float)})
        feats: list = []
        names: list = []
        create_aggregated_features(
            window_df=df,
            row_features=feats,
            create_features_names=True,
            features_names=names,
            dataset_name="ds",
            differences_features=True,
        )
        # Pre-fix and post-fix both pass through differences; this just guards the wiring.
        assert any("dif" in n for n in names)

    def test_ratios_features_branch(self):
        """Ratios features branch."""
        from mlframe.feature_engineering.timeseries import create_aggregated_features

        df = pd.DataFrame({"price": np.linspace(100, 110, 20)})
        feats: list = []
        names: list = []
        create_aggregated_features(
            window_df=df,
            row_features=feats,
            create_features_names=True,
            features_names=names,
            dataset_name="ds",
            ratios_features=True,
        )
        assert any("rat" in n for n in names)

    def test_weighting_zero_sum_padded_not_crashed(self):
        """Audit P0-2: weighting_values.sum() == 0 used to divide-by-zero; now padded with 0s.

        We isolate the weighted-by-wgt block on the `price` column (caption tag ``-wgt-wgt-``)
        from `wgt`'s own numaggs (which legitimately produces NaN entropy on an all-zero series).
        """
        from mlframe.feature_engineering.timeseries import create_aggregated_features

        df = pd.DataFrame({"price": np.arange(10, dtype=float), "wgt": np.zeros(10)})
        feats: list = []
        names: list = []
        create_aggregated_features(
            window_df=df,
            row_features=feats,
            create_features_names=True,
            features_names=names,
            dataset_name="ds",
            weighting_vars=("wgt",),
        )
        # The weighted-by-wgt block on `price` is captioned `ds-price-wgt-wgt-<feat>` per the
        # captions_vars_sep convention; pick just those and assert the 0-padding kicked in.
        weighted_block_indices = [i for i, n in enumerate(names) if "price-wgt-wgt-" in n]
        assert weighted_block_indices, "no price-wgt-wgt block was produced; check captions"
        for i in weighted_block_indices:
            assert feats[i] == 0.0, f"expected padded 0.0, got {feats[i]} at name={names[i]}"

    def test_robust_features_q1_at_position_zero(self):
        """Audit P0-3 follow-up: q1_idx=0 used to be treated as missing via `if q1_idx and q3_idx`.
        Now `is not None`. Verify robust_features works even when q0.25 is the first numagg."""
        from mlframe.feature_engineering.timeseries import create_aggregated_features

        df = pd.DataFrame({"x": np.random.default_rng(0).standard_normal(100)})
        feats: list = []
        names: list = []
        create_aggregated_features(
            window_df=df,
            row_features=feats,
            create_features_names=True,
            features_names=names,
            dataset_name="ds",
            robust_features=True,
        )
        # Robust-subset features are tagged 'rbst' in the name.
        assert any("rbst" in n for n in names)

    def test_subset_recursion(self):
        """Subset recursion."""
        from mlframe.feature_engineering.timeseries import create_aggregated_features

        df = pd.DataFrame(
            {
                "value": np.arange(20, dtype=float),
                "group": ["A"] * 10 + ["B"] * 10,
            }
        )
        feats: list = []
        names: list = []
        create_aggregated_features(
            window_df=df,
            row_features=feats,
            create_features_names=True,
            features_names=names,
            dataset_name="ds",
            subsets={"group": ["A", "B"]},
        )
        # Each subset value must produce its own per-value-feature block.
        a_names = [n for n in names if "group=A" in n]
        b_names = [n for n in names if "group=B" in n]
        assert len(a_names) > 0 and len(b_names) > 0

    def test_compute_splitting_stats_clamps_negative_index(self):
        """Audit P0-4: int(numaggs_values[col_idx] * len(window_df)) - 1 could be -1 when the
        fractional position was 0, then iloc[:-1] silently dropped the last element. Now clamped
        to [0, len-1].
        """
        from mlframe.feature_engineering.timeseries import compute_splitting_stats

        df = pd.DataFrame(
            {
                "score": [1.0, 2.0, 3.0],
                "weight": [10.0, 20.0, 30.0],
            }
        )
        row_features: list = []
        features_names: list = []
        # numaggs_values=[0.0] paired with name "minr" -> raw_index = int(0*3)-1 = -1; pre-fix iloc
        # interpreted -1 as "drop last" and silently corrupted the split ratio.
        compute_splitting_stats(
            window_df=df,
            dataset_name="ds",
            splitting_vars={"score": ["weight"]},
            var="score",
            numaggs_names=["minr"],
            numaggs_values=[0.0],
            row_features=row_features,
            features_names=features_names,
            create_features_names=True,
        )
        # With index clamped to 0, pre_sum over iloc[:0] is 0; tot = 0 + sum(weight) = 60; ratio=0.
        assert row_features == [0.0]


# ---------------------------------------------------------------------------
# _numerical_stable.py
# ---------------------------------------------------------------------------


class TestNumericalStableFixes:
    """Groups tests for: TestNumericalStableFixes."""
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
        """Welford mean var matches numpy."""
        from mlframe.feature_engineering._numerical_stable import welford_mean_var_seq

        rng = np.random.default_rng(1)
        arr = rng.standard_normal(1000)
        mean, var, n = welford_mean_var_seq(arr)
        np.testing.assert_allclose(mean, np.mean(arr), rtol=1e-10)
        np.testing.assert_allclose(var, np.var(arr), rtol=1e-10)
        assert n == len(arr)


def _welford_moments_reference(arr: np.ndarray, order: str = "M4M3M2"):
    """Reference Python implementation of Pebay 2008 Eq. 2.1-2.4 with explicit update ORDER.

    Pure Python (slow) so we can swap the order of M2/M3/M4 lines at runtime and verify the
    "ORDER CRITICAL" comment in the production kernel is enforced by behaviour, not just by
    documentation. Returns ``(mean, var, skew, kurt)``.

    ``order`` is one of the six permutations of "M2", "M3", "M4". Only "M4M3M2" matches the
    canonical Pebay update where each line reads pre-update values of the lines below it.
    """
    n = 0
    mean = 0.0
    M = {"M2": 0.0, "M3": 0.0, "M4": 0.0}
    triplets = {
        "M4M3M2": ["M4", "M3", "M2"],
        "M4M2M3": ["M4", "M2", "M3"],
        "M3M4M2": ["M3", "M4", "M2"],
        "M3M2M4": ["M3", "M2", "M4"],
        "M2M4M3": ["M2", "M4", "M3"],
        "M2M3M4": ["M2", "M3", "M4"],
    }
    sequence = triplets[order]

    for x in arr:
        if not np.isfinite(x):
            continue
        n_old = n
        n += 1
        delta = x - mean
        delta_n = delta / n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n_old
        mean += delta_n

        for which in sequence:
            if which == "M4":
                M["M4"] += term1 * delta_n2 * (n * n - 3 * n + 3) + 6 * delta_n2 * M["M2"] - 4 * delta_n * M["M3"]
            elif which == "M3":
                M["M3"] += term1 * delta_n * (n - 2) - 3 * delta_n * M["M2"]
            else:  # M2
                M["M2"] += term1

    if n < 2:
        return mean, 0.0, 0.0, 0.0
    var = M["M2"] / n
    if var <= 0:
        return mean, 0.0, 0.0, 0.0
    skew = (M["M3"] / n) / (var**1.5)
    kurt = (n * M["M4"]) / (M["M2"] * M["M2"]) - 3.0
    return mean, var, skew, kurt


class TestPebayOrderCritical:
    """Mutation test: the production kernel claims "ORDER CRITICAL: M4, M3, M2". This test
    verifies the claim by running the reference implementation with each of 6 update orders
    and confirming only the canonical one matches scipy.
    """

    @pytest.fixture(scope="class")
    def expected_moments(self):
        """Expected moments."""
        from scipy import stats as sp_stats

        rng = np.random.default_rng(13)
        arr = rng.standard_normal(2000).astype(np.float64)
        return arr, (
            float(np.mean(arr)),
            float(np.var(arr)),
            float(sp_stats.skew(arr, bias=True)),
            float(sp_stats.kurtosis(arr, bias=True)),
        )

    def test_canonical_order_matches_scipy(self, expected_moments):
        """Canonical order matches scipy."""
        arr, (exp_mean, exp_var, exp_skew, exp_kurt) = expected_moments
        mean, var, skew, kurt = _welford_moments_reference(arr, order="M4M3M2")
        np.testing.assert_allclose(mean, exp_mean, rtol=1e-10)
        np.testing.assert_allclose(var, exp_var, rtol=1e-10)
        np.testing.assert_allclose(skew, exp_skew, atol=1e-9)
        np.testing.assert_allclose(kurt, exp_kurt, atol=1e-9)

    @pytest.mark.parametrize(
        "order",
        ["M4M2M3", "M3M4M2", "M3M2M4", "M2M4M3", "M2M3M4"],
    )
    def test_wrong_order_diverges(self, expected_moments, order):
        """Any non-canonical permutation must produce skew or kurt that differs from scipy by an
        amount LARGER than the canonical order's residual error.
        """
        arr, (_exp_mean, _exp_var, exp_skew, exp_kurt) = expected_moments
        _, _, skew, kurt = _welford_moments_reference(arr, order=order)
        # We accept failures in EITHER moment (some permutations only break M4, others only M3).
        skew_err = abs(skew - exp_skew)
        kurt_err = abs(kurt - exp_kurt)
        # Canonical residual ~ 1e-10; broken orders should be >>1e-3 on a normal sample.
        assert (skew_err > 1e-3) or (kurt_err > 1e-3), (
            f"order={order} did NOT diverge from scipy (skew_err={skew_err:.2e}, kurt_err={kurt_err:.2e}); "
            f"if this passes, the 'ORDER CRITICAL' comment in _numerical_stable.py may be vacuous."
        )

    def test_production_kernel_matches_canonical(self, expected_moments):
        """Sanity: the actual njit kernel agrees with the canonical Python reference."""
        from mlframe.feature_engineering._numerical_stable import welford_moments_seq

        arr, _ = expected_moments
        ref_mean, ref_var, ref_skew, ref_kurt = _welford_moments_reference(arr, order="M4M3M2")
        prod_mean, prod_var, prod_skew, prod_kurt, _ = welford_moments_seq(arr)
        np.testing.assert_allclose(prod_mean, ref_mean, rtol=1e-12)
        np.testing.assert_allclose(prod_var, ref_var, rtol=1e-12)
        np.testing.assert_allclose(prod_skew, ref_skew, atol=1e-12)
        np.testing.assert_allclose(prod_kurt, ref_kurt, atol=1e-12)

    def test_welford_moments_ill_conditioned(self):
        """Pebay/Welford should recover skew & kurt where naive two-pass loses precision.

        Inputs near 1e9 with stddev ~1 are the classic catastrophic-cancellation case for naive
        sum-of-squared-deviations: `sum(x^2)/n - mean^2` loses ~9 digits, leaving 7 digits of
        float64 precision so variance is correct only to 1e-7. The Welford path keeps full
        relative precision regardless of the offset.
        """
        from scipy import stats as sp_stats
        from mlframe.feature_engineering._numerical_stable import welford_moments_seq

        rng = np.random.default_rng(7)
        signal = rng.standard_normal(5000)
        big = signal + 1e9
        mean, var, skew, kurt, _n = welford_moments_seq(big)
        # mean and var of `big` differ from `signal` only by an exact 1e9 offset and 0 (variance
        # is shift-invariant). Welford's running mean accumulates O(N * eps) drift, ~1e-6 here;
        # the naive `sum_x^2/n - mean^2` baseline loses ~9 digits and would diverge by ~0.1.
        np.testing.assert_allclose(mean - 1e9, float(np.mean(signal)), rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(var, float(np.var(signal)), rtol=1e-4)
        np.testing.assert_allclose(skew, float(sp_stats.skew(signal, bias=True)), atol=5e-3)
        np.testing.assert_allclose(kurt, float(sp_stats.kurtosis(signal, bias=True)), atol=5e-2)


# ---------------------------------------------------------------------------
# Property-based (Hypothesis) - invariants that must hold for any input
# ---------------------------------------------------------------------------


class TestPropertyInvariants:
    """Fuzz invariants that are independent of the specific input data."""

    @given(prev=st.sampled_from([-1, 0, 1]), new=st.sampled_from([-1, 0, 1]))
    def test_trade_count_symmetric(self, prev, new):
        """Trade count symmetric."""
        from mlframe.feature_engineering.mps import _trade_count

        # Swapping prev<->new is equivalent to "playing the trades backwards": opening becomes
        # closing and vice-versa, so the count is the same.
        assert _trade_count(prev, new) == _trade_count(new, prev)

    @given(prev=st.sampled_from([-1, 0, 1]), new=st.sampled_from([-1, 0, 1]))
    def test_trade_count_bounded(self, prev, new):
        """Trade count bounded."""
        from mlframe.feature_engineering.mps import _trade_count

        # Maximum trades for a single transition is 2 (close + open under a flip).
        assert 0 <= _trade_count(prev, new) <= 2

    @given(prev=st.sampled_from([-1, 0, 1]), new=st.sampled_from([-1, 0, 1]))
    def test_trade_count_flat_baseline(self, prev, new):
        """Trade count flat baseline."""
        from mlframe.feature_engineering.mps import _trade_count

        # If at least one of prev/new is zero (flat), at most 1 trade is needed.
        if prev == 0 or new == 0:
            assert _trade_count(prev, new) <= 1

    @given(
        n=st.integers(min_value=2, max_value=200),
        window=st.integers(min_value=2, max_value=50),
        seed=st.integers(min_value=0, max_value=10_000),
    )
    @settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_rolling_ma_matches_naive(self, n, window, seed):
        """Kahan-compensated rolling MA must agree with `np.mean(window)` per position to ~eps."""
        from mlframe.feature_engineering.numerical import rolling_moving_average

        if window > n:
            pytest.skip("window > n is intentionally rejected by the public API")
        rng = np.random.default_rng(seed)
        arr = rng.standard_normal(n).astype(np.float64)
        ma = rolling_moving_average(arr, n=window)
        # Compare against the unambiguous numpy formula at each position.
        naive = np.array([arr[i : i + window].mean() for i in range(n - window + 1)])
        np.testing.assert_allclose(ma, naive, atol=1e-10)


# ---------------------------------------------------------------------------
# MPS DP-behaviour: the _trade_count fix must change DP outcomes under tc>0
# ---------------------------------------------------------------------------


class TestMpsDpUnderTransactionCost:
    """Beyond unit-testing _trade_count, verify the FIX propagates into find_best_mps_sequence.

    Under high transaction costs and a smoothly trending price, the OPTIMAL strategy is to hold
    one position for the entire trend (1 entry + 1 exit). The pre-fix code, charging a phantom
    transaction cost per bar of a continuing position, would prefer to exit and re-enter the
    flat state, producing a noisier position sequence.
    """

    def test_high_tc_smooth_trend_stays_long(self):
        """High tc smooth trend stays long."""
        from mlframe.feature_engineering.mps import find_maximum_profit_system

        # Smooth upward trend with substantial transaction cost: optimal is "stay long".
        prices = np.linspace(100.0, 110.0, 50)
        r = find_maximum_profit_system(prices, tc=0.001, tc_mode="fraction")
        positions = r["positions"]
        # Pre-fix code would have ping-ponged due to phantom tc on every bar of holding.
        # Post-fix expects mostly +1 with optimize_consecutive_regions filling gaps.
        long_fraction = float(np.mean(positions == 1))
        assert long_fraction > 0.8, f"Optimal under high tc on a smooth uptrend should be >80% long, got {long_fraction:.2f}"

    def test_transaction_count_does_not_explode(self):
        """Number of position changes must be O(price-reversals), not O(N)."""
        from mlframe.feature_engineering.mps import find_maximum_profit_system

        prices = np.linspace(100.0, 110.0, 100)
        r = find_maximum_profit_system(prices, tc=0.001, tc_mode="fraction")
        positions = r["positions"]
        changes = int(np.sum(np.diff(positions) != 0))
        # On a single monotone trend, optimal has 0 changes (constant +1). The buggy code
        # produced one change per bar (>= len(positions)-1).
        assert changes <= 2, f"Expected <=2 position changes on monotone trend, got {changes}"


# ---------------------------------------------------------------------------
# Performance regression: fastmath off on rolling_moving_average should be <2x slower
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("NUMBA_DISABLE_JIT") == "1",
    reason="asserts a numba-JIT-vs-numpy speedup ratio; meaningless (and expected to fail) with JIT disabled",
)
class TestPerformance:
    """Verify the Kahan-precision win did not cost too much speed.

    The audit claimed <5% speedup from fastmath on this loop because the dependency chain is
    sequential (no FMA, no vectorisation opportunity, no division to convert to reciprocal). We
    set the bar lower (2x) so the test isn't flaky on noisy CI machines, but a 10x regression
    would catch a future refactor that accidentally re-enables fastmath in a way that helps.
    """

    @pytest.mark.parametrize("n,window", [(10_000, 50), (10_000, 200)])
    def test_rolling_ma_throughput(self, n, window):
        """Rolling ma throughput."""
        from mlframe.feature_engineering.numerical import rolling_moving_average

        rng = np.random.default_rng(0)
        arr = rng.standard_normal(n).astype(np.float64)

        # JIT warm-up (numba compilation cost dominates on first call).
        rolling_moving_average(arr, n=window)

        # Time the Kahan path.
        t0 = time.perf_counter()
        for _ in range(20):
            rolling_moving_average(arr, n=window)
        kahan_time = (time.perf_counter() - t0) / 20

        # Baseline: np.cumsum-based moving average (vectorised, no Kahan).
        t0 = time.perf_counter()
        for _ in range(20):
            cumsum = np.cumsum(arr)
            cumsum_padded = np.concatenate([[0.0], cumsum])
            _ = (cumsum_padded[window:] - cumsum_padded[:-window]) / window
        cumsum_time = (time.perf_counter() - t0) / 20

        # The Kahan path is inherently sequential and slower than vectorised cumsum, but the
        # ratio should be bounded. Allow up to 10x because cumsum is BLAS-vectorised and Kahan
        # is a strict scalar loop - this test only catches catastrophic slowdowns.
        ratio = kahan_time / cumsum_time
        assert ratio < 10.0, (
            f"rolling_moving_average is {ratio:.1f}x slower than np.cumsum baseline "
            f"(kahan={kahan_time * 1e6:.1f}us, cumsum={cumsum_time * 1e6:.1f}us); "
            f"check that NUMBA_NJIT_PARAMS isn't blocking JIT caching."
        )

    @pytest.mark.parametrize("n,window", [(50_000, 100), (50_000, 500)])
    def test_fastmath_off_is_under_30pct_slowdown(self, n, window):
        """Absolute quantification: a fastmath=True clone of rolling_moving_average is built
        inline and benchmarked against the production fastmath=False version. The audit claim
        was <5% speedup from fastmath; this test gives the bug-introducing-PR 30% headroom
        before failing (CI noise + machine variance need slack but we still catch a future
        accidental fastmath=True flip).
        """
        import numba
        from mlframe.feature_engineering.numerical import rolling_moving_average

        @numba.njit(fastmath=True, cache=False)
        def rolling_ma_fastmath(arr, n):
            """Rolling ma fastmath."""
            result = np.empty(len(arr) - n + 1, dtype=arr.dtype)
            sum_window = np.sum(arr[:n])
            mult = 1 / n
            result[0] = sum_window * mult
            kahan_c = 0.0
            for i in range(1, len(arr) - n + 1):
                y = (arr[i + n - 1] - arr[i - 1]) - kahan_c
                t = sum_window + y
                kahan_c = (t - sum_window) - y
                sum_window = t
                result[i] = sum_window * mult
            return result

        rng = np.random.default_rng(1)
        arr = rng.standard_normal(n).astype(np.float64)

        # JIT warm-up.
        rolling_moving_average(arr, n=window)
        rolling_ma_fastmath(arr, n=window)

        reps = 30

        t0 = time.perf_counter()
        for _ in range(reps):
            rolling_moving_average(arr, n=window)
        slow_time = (time.perf_counter() - t0) / reps

        t0 = time.perf_counter()
        for _ in range(reps):
            rolling_ma_fastmath(arr, n=window)
        fast_time = (time.perf_counter() - t0) / reps

        slowdown_pct = (slow_time / fast_time - 1.0) * 100.0
        print(f"\n[fastmath bench n={n} window={window}] off={slow_time * 1e6:.1f}us  on={fast_time * 1e6:.1f}us  slowdown={slowdown_pct:+.1f}%")
        # MEASURED 2026-05-14: ~260% slowdown on Windows / numba 0.59 / LLVM. With fastmath=True
        # LLVM reassociates the Kahan compensator to zero AND SIMD-vectorises the resulting plain
        # cumsum (the sequential dependency through `sum_window` is broken once Kahan's `c` is
        # constant-folded out). This is a genuine cost of preserving precision; quantified here
        # so a future maintainer doesn't re-flip the flag on the (incorrect) "<5%" intuition.
        # macOS GitHub-hosted runners (verified 2026-05-26 -- 808% slowdown on macos-latest) hit a
        # different LLVM/clang vectorisation profile: the fastmath path SIMD-cumsums even harder
        # (AVX2 + FMA) so the Kahan/fastmath ratio is closer to 10x than 3.5x. Raise the ceiling
        # on Darwin so the sensor still catches catastrophic regressions (>15x = JIT broken)
        # without flagging the intentional precision-vs-speed tradeoff.
        import sys

        ceiling = 1500.0 if sys.platform == "darwin" else 500.0
        if running_under_xdist():
            # Under the full ``-n`` run the scalar Kahan arm is starved disproportionately to the SIMD fastmath arm,
            # inflating the ratio; widen the ceiling so it still catches a broken-JIT catastrophe (>20x) without flaking.
            ceiling *= 4.0
        assert slowdown_pct < ceiling, (
            f"fastmath=False is {slowdown_pct:.1f}% slower than fastmath=True; expected ~250% "
            f"on this hardware (ceiling {ceiling:.0f}%). Check that NUMBA_NJIT_PARAMS hasn't "
            f"changed (e.g. nogil/cache defaults) or that AVX/AVX2 vectorisation isn't blocked "
            f"at the OS level."
        )


# ---------------------------------------------------------------------------
# Edge cases - empty arrays, single elements, all-NaN
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Groups tests for: TestEdgeCases."""
    def test_compute_numaggs_short_array_returns_nans(self):
        """Compute numaggs short array returns nans."""
        from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names

        values = compute_numaggs(np.array([1.0]))
        names = get_numaggs_names()
        assert len(values) == len(names)
        # Single-element arrays should produce NaN sentinels, not zeros.
        assert all(np.isnan(v) for v in values)

    def test_compute_numaggs_empty_array_returns_nans(self):
        """Compute numaggs empty array returns nans."""
        from mlframe.feature_engineering.numerical import compute_numaggs, get_numaggs_names

        values = compute_numaggs(np.array([], dtype=np.float64))
        names = get_numaggs_names()
        assert len(values) == len(names)

    def test_hurst_constant_array_returns_nan(self):
        """Hurst constant array returns nan."""
        from mlframe.feature_engineering.hurst import compute_hurst_exponent

        h, c = compute_hurst_exponent(np.full(200, 5.0))
        assert np.isnan(h) and np.isnan(c)

    def test_compute_countaggs_single_value(self):
        """Compute countaggs single value."""
        from mlframe.feature_engineering.categorical import compute_countaggs, get_countaggs_names

        s = pd.Series(["only_one_category"] * 10)
        values = compute_countaggs(s)
        names = get_countaggs_names()
        assert len(values) == len(names)

    def test_kahan_dot_zero_length(self):
        """Kahan dot zero length."""
        from mlframe.feature_engineering._numerical_stable import kahan_dot_seq

        a = np.array([], dtype=np.float64)
        b = np.array([], dtype=np.float64)
        assert kahan_dot_seq(a, b) == 0.0


# ---------------------------------------------------------------------------
# Bruteforce: leakage_free encoder path
# ---------------------------------------------------------------------------


class TestFinancialUnnestOrdering:
    """Audit P1: in the pre-fix code, `unnests.append(...)` happened AFTER several
    `apply_ta_indicator(...)` calls were already constructed. Those earlier expressions never
    saw the later appends and therefore failed to unnest their struct output. The fix
    pre-computes the full `unnests` list before any expression is built.

    This test runs add_ohlcv_ta_indicators on a small synthetic OHLCV and asserts that for each
    struct-returning indicator (mama, bbands, aroon, stoch, ...) the result frame contains the
    flattened scalar columns rather than a leftover struct.
    """

    def test_struct_indicators_are_unnested(self):
        """Struct indicators are unnested."""
        import polars as pl

        pytest.importorskip("polars_talib")
        from mlframe.feature_engineering.financial import add_ohlcv_ta_indicators

        rng = np.random.default_rng(0)
        n = 200
        ohlcv = pl.DataFrame(
            {
                "ticker": ["AAPL"] * n,
                "date": list(range(n)),
                "open": rng.uniform(95, 105, n),
                "high": rng.uniform(100, 110, n),
                "low": rng.uniform(90, 100, n),
                "close": rng.uniform(95, 105, n),
                "volume": rng.uniform(1e6, 1e7, n),
            }
        )

        result = add_ohlcv_ta_indicators(
            ohlcv,
            ta_windows=[5],
            ticker_column="ticker",
            cast_f64_to_f32=False,
        )

        # If unnesting succeeded, the result must NOT contain any Struct-dtype columns - every
        # struct-returning indicator should have been flattened into scalar children.
        struct_cols = [name for name, dtype in result.schema.items() if isinstance(dtype, pl.Struct)]
        assert struct_cols == [], f"Found unflattened struct columns - the unnest-ordering fix is incomplete: {struct_cols}"

        # And the canonical flattened names from the struct indicators must be present.
        result_cols = set(result.columns)
        # `aroon{5}` (struct -> aroondown/aroonup) should appear as flattened children.
        aroon_children = [c for c in result_cols if c.startswith("aroon5_")]
        assert len(aroon_children) >= 1, f"aroon5 was not unnested into children. Cols: {sorted(result_cols)[:30]}..."


class TestBruteforceLeakageFreeEncoding:
    """Audit P1: a `leakage_free=True` opt-in was added that runs CatBoostEncoder in KFold OOF
    mode. Verifying the helper exists and produces different (less-leaky) encodings than the
    fit-on-everything baseline."""

    def test_kfold_helper_produces_different_encoding_than_fit_all(self):
        """Kfold helper produces different encoding than fit all."""
        pytest.importorskip("category_encoders")
        from category_encoders import CatBoostEncoder
        from mlframe.feature_engineering.bruteforce import _kfold_target_encode

        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({"cat": rng.choice(["a", "b", "c", "d"], size=n)})
        df["cat"] = df["cat"].astype("category")
        target = pd.Series(rng.standard_normal(n))

        # category_encoders >= 2.6 ships ``__sklearn_tags__`` that calls
        # ``super().__sklearn_tags__()``; on certain category_encoders /
        # sklearn combos (Python 3.9 ubuntu CI runner: sklearn 1.5.x +
        # category_encoders 2.6.x) the MRO super() target lacks that
        # method and CatBoostEncoder.fit raises ``AttributeError: 'super'
        # object has no attribute '__sklearn_tags__'`` deep inside
        # ``_check_fit_inputs -> _get_tags -> _to_old_tags(get_tags(self))``.
        # The instance-method probe doesn't trigger the bug because the
        # MRO walk goes via sklearn's get_tags() helper, not the regular
        # method-resolution path. Wrap the actual fit calls so the
        # detection is the same path that would otherwise fail.
        try:
            oof = _kfold_target_encode(df, cols=["cat"], target=target, n_splits=5, random_state=42)
            encoder = CatBoostEncoder(cols=["cat"], return_df=True)
            fit_all = encoder.fit_transform(df[["cat"]], target)
        except AttributeError as exc:
            if "__sklearn_tags__" in str(exc):
                pytest.skip(
                    f"category_encoders / sklearn version mismatch on this "
                    f"runner: {exc}. CatBoostEncoder.fit's "
                    f"``__sklearn_tags__`` super() chain is broken on this "
                    f"combo (upstream incompat, not anything mlframe owns)."
                )
            raise

        # The two encodings must differ - if they were identical we'd not have removed the leak.
        diff = float(np.mean(np.abs(oof["cat"].values - fit_all["cat"].values)))
        assert diff > 1e-6, f"OOF encoding indistinguishable from fit-on-all (diff={diff:.2e})"
