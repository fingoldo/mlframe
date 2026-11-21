"""Edge-case regressions from audit 2026-11-20, cluster 24: NaN windows, index alignment, empty spans, shifted tails (24-F12..F15)."""

import numpy as np
import pandas as pd

from mlframe.feature_engineering.mps import find_best_mps_sequence
from mlframe.feature_engineering.nearest_past_join import nearest_past_join
from mlframe.feature_engineering.timeseries import create_ts_features_parallel
from mlframe.feature_engineering.windowed_shape import rolling_longest_monotone_run


def test_monotone_run_is_nan_for_a_window_with_a_missing_value():
    """A NaN inside the window used to reset the run and return a finite count (24-F12)."""
    values = np.array([1.0, 2.0, np.nan, 3.0, 4.0])
    out = rolling_longest_monotone_run(values, np.zeros(5, dtype=np.int64), window_K=5, direction="up")
    assert np.isnan(out[-1])


def test_monotone_run_is_finite_for_a_clean_window():
    """Windows without missing values keep their finite run length."""
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = rolling_longest_monotone_run(values, np.zeros(5, dtype=np.int64), window_K=5, direction="up")
    assert out[-1] == 5.0


def test_nearest_past_join_fallback_tier_accepts_a_non_range_index():
    """The fallback tier used to look up positional labels in the caller's original index and raised KeyError (24-F13)."""
    left = pd.DataFrame({"g": ["a", "a", "b", "b"], "t": [1, 2, 3, 4]}, index=[10, 11, 12, 13])
    right = pd.DataFrame({"g": ["a", "a"], "t": [0, 1], "v": [5.0, 6.0]})
    out = nearest_past_join(left.iloc[1:], right, on="t", by=["g"], right_value_cols=["v"], fallback_by_chain=[[]])
    np.testing.assert_allclose(out["v"].to_numpy(), [6.0, 6.0, 6.0])


def test_parallel_ts_features_empty_span_returns_none_instead_of_dividing_by_zero():
    """start == end used to compute step = span // 0 (24-F14)."""
    result = create_ts_features_parallel(0, 0, ts_func=lambda **kwargs: None)
    assert result == (None, None)


def test_mps_shift_fills_the_vacated_tail_from_the_last_valid_position():
    """After a left shift the last `shift` entries repeat the last valid position, not stale values (24-F15)."""
    rng = np.random.default_rng(0)
    prices = np.cumsum(rng.standard_normal(80)) + 100.0
    positions, _profits = find_best_mps_sequence(prices, prices, tc=0.0, tc_mode_is_fraction=True, shift=2)
    assert positions[-1] == positions[-2] == positions[-3]
