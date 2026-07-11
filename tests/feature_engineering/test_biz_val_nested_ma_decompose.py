"""biz_value test for ``feature_engineering.nested_ma_decompose.nested_ma_decompose``.

Source: av_top3_rampaging_datahulk_minihack2017.md -- ``MA_last_10_3 = (Ten_Day_MA*10 - Three_Day_MA*3)/7``,
recovering "the average of the seven days preceding the last three days" algebraically from two already-
computed moving averages, instead of a third rolling pass over the raw series. This test confirms the
algebraic decomposition is numerically identical to directly computing the exclusive window's average from
raw data (the actual claimed win: same result, one fewer rolling computation).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_engineering.nested_ma_decompose import nested_ma_decompose


def test_biz_val_nested_ma_decompose_matches_direct_exclusive_window_average():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n).cumsum() + 100

    window_short, window_long = 3, 10
    ma_short = pd.Series(x).rolling(window_short).mean().to_numpy()
    ma_long = pd.Series(x).rolling(window_long).mean().to_numpy()

    decomposed = nested_ma_decompose(ma_short, ma_long, window_short, window_long)

    true_exclusive = np.full(n, np.nan)
    for i in range(window_long - 1, n):
        true_exclusive[i] = x[i - window_long + 1 : i - window_short + 1].mean()

    valid = ~np.isnan(decomposed) & ~np.isnan(true_exclusive)
    assert valid.sum() > 100
    np.testing.assert_allclose(decomposed[valid], true_exclusive[valid], atol=1e-10)


def test_nested_ma_decompose_rejects_non_nested_windows():
    import pytest

    with pytest.raises(ValueError):
        nested_ma_decompose(np.array([1.0]), np.array([1.0]), window_short=10, window_long=5)


def test_nested_ma_decompose_simple_hand_computed_case():
    # MA(10)=5.0 (sum=50), MA(3)=6.0 (sum=18) -> exclusive 7-day sum = 50-18=32, avg = 32/7.
    result = nested_ma_decompose(np.array([6.0]), np.array([5.0]), window_short=3, window_long=10)
    np.testing.assert_allclose(result, [32.0 / 7.0])
