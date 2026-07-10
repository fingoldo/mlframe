"""biz_value test for ``feature_engineering.select_significant_lags``.

The win: on a known generating process with direct dependence at exactly lags 1 and 7 (an AR-like series
skipping lags 2-6), the top-2 PACF-ranked candidates should recover exactly ``{1, 7}`` -- the true generating
lags -- rather than a guessed fixed grid. A white-noise control series (no true lag structure) should flag
close to the expected ~5% false-positive rate, not a large fraction of lags.
"""
from __future__ import annotations

import numpy as np

from mlframe.feature_engineering.acf_lag_selection import select_significant_lags


def test_biz_val_select_significant_lags_recovers_true_generating_lags():
    rng = np.random.default_rng(0)
    n = 5000
    y = np.zeros(n)
    noise = rng.normal(0, 1, n)
    for t in range(10, n):
        y[t] = 0.6 * y[t - 1] + 0.3 * y[t - 7] + noise[t]

    result = select_significant_lags(y, max_lag=20, max_candidates=2)
    assert result["significant_lags"] == [1, 7], result["significant_lags"]
    assert abs(result["pacf_values"][1]) > abs(result["pacf_values"][7])


def test_select_significant_lags_white_noise_low_false_positive_rate():
    rng = np.random.default_rng(1)
    n = 5000
    white_noise = rng.normal(0, 1, n)
    result = select_significant_lags(white_noise, max_lag=40)
    # expected ~5% false-positive rate at alpha=0.05 over 40 lags -> a handful, not a large fraction.
    assert len(result["significant_lags"]) <= 6


def test_select_significant_lags_invalid_alpha_raises():
    import pytest

    with pytest.raises(ValueError):
        select_significant_lags(np.zeros(100), alpha=0.1)
