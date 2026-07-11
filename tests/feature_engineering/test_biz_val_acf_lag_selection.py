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


def test_biz_val_select_significant_lags_per_group_consensus_recovers_both_lags():
    """Panel with two entities of genuinely different AR structure: group A driven by lag 3, group B by lag 7.

    Pooling both entities into one global series (the default, group-blind mode) washes out one entity's
    lag structure whenever the other entity's coefficient/variance dominates the pooled PACF -- exactly the
    under/over-fitting failure the per-group consensus mode exists to avoid. The per-group mode must recover
    BOTH true generating lags {3, 7}; the global-only mode on the same pooled data must miss at least one.
    """
    rng = np.random.default_rng(42)
    n_per_group = 4000

    y_a = np.zeros(n_per_group)
    noise_a = rng.normal(0, 1, n_per_group)
    for t in range(10, n_per_group):
        y_a[t] = 0.75 * y_a[t - 3] + noise_a[t]

    # group B: much larger innovation variance so its own weaker lag-7 dependence is swamped once pooled
    # with group A into a single global series -- this is the realistic panel failure mode.
    y_b = np.zeros(n_per_group)
    noise_b = rng.normal(0, 8, n_per_group)
    for t in range(10, n_per_group):
        y_b[t] = 0.55 * y_b[t - 7] + noise_b[t]

    series = np.concatenate([y_a, y_b])
    groups = np.concatenate([np.zeros(n_per_group, dtype=int), np.ones(n_per_group, dtype=int)])

    per_group_result = select_significant_lags(series, max_lag=15, groups=groups)
    assert per_group_result["n_groups_scored"] == 2
    assert 3 in per_group_result["significant_lags"], per_group_result["significant_lags"]
    assert 7 in per_group_result["significant_lags"], per_group_result["significant_lags"]

    global_result = select_significant_lags(series, max_lag=15)
    assert not ({3, 7} <= set(global_result["significant_lags"])), (
        f"expected the pooled global-only result to miss at least one of {{3, 7}}, got {global_result['significant_lags']}"
    )


def test_select_significant_lags_default_unchanged_when_groups_omitted():
    """Opt-in gate: omitting ``groups`` must reproduce the pre-existing single-series result bit-identically."""
    rng = np.random.default_rng(7)
    series = rng.normal(0, 1, 2000)
    for t in range(5, 2000):
        series[t] += 0.5 * series[t - 4]

    baseline = select_significant_lags(series, max_lag=20)
    same = select_significant_lags(series, max_lag=20, groups=None)

    assert baseline["significant_lags"] == same["significant_lags"]
    assert baseline["pacf_values"] == same["pacf_values"]
    assert baseline["significance_band"] == same["significance_band"]
