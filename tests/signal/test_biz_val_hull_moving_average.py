"""biz_value test for ``signal.hull_moving_average`` (``hull_moving_average``, ``hull_ma_deviation``).

The win (9th_g-research-crypto-forecasting.md): a plain SMA lags a sharp trend change by roughly half its
window length; the Hull MA construction cancels most of that lag while still smoothing noise. This test
confirms Hull MA tracks a step-change trend (a sudden regime shift) with materially lower lag error than a
plain SMA of the same window, and that the deviation feature (value - HullMA) correctly identifies the
regime-shift point sooner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.signal.hull_moving_average import hull_ma_deviation, hull_moving_average


def _sma(x: np.ndarray, window: int) -> np.ndarray:
    return np.asarray(pd.Series(x).rolling(window=window, min_periods=window).mean().to_numpy())


def _make_step_change_series(n: int, change_point: int, seed: int):
    rng = np.random.default_rng(seed)
    base = np.concatenate([np.full(change_point, 100.0), np.full(n - change_point, 130.0)])
    return base + rng.normal(scale=0.5, size=n)


def test_biz_val_hull_ma_tracks_step_change_with_lower_lag_than_sma():
    window = 20
    n, change_point = 200, 100
    series = _make_step_change_series(n, change_point, seed=0)

    hma = hull_moving_average(series, window)
    sma = _sma(series, window)

    true_level = np.concatenate([np.full(change_point, 100.0), np.full(n - change_point, 130.0)])
    valid = ~np.isnan(hma) & ~np.isnan(sma)

    mae_hma = float(np.mean(np.abs(true_level[valid] - hma[valid])))
    mae_sma = float(np.mean(np.abs(true_level[valid] - sma[valid])))

    assert mae_hma < mae_sma, f"expected Hull MA to track the step-change trend with lower lag error than plain SMA, got hma_mae={mae_hma:.4f} sma_mae={mae_sma:.4f}"


def test_hull_ma_deviation_spikes_at_regime_shift():
    window = 20
    n, change_point = 200, 100
    series = _make_step_change_series(n, change_point, seed=1)

    deviation = hull_ma_deviation(series, window)
    valid = ~np.isnan(deviation)

    # the deviation should be near-zero in the stable pre-change region and spike right after the shift.
    pre_change_region = valid & (np.arange(n) < change_point - 5)
    post_change_spike_region = valid & (np.arange(n) >= change_point) & (np.arange(n) < change_point + 10)

    mean_abs_pre = float(np.mean(np.abs(deviation[pre_change_region])))
    mean_abs_post_spike = float(np.mean(np.abs(deviation[post_change_spike_region])))

    assert mean_abs_post_spike > mean_abs_pre * 3, f"expected the deviation feature to spike right after the regime shift relative to the stable pre-change region, got post={mean_abs_post_spike:.4f} pre={mean_abs_pre:.4f}"


def test_hull_moving_average_nan_prefix_length():
    x = np.arange(50, dtype=np.float64)
    hma = hull_moving_average(x, window=10)
    assert np.isnan(hma[0])
    assert not np.isnan(hma[-1])
