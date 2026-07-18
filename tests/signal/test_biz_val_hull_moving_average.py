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

from mlframe.signal.hull_moving_average import hull_ma_deviation, hull_moving_average, hull_moving_average_multi


def _sma(x: np.ndarray, window: int) -> np.ndarray:
    """Helper that sma."""
    return np.asarray(pd.Series(x).rolling(window=window, min_periods=window).mean().to_numpy())


def _make_step_change_series(n: int, change_point: int, seed: int):
    """Helper that make step change series."""
    rng = np.random.default_rng(seed)
    base = np.concatenate([np.full(change_point, 100.0), np.full(n - change_point, 130.0)])
    return base + rng.normal(scale=0.5, size=n)


def test_biz_val_hull_ma_tracks_step_change_with_lower_lag_than_sma():
    """Hull ma tracks step change with lower lag than sma."""
    window = 20
    n, change_point = 200, 100
    series = _make_step_change_series(n, change_point, seed=0)

    hma = hull_moving_average(series, window)
    sma = _sma(series, window)

    true_level = np.concatenate([np.full(change_point, 100.0), np.full(n - change_point, 130.0)])
    valid = ~np.isnan(hma) & ~np.isnan(sma)

    mae_hma = float(np.mean(np.abs(true_level[valid] - hma[valid])))
    mae_sma = float(np.mean(np.abs(true_level[valid] - sma[valid])))

    assert (
        mae_hma < mae_sma
    ), f"expected Hull MA to track the step-change trend with lower lag error than plain SMA, got hma_mae={mae_hma:.4f} sma_mae={mae_sma:.4f}"


def test_hull_ma_deviation_spikes_at_regime_shift():
    """Hull ma deviation spikes at regime shift."""
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

    assert (
        mean_abs_post_spike > mean_abs_pre * 3
    ), f"expected the deviation feature to spike right after the regime shift relative to the stable pre-change region, got post={mean_abs_post_spike:.4f} pre={mean_abs_pre:.4f}"


def test_hull_moving_average_nan_prefix_length():
    """Hull moving average nan prefix length."""
    x = np.arange(50, dtype=np.float64)
    hma = hull_moving_average(x, window=10)
    assert np.isnan(hma[0])
    assert not np.isnan(hma[-1])


def test_hull_moving_average_multi_matches_single_window_calls_bit_identical():
    """``hull_moving_average_multi`` shares the fast/slow-SMA cumsum across windows internally -- verify
    that restructuring didn't change a single bit of the per-window result vs the original single-window
    entry point."""
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.normal(size=300)) + 100
    windows = [7, 15, 32, 50]

    multi = hull_moving_average_multi(x, windows)
    for window in windows:
        single = hull_moving_average(x, window)
        combined = np.stack([multi[window], single])
        assert np.array_equal(combined[0], combined[1], equal_nan=True), f"window={window} diverged between hull_moving_average and hull_moving_average_multi"


def _sign_flip_count(signal: np.ndarray, region: np.ndarray, valid: np.ndarray) -> int:
    """Helper that sign flip count."""
    idx = np.where(region & valid)[0]
    return int(np.sum(np.diff(signal[idx]) != 0))


def test_biz_val_hull_moving_average_multi_crossover_beats_single_window_signal():
    """Real usage: a fast/slow Hull MA pair computed together (``hull_moving_average_multi``) drives a
    crossover regime signal (``sign(hma_fast - hma_slow)``) -- the standard fast/slow-MA-pair trend
    detector. A single-window 'value crosses its own HMA' signal is far noisier: every brief spike pushes
    the raw value above/below its own (fast-reacting) HMA and flips the signal, even when nothing about the
    underlying trend changed. The two-window crossover barely reacts to the same brief spikes (both windows
    absorb them similarly) yet still flips essentially immediately at a REAL, sustained regime shift --
    fewer false signals with no added detection latency, only obtainable by having both windows available
    from one call."""
    n, change_point = 300, 150
    fast_window, slow_window = 8, 45
    rng = np.random.default_rng(3)

    base = np.concatenate([np.full(change_point, 100.0), np.full(n - change_point, 130.0)])
    series = base + rng.normal(scale=0.5, size=n)

    # three short-lived noise bursts well before the real regime shift, unrelated to any real trend change.
    burst_centers = [40, 90, 230]
    burst_half_width = 2
    for center in burst_centers:
        series[center - burst_half_width : center + burst_half_width + 1] += 25.0

    multi = hull_moving_average_multi(series, [fast_window, slow_window])
    fast_hma, slow_hma = multi[fast_window], multi[slow_window]
    valid = ~np.isnan(fast_hma) & ~np.isnan(slow_hma)

    naive_signal = np.sign(series - fast_hma)
    crossover_signal = np.sign(fast_hma - slow_hma)

    burst_region = np.zeros(n, dtype=bool)
    for center in burst_centers:
        burst_region[center - 8 : center + 8] = True
    burst_region[change_point - 5 :] = False  # only pre-change bursts count as pure false-flip traps

    naive_false_flips = _sign_flip_count(naive_signal, burst_region, valid)
    crossover_false_flips = _sign_flip_count(crossover_signal, burst_region, valid)

    assert naive_false_flips > 0, "test setup should make the single-window signal flip on noise bursts"
    assert (
        crossover_false_flips < naive_false_flips
    ), f"expected the fast/slow crossover signal to flip less often than the single-window signal on noise bursts, got crossover={crossover_false_flips} naive={naive_false_flips}"

    # detection latency: index of the first crossover flip to the new (upward) trend direction after the real shift.
    pre_change_idx = np.where(valid & (np.arange(n) < change_point))[0]
    pre_change_sign = crossover_signal[pre_change_idx[-1]]
    post_change_idx = np.where(valid & (np.arange(n) >= change_point))[0]
    post_change_signal = crossover_signal[post_change_idx]
    flip_positions = np.where(post_change_signal != pre_change_sign)[0]
    assert flip_positions.size > 0, "crossover signal never flips to the new trend direction after the real regime shift"
    detection_lag = int(post_change_idx[flip_positions[0]]) - change_point
    assert detection_lag <= 5, f"expected the crossover signal to catch the real regime shift almost immediately, got detection_lag={detection_lag}"
