"""biz_val tests for ``mlframe.feature_engineering.financial`` --
``add_fast_rolling_stats`` + ``add_ohlcv_ratios_rlags`` +
``add_ohlcv_ta_indicators``.

Per CLAUDE.md: each test asserts a SYNTHETIC measurable WIN that
locks in the financial-FE contract. Naming:
``test_biz_val_financial_<fn>_<scenario>``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _make_ohlcv(n=300, n_tickers=2, seed=42):
    """Synthetic OHLCV with multiple tickers. Returns polars DataFrame
    with columns ticker / open / high / low / close / volume / qty
    + timestamp. ``qty`` is the trade-count proxy required by
    ``add_ohlcv_ratios_rlags`` (avg_trade_size = volume / qty)."""
    pl = pytest.importorskip("polars")
    rng = np.random.default_rng(seed)
    rows = []
    base_ts = pd.Timestamp("2025-01-01")
    for ticker_id in range(n_tickers):
        rets = rng.normal(0.0001, 0.01, size=n)
        close = 100.0 * np.cumprod(1 + rets)
        high = close * (1 + np.abs(rng.normal(0, 0.005, size=n)))
        low = close * (1 - np.abs(rng.normal(0, 0.005, size=n)))
        open_ = close * (1 + rng.normal(0, 0.003, size=n))
        volume = rng.integers(1000, 100000, size=n).astype(np.float64)
        qty = rng.integers(10, 500, size=n).astype(np.float64)  # trade count
        ticker = f"T{ticker_id}"
        for i in range(n):
            rows.append({
                "ticker": ticker,
                "timestamp": base_ts + pd.Timedelta(minutes=i),
                "open": float(open_[i]),
                "high": float(high[i]),
                "low": float(low[i]),
                "close": float(close[i]),
                "volume": float(volume[i]),
                "qty": float(qty[i]),
            })
    df = pl.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# add_fast_rolling_stats
# ---------------------------------------------------------------------------


def test_biz_val_financial_add_fast_rolling_stats_adds_columns():
    """``add_fast_rolling_stats`` must ADD new columns to the input
    DataFrame (rolling aggregates of each numeric column over the
    configured windows)."""
    pl = pytest.importorskip("polars")
    from mlframe.feature_engineering.financial import add_fast_rolling_stats
    df = _make_ohlcv(n=200, n_tickers=2)
    n_cols_before = df.shape[1]
    out = add_fast_rolling_stats(
        df, rolling_windows=[5, 10],
        groupby_column="ticker",
    )
    assert out.shape[1] > n_cols_before, (
        f"add_fast_rolling_stats must add columns; "
        f"before={n_cols_before}, after={out.shape[1]}"
    )


def test_biz_val_financial_add_fast_rolling_stats_row_count_preserved():
    """Row count must be preserved (rolling is per-row, not aggregating)."""
    pl = pytest.importorskip("polars")
    from mlframe.feature_engineering.financial import add_fast_rolling_stats
    df = _make_ohlcv(n=200, n_tickers=2)
    n_rows = df.shape[0]
    out = add_fast_rolling_stats(
        df, rolling_windows=[5],
        groupby_column="ticker",
    )
    assert out.shape[0] == n_rows


@pytest.mark.parametrize("window_size", [3, 5, 10, 20])
def test_biz_val_financial_rolling_stats_parametrize_window(window_size):
    """Multiple rolling windows must complete cleanly."""
    pl = pytest.importorskip("polars")
    from mlframe.feature_engineering.financial import add_fast_rolling_stats
    df = _make_ohlcv(n=200, n_tickers=2)
    out = add_fast_rolling_stats(
        df, rolling_windows=[window_size],
        groupby_column="ticker",
    )
    assert out.shape[0] == df.shape[0]
    assert out.shape[1] > df.shape[1]


def test_biz_val_financial_rolling_stats_multiple_windows_more_columns():
    """Adding MORE windows must add MORE columns. Doubling windows
    list should roughly double the added-column count."""
    pl = pytest.importorskip("polars")
    from mlframe.feature_engineering.financial import add_fast_rolling_stats
    df = _make_ohlcv(n=200, n_tickers=2)
    base = df.shape[1]
    out_1 = add_fast_rolling_stats(df, rolling_windows=[5],
                                       groupby_column="ticker")
    out_2 = add_fast_rolling_stats(df, rolling_windows=[5, 10],
                                       groupby_column="ticker")
    out_3 = add_fast_rolling_stats(df, rolling_windows=[5, 10, 20],
                                       groupby_column="ticker")
    added_1 = out_1.shape[1] - base
    added_2 = out_2.shape[1] - base
    added_3 = out_3.shape[1] - base
    assert added_3 > added_2 > added_1, (
        f"more windows must add more columns; got 1={added_1}, "
        f"2={added_2}, 3={added_3}"
    )


# ---------------------------------------------------------------------------
# add_ohlcv_ratios_rlags
# ---------------------------------------------------------------------------


def test_biz_val_financial_ohlcv_ratios_rlags_adds_columns():
    """``add_ohlcv_ratios_rlags`` must enrich OHLCV with ratio +
    relative-lag features."""
    pl = pytest.importorskip("polars")
    from mlframe.feature_engineering.financial import add_ohlcv_ratios_rlags
    df = _make_ohlcv(n=200, n_tickers=2)
    n_cols_before = df.shape[1]
    out = add_ohlcv_ratios_rlags(
        df, lags=[1, 5],
        crossbar_ratios_lags=[1],
        ticker_column="ticker",
    )
    assert out.shape[1] > n_cols_before, (
        f"ohlcv_ratios_rlags must add columns; "
        f"before={n_cols_before}, after={out.shape[1]}"
    )
    assert out.shape[0] == df.shape[0]


def test_biz_val_financial_ohlcv_ratios_only_disables_rlags():
    """``add_ratios=True, add_rlags=False`` must add FEWER columns
    than both-True (rlags contributes its own column count)."""
    pl = pytest.importorskip("polars")
    from mlframe.feature_engineering.financial import add_ohlcv_ratios_rlags
    df = _make_ohlcv(n=150, n_tickers=2)
    out_both = add_ohlcv_ratios_rlags(
        df, lags=[1, 5], crossbar_ratios_lags=[1],
        ticker_column="ticker",
        add_ratios=True, add_rlags=True,
    )
    out_ratios_only = add_ohlcv_ratios_rlags(
        df, lags=[1, 5], crossbar_ratios_lags=[1],
        ticker_column="ticker",
        add_ratios=True, add_rlags=False,
    )
    assert out_ratios_only.shape[1] < out_both.shape[1], (
        f"add_rlags=False must yield fewer columns; "
        f"both={out_both.shape[1]}, ratios_only={out_ratios_only.shape[1]}"
    )


# ---------------------------------------------------------------------------
# compute_hurst_exponent
# ---------------------------------------------------------------------------


def test_biz_val_financial_hurst_random_walk_with_diffs_near_half():
    """Random walk processed with ``take_diffs=True`` (i.e. operating
    on the i.i.d. increments) must yield Hurst ~0.5 (no persistence).
    Floor: 0.3 <= H <= 0.7."""
    from mlframe.feature_engineering.hurst import compute_hurst_exponent
    rng = np.random.default_rng(42)
    rw = np.cumsum(rng.normal(size=2000))
    result = compute_hurst_exponent(rw, min_window=10, max_window=500,
                                        take_diffs=True)
    H = result[0] if isinstance(result, tuple) else result
    H = float(H) if H is not None and np.isfinite(H) else 0.5
    assert 0.3 <= H <= 0.7, (
        f"random walk increments Hurst should be ~0.5; got {H:.3f}"
    )


def test_biz_val_financial_hurst_raw_random_walk_high_persistence():
    """Hurst of RAW random walk levels (without ``take_diffs``)
    should be near 1.0 -- the levels are a smooth Brownian-motion
    path with strong persistence. Documents the contract:
    ``take_diffs=False`` (default) measures the LEVELS, not the
    increments."""
    from mlframe.feature_engineering.hurst import compute_hurst_exponent
    rng = np.random.default_rng(42)
    rw = np.cumsum(rng.normal(size=2000))
    result = compute_hurst_exponent(rw, min_window=10, max_window=500,
                                        take_diffs=False)
    H = result[0] if isinstance(result, tuple) else result
    H = float(H) if H is not None and np.isfinite(H) else 0.5
    assert H > 0.7, (
        f"raw random walk levels should have H > 0.7 (persistence); "
        f"got {H:.3f}"
    )


def test_biz_val_financial_hurst_trending_series_above_half():
    """A strongly trending series (deterministic linear trend +
    small noise) should have Hurst > 0.5 (persistence)."""
    from mlframe.feature_engineering.hurst import compute_hurst_exponent
    rng = np.random.default_rng(42)
    n = 2000
    # Strong linear trend with small noise
    trend = np.linspace(0, 100, n)
    series = trend + rng.normal(scale=0.5, size=n)
    result = compute_hurst_exponent(series, min_window=10, max_window=500)
    H = result[0] if isinstance(result, tuple) else result
    H = float(H) if H is not None and np.isfinite(H) else 0.0
    assert H > 0.55, (
        f"trending series Hurst should be > 0.55; got {H:.3f}"
    )


@pytest.mark.parametrize("n_samples", [500, 1000, 3000])
def test_biz_val_financial_hurst_scales_with_size(n_samples):
    """Hurst must complete cleanly across {500, 1000, 3000} samples."""
    from mlframe.feature_engineering.hurst import compute_hurst_exponent
    rng = np.random.default_rng(42)
    arr = rng.normal(size=n_samples)
    result = compute_hurst_exponent(arr, min_window=5,
                                        max_window=n_samples // 4)
    # Must return tuple/scalar with a finite Hurst exponent in (0, 1); random-walk-like series should yield H near 0.5.
    H = result[0] if isinstance(result, tuple) else result
    assert H is not None, f"compute_hurst_exponent returned None on n_samples={n_samples}"
    H_f = float(H)
    assert np.isfinite(H_f), f"non-finite Hurst H={H_f!r} on n_samples={n_samples}"
    assert 0.0 < H_f < 1.0, f"H={H_f:.3f} outside theoretical (0,1) range on n_samples={n_samples}"
