"""Hypothesis-based tests for financial.py module."""

import pytest
import numpy as np
import polars as pl
from hypothesis import given, strategies as st, settings

from mlframe.feature_engineering.financial import (
    add_ohlcv_ratios_rlags,
    add_fast_rolling_stats,
)


def create_sample_ohlcv(n_rows: int) -> pl.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n_rows) * 0.5)
    return pl.DataFrame({
        'ticker': ['A'] * n_rows,
        'open': close + np.random.randn(n_rows) * 0.1,
        'high': close + np.abs(np.random.randn(n_rows) * 0.5),
        'low': close - np.abs(np.random.randn(n_rows) * 0.5),
        'close': close,
        'volume': np.abs(np.random.randn(n_rows) * 1000 + 5000),
        'qty': np.abs(np.random.randn(n_rows) * 100 + 500),
    })


@given(st.integers(min_value=10, max_value=100))
@settings(max_examples=20, deadline=None)
def test_add_ohlcv_ratios_rlags_adds_columns(n_rows):
    """Test that function adds expected columns."""
    df = create_sample_ohlcv(n_rows)
    result = add_ohlcv_ratios_rlags(df)
    assert len(result.columns) > len(df.columns)


@given(st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=3, unique=True))
@settings(max_examples=20, deadline=None)
def test_add_ohlcv_ratios_custom_lags(lags):
    """Test with custom lag values."""
    df = create_sample_ohlcv(50)
    result = add_ohlcv_ratios_rlags(df, lags=lags)
    # Should have columns for each lag
    for lag in lags:
        lag_cols = [c for c in result.columns if f'_rlag{lag}' in c]
        assert len(lag_cols) > 0


@given(st.booleans(), st.booleans())
def test_add_ohlcv_ratios_options(add_ratios, add_rlags):
    """Test different ratio/rlag options."""
    df = create_sample_ohlcv(30)
    result = add_ohlcv_ratios_rlags(df, add_ratios=add_ratios, add_rlags=add_rlags)

    if not add_ratios and not add_rlags:
        # Should return original columns
        assert len(result.columns) == len(df.columns)


def test_add_ohlcv_ratios_no_nans_in_added():
    """Test that added columns don't have unexpected NaNs."""
    df = create_sample_ohlcv(100)
    result = add_ohlcv_ratios_rlags(df, nans_filler=0.0)

    # After first few rows (due to lags), shouldn't have NaNs
    for col in result.columns:
        if col not in df.columns:
            # New columns should be filled
            nan_count = result[col].null_count()
            # Allow some nulls due to lag initialization
            assert nan_count <= 5, f"Column {col} has {nan_count} nulls"


@given(st.lists(st.integers(min_value=2, max_value=20), min_size=1, max_size=3, unique=True))
@settings(max_examples=20, deadline=None)
def test_add_fast_rolling_stats_windows(windows):
    """Test rolling stats with various window sizes."""
    df = pl.DataFrame({
        'a': np.random.rand(100),
        'b': np.random.rand(100),
    })
    result = add_fast_rolling_stats(df, rolling_windows=windows)
    assert result is not None
    assert len(result.columns) > len(df.columns)


@given(st.booleans())
def test_add_fast_rolling_stats_relative(relative):
    """Test relative vs absolute mode."""
    df = pl.DataFrame({'a': np.random.rand(50) + 1})
    result = add_fast_rolling_stats(df, relative=relative, rolling_windows=[5])
    assert len(result.columns) > 1


def test_add_fast_rolling_stats_custom_numaggs():
    """Test with custom aggregation functions."""
    df = pl.DataFrame({'a': np.random.rand(50)})
    numaggs = ['rolling_mean', 'rolling_std']
    result = add_fast_rolling_stats(df, numaggs=numaggs, rolling_windows=[5])

    # Should have columns for each numagg
    mean_cols = [c for c in result.columns if 'mean' in c]
    std_cols = [c for c in result.columns if 'std' in c]
    assert len(mean_cols) > 0
    assert len(std_cols) > 0


def test_add_ohlcv_ratios_cast_f64_to_f32():
    """Test float64 to float32 casting option."""
    df = create_sample_ohlcv(30)
    result = add_ohlcv_ratios_rlags(df, cast_f64_to_f32=True)

    # Check that numeric columns are float32
    for col in result.columns:
        dtype = result[col].dtype
        if dtype in [pl.Float64, pl.Float32]:
            assert dtype == pl.Float32, f"Column {col} is {dtype}, expected Float32"


def test_add_ohlcv_ratios_exclude_fields():
    """Test excluding specific fields."""
    df = create_sample_ohlcv(30)
    result = add_ohlcv_ratios_rlags(df, exclude_fields=['volume', 'qty'])

    # Excluded fields shouldn't have rlags
    volume_rlags = [c for c in result.columns if 'volume' in c and 'rlag' in c]
    qty_rlags = [c for c in result.columns if 'qty' in c and 'rlag' in c]
    assert len(volume_rlags) == 0
    assert len(qty_rlags) == 0


def test_add_fast_rolling_stats_groupby():
    """Test with groupby column."""
    df = pl.DataFrame({
        'ticker': ['A'] * 50 + ['B'] * 50,
        'value': np.random.rand(100),
    })
    result = add_fast_rolling_stats(df, groupby_column='ticker', rolling_windows=[5])
    assert len(result) == 100


def test_add_ohlcv_ratios_multiple_tickers():
    """Test with multiple tickers."""
    df1 = create_sample_ohlcv(50)
    df2 = create_sample_ohlcv(50).with_columns(pl.lit('B').alias('ticker'))
    df = pl.concat([df1, df2])

    result = add_ohlcv_ratios_rlags(df)
    assert len(result) == 100
    assert len(result.columns) > len(df.columns)
