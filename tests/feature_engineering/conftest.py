"""
Shared pytest fixtures for feature engineering tests.
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
from datetime import datetime, timedelta


@pytest.fixture
def sample_dates_df_pandas():
    """Pandas DataFrame with date columns for date feature extraction tests."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]
    return pd.DataFrame({'date': dates, 'value': range(len(dates))})


@pytest.fixture
def sample_dates_df_polars():
    """Polars DataFrame with date columns for date feature extraction tests."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]
    return pl.DataFrame({'date': dates, 'value': list(range(len(dates)))})


@pytest.fixture
def sample_ohlcv_polars():
    """Polars OHLCV DataFrame for financial feature tests."""
    np.random.seed(42)
    n = 100
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    return pl.DataFrame({
        'ticker': ['A'] * n,
        'open': close + np.random.randn(n) * 0.1,
        'high': close + np.abs(np.random.randn(n) * 0.5),
        'low': close - np.abs(np.random.randn(n) * 0.5),
        'close': close,
        'volume': np.abs(np.random.randn(n) * 1000 + 5000),
        'qty': np.abs(np.random.randn(n) * 100 + 500),
    })
