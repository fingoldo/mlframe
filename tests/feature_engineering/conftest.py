"""
Shared pytest fixtures for feature engineering tests.
"""

import matplotlib

# Set the non-GUI matplotlib backend at COLLECTION time, before any test imports plot_positions
# / fig.show() etc. Without this the mps plotting tests would crash on headless CI machines
# or pop interactive windows during local runs.
matplotlib.use("Agg")

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


@pytest.fixture(scope="session")
def sample_ohlcv_polars():
    """Polars OHLCV DataFrame for financial feature tests; session-scoped."""
    # Local Generator instead of mutating np.random global state; data is
    # deterministic + read-only so session scope is safe.
    rng = np.random.default_rng(42)
    n = 100
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pl.DataFrame({
        'ticker': ['A'] * n,
        'open': close + rng.standard_normal(n) * 0.1,
        'high': close + np.abs(rng.standard_normal(n) * 0.5),
        'low': close - np.abs(rng.standard_normal(n) * 0.5),
        'close': close,
        'volume': np.abs(rng.standard_normal(n) * 1000 + 5000),
        'qty': np.abs(rng.standard_normal(n) * 100 + 500),
    })
