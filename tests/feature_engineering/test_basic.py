"""Hypothesis-based tests for basic.py module."""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings

from mlframe.feature_engineering.basic import create_date_features


@pytest.mark.parametrize("df_lib", ["pandas", "polars"])
@given(n_rows=st.integers(min_value=1, max_value=100))
@settings(deadline=None)
def test_create_date_features_adds_columns(df_lib, n_rows):
    """Test that date features are added correctly for pandas and polars."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(n_rows)]
    if df_lib == "pandas":
        df = pd.DataFrame({'date': dates, 'value': range(n_rows)})
    else:
        df = pl.DataFrame({'date': dates, 'value': list(range(n_rows))})
    result = create_date_features(df, ['date'])

    assert 'date_day' in result.columns
    assert 'date_weekday' in result.columns
    assert 'date_month' in result.columns
    assert 'date' not in result.columns  # Original deleted


def test_create_date_features_keep_original():
    """Test keeping original columns."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    df = pd.DataFrame({'date': dates})
    result = create_date_features(df, ['date'], delete_original_cols=False)

    assert 'date' in result.columns
    assert 'date_day' in result.columns


def test_create_date_features_empty_cols():
    """Test with empty columns list."""
    df = pd.DataFrame({'date': [datetime(2023, 1, 1)], 'value': [1]})
    result = create_date_features(df, [])

    # Should return unchanged
    assert list(result.columns) == list(df.columns)


def test_create_date_features_custom_methods():
    """Test with custom methods."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    df = pd.DataFrame({'date': dates})

    result = create_date_features(
        df, ['date'],
        methods={'day': np.int16, 'month': np.int16}
    )

    assert 'date_day' in result.columns
    assert 'date_month' in result.columns
    assert 'date_weekday' not in result.columns


@pytest.mark.parametrize("df_lib", ["pandas", "polars"])
def test_create_date_features_weekday_range(df_lib):
    """Test that weekday values are in correct range (0-6)."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(7)]
    if df_lib == "pandas":
        df = pd.DataFrame({'date': dates})
    else:
        df = pl.DataFrame({'date': dates})
    result = create_date_features(df, ['date'])

    weekdays = result['date_weekday'].to_list() if df_lib == "polars" else result['date_weekday'].values.tolist()
    assert all(0 <= w <= 6 for w in weekdays)


def test_create_date_features_multiple_cols():
    """Test with multiple date columns."""
    dates1 = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)]
    dates2 = [datetime(2023, 6, 1) + timedelta(days=i) for i in range(5)]
    df = pd.DataFrame({'date1': dates1, 'date2': dates2})

    result = create_date_features(df, ['date1', 'date2'])

    assert 'date1_day' in result.columns
    assert 'date2_day' in result.columns
    assert 'date1' not in result.columns
    assert 'date2' not in result.columns


def test_create_date_features_day_values():
    """Test that day values are correct."""
    dates = [datetime(2023, 1, i) for i in range(1, 11)]
    df = pd.DataFrame({'date': dates})
    result = create_date_features(df, ['date'])

    expected_days = list(range(1, 11))
    assert list(result['date_day'].values) == expected_days


def test_create_date_features_month_values():
    """Test that month values are correct."""
    dates = [datetime(2023, i, 1) for i in range(1, 13)]
    df = pd.DataFrame({'date': dates})
    result = create_date_features(df, ['date'])

    expected_months = list(range(1, 13))
    assert list(result['date_month'].values) == expected_months


def test_create_date_features_invalid_df():
    """Test with invalid dataframe type."""
    with pytest.raises(ValueError):
        create_date_features({'not': 'a dataframe'}, ['date'])


@pytest.mark.parametrize("df_lib", ["pandas", "polars"])
def test_create_date_features_dtypes(df_lib):
    """Test that dtypes are correctly applied."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(10)]
    if df_lib == "pandas":
        df = pd.DataFrame({'date': dates})
        expected_dtype = np.int8
    else:
        df = pl.DataFrame({'date': dates})
        expected_dtype = pl.Int8
    result = create_date_features(df, ['date'])

    assert result['date_day'].dtype == expected_dtype
    assert result['date_weekday'].dtype == expected_dtype
    assert result['date_month'].dtype == expected_dtype
