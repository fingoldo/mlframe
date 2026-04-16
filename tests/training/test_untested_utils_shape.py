"""Tests for mlframe.training.utils + preprocessing shape/memory helpers.

Targets:
- estimate_df_size_mb
- create_split_dataframes
- drop_columns_from_dataframe
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.utils import estimate_df_size_mb, drop_columns_from_dataframe
from mlframe.training.preprocessing import create_split_dataframes


# ----- estimate_df_size_mb -----

def test_estimate_pandas_nonneg():
    df = pd.DataFrame({"a": np.arange(1000), "b": np.random.default_rng(0).standard_normal(1000)})
    size = estimate_df_size_mb(df)
    assert isinstance(size, float)
    assert size > 0
    assert size < 1.0  # tiny frame, under 1 MB


def test_estimate_polars_nonneg():
    df = pl.DataFrame({"a": list(range(1000))})
    assert estimate_df_size_mb(df) >= 0.0


def test_estimate_unsupported_inf():
    arr = np.zeros((10, 3))
    assert estimate_df_size_mb(arr) == float("inf")


def test_estimate_empty_pandas():
    df = pd.DataFrame()
    assert estimate_df_size_mb(df) >= 0.0


# ----- create_split_dataframes -----

def test_create_split_pandas():
    df = pd.DataFrame({"x": np.arange(100), "y": np.arange(100, 200)})
    tr_idx = np.arange(0, 60)
    va_idx = np.arange(60, 80)
    te_idx = np.arange(80, 100)
    tr, va, te = create_split_dataframes(df, tr_idx, va_idx, te_idx)
    assert len(tr) == 60 and len(va) == 20 and len(te) == 20
    assert tr["x"].iloc[0] == 0
    assert va["x"].iloc[0] == 60


def test_create_split_polars():
    df = pl.DataFrame({"x": list(range(50))})
    tr, va, te = create_split_dataframes(
        df, np.arange(0, 30), np.arange(30, 40), np.arange(40, 50)
    )
    assert tr.height == 30 and va.height == 10 and te.height == 10


def test_create_split_empty_val_test_pandas():
    df = pd.DataFrame({"x": np.arange(10)})
    tr, va, te = create_split_dataframes(df, np.arange(10), np.array([], dtype=int), np.array([], dtype=int))
    assert len(tr) == 10
    assert isinstance(va, pd.DataFrame) and va.empty
    assert isinstance(te, pd.DataFrame) and te.empty


def test_create_split_empty_val_test_polars():
    df = pl.DataFrame({"x": list(range(10))})
    tr, va, te = create_split_dataframes(df, np.arange(10), np.array([], dtype=int), np.array([], dtype=int))
    assert tr.height == 10
    assert isinstance(va, pl.DataFrame) and va.is_empty()


# ----- drop_columns_from_dataframe -----

def test_drop_cols_pandas_both_args():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})
    out = drop_columns_from_dataframe(df, additional_columns_to_drop=["a"], config_drop_columns=["b"], verbose=0)
    assert set(out.columns) == {"c", "d"}


def test_drop_cols_polars():
    df = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
    out = drop_columns_from_dataframe(df, additional_columns_to_drop=["a", "nonexistent"], config_drop_columns=None, verbose=0)
    # polars drop(strict=False) handles missing
    assert "a" not in out.columns
    assert set(out.columns) == {"b", "c"}


def test_drop_cols_pandas_nonexistent_silently_skipped():
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = drop_columns_from_dataframe(df, additional_columns_to_drop=["nope"], config_drop_columns=None, verbose=0)
    # nonexistent col silently skipped — code uses intersection
    assert set(out.columns) == {"a", "b"}


def test_drop_cols_both_none_returns_df_unchanged():
    df = pd.DataFrame({"a": [1]})
    out = drop_columns_from_dataframe(df, additional_columns_to_drop=None, config_drop_columns=None, verbose=0)
    assert out is df


def test_drop_cols_duplicate_names_deduped():
    df = pd.DataFrame({"a": [1], "b": [2]})
    out = drop_columns_from_dataframe(
        df, additional_columns_to_drop=["a", "a"], config_drop_columns=["a"], verbose=0
    )
    assert list(out.columns) == ["b"]
