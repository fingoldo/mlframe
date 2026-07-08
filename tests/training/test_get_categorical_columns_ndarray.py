"""Regression: ``get_categorical_columns`` must not crash on a bare ndarray input.

Pre-fix (fuzz c0097_c88d18a0), the pandas branch had no ``isinstance(df, pd.DataFrame)`` guard --
it unconditionally called ``df.select_dtypes(...)`` whenever ``df`` was not a polars DataFrame. A
pre_pipeline that outputs numpy (sklearn's default array output) reaches ``run_confidence_analysis``
with a transformed ndarray test split, and ``get_categorical_columns(ndarray, ...)`` raised
``AttributeError: 'numpy.ndarray' object has no attribute 'select_dtypes'``.

A bare ndarray has no column names/dtypes to inspect, so the fix returns an empty list rather than
crashing -- matching the function's job (categorical COLUMN NAMES; an ndarray has none).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from mlframe.training._nan_processing import get_categorical_columns


def test_ndarray_input_returns_empty_list_instead_of_raising():
    arr = np.random.default_rng(0).normal(size=(20, 4))
    assert get_categorical_columns(arr) == []
    assert get_categorical_columns(arr, include_string=False) == []


def test_pandas_path_still_works():
    df = pd.DataFrame({"num": [1.0, 2.0], "cat": pd.Categorical(["a", "b"])})
    assert get_categorical_columns(df, include_string=False) == ["cat"]


def test_polars_path_still_works():
    df = pl.DataFrame({"num": [1.0, 2.0], "cat": pl.Series(["a", "b"], dtype=pl.Categorical)})
    assert get_categorical_columns(df, include_string=False) == ["cat"]
