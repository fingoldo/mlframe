"""Sensor: detect_cat_columns_by_dtype matches mlframe's existing
categorical-like dtype convention (_phase_helpers.py:920-931).

Reconciles the docstring-drift agent's wave-10 #4 finding: pre-fix
``feature_handling_apply(candidate_cat_columns=None)`` fell to ``cat_cols = []``,
silently dropping every target_mean / WoE handler the FHC was configured for.
Post-fix the by-dtype detector populates cat_cols using the same convention
the rest of mlframe uses to declare cats.

Convention (from _phase_helpers.py:920-931):
- Polars: pl.Categorical OR pl.Enum OR pl.Utf8 OR pl.String -> cat
- Pandas: select_dtypes(['category','object','string']) -> cat
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from mlframe.training.feature_handling.text_detection import detect_cat_columns_by_dtype


# ---- pandas paths --------------------------------------------------------


def test_pandas_category_dtype_detected():
    df = pd.DataFrame(
        {
            "num": np.arange(10),
            "cat": pd.Categorical(["A", "B"] * 5),
        }
    )
    out = detect_cat_columns_by_dtype(df)
    assert "cat" in out
    assert "num" not in out


def test_pandas_object_dtype_detected():
    df = pd.DataFrame(
        {
            "num": np.arange(5),
            "obj": ["x", "y", "z", "x", "y"],
        }
    )
    out = detect_cat_columns_by_dtype(df)
    assert "obj" in out


def test_pandas_string_dtype_detected():
    df = pd.DataFrame(
        {
            "num": np.arange(5),
            "str_ext": pd.array(["x", "y", "z", "x", "y"], dtype=pd.StringDtype()),
        }
    )
    out = detect_cat_columns_by_dtype(df)
    assert "str_ext" in out


def test_pandas_pure_numeric_excluded():
    df = pd.DataFrame(
        {
            "f": np.arange(5, dtype=np.float64),
            "i": np.arange(5, dtype=np.int64),
            "b": [True, False] * 2 + [True],
        }
    )
    out = detect_cat_columns_by_dtype(df)
    assert out == [], f"numeric/bool must NOT be detected as cat (would force-promote low-card int): {out}"


def test_pandas_exclude_columns_dropped_from_output():
    """text-detected cols passed via exclude_columns must NOT appear in cat output."""
    df = pd.DataFrame(
        {
            "cat_x": pd.Categorical(["A", "B"] * 5),
            "text_y": ["long sentence here"] * 10,
        }
    )
    out = detect_cat_columns_by_dtype(df, exclude_columns=["text_y"])
    assert "cat_x" in out
    assert "text_y" not in out


# ---- polars paths --------------------------------------------------------


def test_polars_categorical_dtype_detected():
    df = pl.DataFrame(
        {
            "num": list(range(10)),
            "cat": pl.Series(["A", "B"] * 5, dtype=pl.Categorical),
        }
    )
    out = detect_cat_columns_by_dtype(df)
    assert "cat" in out
    assert "num" not in out


def test_polars_enum_dtype_detected():
    enum = pl.Enum(["A", "B"])
    df = pl.DataFrame(
        {
            "num": list(range(4)),
            "enum_col": pl.Series(["A", "B", "A", "B"], dtype=enum),
        }
    )
    out = detect_cat_columns_by_dtype(df)
    assert "enum_col" in out


def test_polars_string_utf8_detected():
    df = pl.DataFrame(
        {
            "num": [1.0, 2.0],
            "s": ["x", "y"],
        }
    )
    out = detect_cat_columns_by_dtype(df)
    assert "s" in out


def test_polars_pure_numeric_excluded():
    df = pl.DataFrame(
        {
            "f": [1.0, 2.0, 3.0],
            "i": [1, 2, 3],
        }
    )
    out = detect_cat_columns_by_dtype(df)
    assert out == []


def test_polars_exclude_text_cols():
    df = pl.DataFrame(
        {
            "cat_x": pl.Series(["A", "B"] * 3, dtype=pl.Categorical),
            "text_y": ["long sentence"] * 6,
        }
    )
    out = detect_cat_columns_by_dtype(df, exclude_columns=["text_y"])
    assert "cat_x" in out
    assert "text_y" not in out


# ---- parity with mlframe convention ----------------------------------------


def test_parity_with_existing_phase_helpers_convention():
    """The convention at _phase_helpers.py:920-931 is what this helper extracts.

    Polars: Categorical | Enum | Utf8 | String. Pandas: category | object | string.
    Both backends should produce the same set of names for equivalent input.
    """
    pd_df = pd.DataFrame(
        {
            "num": np.arange(5),
            "cat": pd.Categorical(["A", "B", "A", "B", "A"]),
            "obj": ["x", "y", "z", "x", "y"],
        }
    )
    pl_df = pl.DataFrame(
        {
            "num": list(range(5)),
            "cat": pl.Series(["A", "B", "A", "B", "A"], dtype=pl.Categorical),
            "obj": ["x", "y", "z", "x", "y"],
        }
    )
    pd_out = set(detect_cat_columns_by_dtype(pd_df))
    pl_out = set(detect_cat_columns_by_dtype(pl_df))
    assert pd_out == pl_out == {"cat", "obj"}, f"backend parity broken: pd={pd_out} pl={pl_out}"
