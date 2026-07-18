"""Sensor: remove_constant_columns parity between pandas and polars.

Agent finding #2 of pandas/polars asymmetry audit claimed a divergence on
"single-real-value + many nulls" categorical columns. This test characterises
the ACTUAL behaviour per backend on a matrix of input shapes and verifies the
two paths agree on which columns to drop.

If the test fails on a shape, that's a real asymmetry to investigate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from mlframe.training._nan_processing import remove_constant_columns


def _columns(df) -> set:
    """Columns."""
    return set(df.columns)


# ---- Shapes that should agree across backends -----------------------------


def test_constant_numeric_dropped_both_backends():
    """Constant numeric dropped both backends."""
    pd_df = pd.DataFrame({"x": [1, 2, 3], "const": [7.0, 7.0, 7.0]})
    pl_df = pl.DataFrame({"x": [1, 2, 3], "const": [7.0, 7.0, 7.0]})
    assert "const" not in _columns(remove_constant_columns(pd_df, verbose=0))
    assert "const" not in _columns(remove_constant_columns(pl_df, verbose=0))


def test_varying_numeric_kept_both_backends():
    """Varying numeric kept both backends."""
    pd_df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    pl_df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    assert "x" in _columns(remove_constant_columns(pd_df, verbose=0))
    assert "x" in _columns(remove_constant_columns(pl_df, verbose=0))


def test_all_nan_numeric_dropped_both_backends():
    """All-NaN numeric column carries no info, must be dropped on both backends."""
    pd_df = pd.DataFrame({"x": [1, 2, 3], "all_nan": [np.nan, np.nan, np.nan]})
    pl_df = pl.DataFrame({"x": [1, 2, 3], "all_nan": [None, None, None]}, schema={"x": pl.Int64, "all_nan": pl.Float64})
    assert "all_nan" not in _columns(remove_constant_columns(pd_df, verbose=0))
    assert "all_nan" not in _columns(remove_constant_columns(pl_df, verbose=0))


def test_two_distinct_values_kept_both_backends():
    """2 distinct non-null values is informative; must be KEPT on both backends."""
    pd_df = pd.DataFrame({"x": [1, 2, 3], "binary": ["A", "B", "A"]})
    pl_df = pl.DataFrame({"x": [1, 2, 3], "binary": ["A", "B", "A"]})
    assert "binary" in _columns(remove_constant_columns(pd_df, verbose=0))
    assert "binary" in _columns(remove_constant_columns(pl_df, verbose=0))


def test_single_real_value_no_nulls_dropped_both_backends():
    """Pure constant string column: 1 distinct value, 0 nulls. Both must drop."""
    pd_df = pd.DataFrame({"x": [1, 2, 3], "const_str": ["A", "A", "A"]})
    pl_df = pl.DataFrame({"x": [1, 2, 3], "const_str": ["A", "A", "A"]})
    assert "const_str" not in _columns(remove_constant_columns(pd_df, verbose=0))
    assert "const_str" not in _columns(remove_constant_columns(pl_df, verbose=0))


# ---- Edge case: one-real-value + many-nulls. Document actual behaviour -----


def test_single_real_value_many_nulls_string_parity():
    """A column with one real value 'A' and 4 NaN/null rows.

    'Constant' is ambiguous here: pandas nunique(dropna=False)==2 (A + NaN) so
    NOT constant by current pandas semantics. Polars n_unique() also counts
    null as a level => also returns 2 => also NOT constant. Both KEEP.

    If a future refactor changes either side's semantics, this test surfaces
    the divergence at CI time rather than at production debug."""
    pd_df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "s": ["A", None, None, None, None]})
    pl_df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "s": ["A", None, None, None, None]})
    pd_out_cols = _columns(remove_constant_columns(pd_df, verbose=0))
    pl_out_cols = _columns(remove_constant_columns(pl_df, verbose=0))
    assert pd_out_cols == pl_out_cols, f"single-real-many-null divergence:\n  pandas: {pd_out_cols}\n  polars: {pl_out_cols}"


# ---- Fused nanmin/nanmax numeric constant-detection identity (2026-06-23 perf) --


def _old_pandas_numeric_constants(df: pd.DataFrame) -> set:
    """Pre-optimization reference: separate Series.min/max + isna().all() passes."""
    numeric_cols = df.select_dtypes(include="number").columns
    constant = [c for c in numeric_cols if df[c].min() == df[c].max()]
    all_nan = [c for c in numeric_cols if c not in constant and df[c].isna().all()]
    return set(constant) | set(all_nan)


def test_fused_numeric_constant_detection_matches_old_reference():
    """The fused nanmin/nanmax path must drop EXACTLY the same numeric columns as
    the prior (min==max) + (isna().all()) union, across const / all-NaN /
    mixed-NaN / one-value-rest-NaN / inf / int / varying columns."""
    df = pd.DataFrame(
        {
            "const_float": [2.0, 2.0, 2.0, 2.0, 2.0],
            "const_int": [7, 7, 7, 7, 7],
            "all_nan": [np.nan, np.nan, np.nan, np.nan, np.nan],
            "one_val_rest_nan": [1.0, np.nan, np.nan, np.nan, 1.0],
            "varying": [1.0, 2.0, 3.0, np.nan, 5.0],
            "two_int": [1, 1, 1, 2, 1],
            "inf_const": [np.inf, np.inf, np.inf, np.inf, np.inf],
            "inf_mix": [np.inf, 1.0, 1.0, 1.0, 1.0],
        }
    )
    expected_dropped = _old_pandas_numeric_constants(df)
    kept = _columns(remove_constant_columns(df, verbose=0))
    actually_dropped = set(df.columns) - kept
    assert (
        actually_dropped == expected_dropped
    ), f"fused path diverged from reference:\n  fused dropped: {actually_dropped}\n  reference dropped: {expected_dropped}"


def test_single_real_value_many_nulls_categorical_parity():
    """Single real value many nulls categorical parity."""
    pd_df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "cat": pd.Categorical(["A", None, None, None, None]),
        }
    )
    pl_df = pl.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "cat": pl.Series(["A", None, None, None, None], dtype=pl.Categorical),
        }
    )
    pd_out_cols = _columns(remove_constant_columns(pd_df, verbose=0))
    pl_out_cols = _columns(remove_constant_columns(pl_df, verbose=0))
    assert pd_out_cols == pl_out_cols, f"Categorical single-real-many-null divergence:\n  pandas: {pd_out_cols}\n  polars: {pl_out_cols}"
