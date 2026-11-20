"""Pre-screen null counting on pandas extension dtypes, and polars float-NaN handling in the variance rule."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.pre_screen import compute_unsupervised_drops


def test_mostly_null_nullable_columns_are_dropped_by_the_null_fraction_rule():
    """Float64 / Int64 columns that are 99.5% null are counted as null (np.issubdtype raised on them and the count fell back to 0)."""
    n = 1000
    vals = np.full(n, np.nan)
    vals[:5] = [1.0, 2.0, 3.0, 4.0, 5.0]
    df = pd.DataFrame(
        {
            "f64": pd.array(vals, dtype="Float64"),
            "i64": pd.array([1, 2, 3, 4, 5] + [None] * (n - 5), dtype="Int64"),
            "ok": np.random.default_rng(0).normal(size=n),
        }
    )
    assert compute_unsupervised_drops(df) == ["f64", "i64"]


def test_polars_float_column_with_a_few_nans_is_kept():
    """Polars var() propagates NaN; a varying column with a few NaNs must be kept, as in the pandas branch."""
    pl = pytest.importorskip("polars")
    vals = list(np.random.default_rng(1).normal(size=200))
    vals[3] = float("nan")
    pdf = pd.DataFrame({"x": vals})
    assert compute_unsupervised_drops(pdf) == []
    assert compute_unsupervised_drops(pl.DataFrame({"x": vals})) == []


def test_polars_constant_float_column_with_nans_is_still_dropped():
    """A constant column with NaN cells is constant over its observed values, in both backends."""
    pl = pytest.importorskip("polars")
    vals = [2.0] * 50 + [float("nan")] * 3
    assert compute_unsupervised_drops(pd.DataFrame({"x": vals})) == ["x"]
    assert compute_unsupervised_drops(pl.DataFrame({"x": vals})) == ["x"]
