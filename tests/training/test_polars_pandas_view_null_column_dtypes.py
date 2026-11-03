"""The documented dtype outcome of the polars-to-pandas view for columns that hold nulls."""

from __future__ import annotations

import numpy as np
import polars as pl

from mlframe.training.utils import get_pandas_view_of_polars_df


def test_dtype_survives_only_without_nulls():
    """Integers without nulls stay int64; integers with nulls turn float64 with NaN; booleans with nulls turn nullable Int8."""
    df = pl.DataFrame({"a": [1, None, 3], "b": [1, 2, 3], "c": [True, None, False]})
    out = get_pandas_view_of_polars_df(df)
    assert out["b"].dtype == np.int64
    assert out["a"].dtype == np.float64 and np.isnan(out["a"].iloc[1])
    assert str(out["c"].dtype) == "Int8"
