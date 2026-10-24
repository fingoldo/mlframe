"""The large-polars path of the binned numeric aggregate family reads a float32 copy of the numeric columns.

The copy must hold only numeric columns, in float32, with values equal to the source up to float32 precision.
"""

import numpy as np
import pandas as pd
import polars as pl

from mlframe.feature_selection.filters._fe_frame_ops import fe_numeric_float32_view


def test_view_keeps_numeric_columns_only_in_float32():
    """A string column is dropped; every numeric column comes back as float32 with the same row count."""
    rng = np.random.default_rng(0)
    X = pl.DataFrame({"a": rng.standard_normal(1000), "k": rng.integers(0, 5, 1000), "s": [f"x{i % 3}" for i in range(1000)]})
    view = fe_numeric_float32_view(X)
    assert isinstance(view, pd.DataFrame)
    assert list(view.columns) == ["a", "k"]
    assert all(view[c].dtype == np.float32 for c in view.columns)
    assert len(view) == 1000


def test_view_values_match_the_source_to_float32_precision():
    """Each value equals the polars value rounded to float32."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal(500)
    view = fe_numeric_float32_view(pl.DataFrame({"a": a}))
    np.testing.assert_allclose(view["a"].to_numpy(), a.astype(np.float32), rtol=0, atol=0)
