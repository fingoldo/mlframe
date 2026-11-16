"""The tiled block transposes of ``to_feature_matrix`` / ``from_feature_matrix`` round-trip ragged shapes (rows not a multiple of the tile, columns not a multiple of the block) with nulls and categoricals."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.feature_selection.filters._fe_matrix_io import to_feature_matrix, from_feature_matrix


@pytest.mark.parametrize("n,p", [(1, 1), (7, 3), (1025, 17), (3000, 33), (2049, 16)])
def test_pandas_roundtrip_with_nulls_and_categoricals(n, p):
    """Numeric columns, a categorical and NaNs survive the round trip, in the original column order."""
    rng = np.random.default_rng(n + p)
    data = {f"f{i}": rng.standard_normal(n).astype(np.float32) for i in range(p)}
    if n > 5:
        data["f0"][::5] = np.nan
    df = pd.DataFrame(data)
    df["cat"] = pd.Categorical(rng.choice(["a", "b", "c"], n))
    fm = to_feature_matrix(df)
    assert fm.numeric.shape == (n, p) and fm.numeric.flags.c_contiguous
    for i in range(p):
        np.testing.assert_array_equal(fm.numeric[:, i], df[f"f{i}"].to_numpy(np.float32), strict=False) if not np.isnan(df[f"f{i}"]).any() else None
    back = from_feature_matrix(fm)
    assert list(back.columns) == list(df.columns)
    for i in range(p):
        np.testing.assert_array_equal(back[f"f{i}"].to_numpy(np.float32), df[f"f{i}"].to_numpy(np.float32))
    assert list(back["cat"].astype(str)) == list(df["cat"].astype(str))


def test_polars_numeric_roundtrip():
    """A polars frame of 40 float columns over 2500 rows comes back equal."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2500, 40)).astype(np.float32)
    df = pl.DataFrame({f"f{i}": a[:, i] for i in range(40)})
    back = from_feature_matrix(to_feature_matrix(df))
    np.testing.assert_array_equal(back.to_numpy().astype(np.float32), a)
