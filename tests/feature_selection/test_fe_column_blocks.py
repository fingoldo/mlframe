"""``fe_column_blocks`` hands out the numeric columns of a pandas or polars frame in budgeted C-contiguous float32 blocks, and the float32 view built on it equals the columns."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _fe_frame_ops as O

pl = pytest.importorskip("polars")


def _frame(n=1000):
    """A frame with float32, float64 and integer columns and a string one that must be left out."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.random(n).astype(np.float32), "b": rng.random(n), "s": ["x"] * n, "i": np.arange(n)})


@pytest.mark.parametrize("make", [lambda d: d, pl.from_pandas], ids=["pandas", "polars"])
def test_blocks_respect_the_budget_and_carry_the_columns(make):
    """Two columns of float32 fit the budget, so three numeric columns come as blocks of 2 and 1, each column equal to the float32 cast of the source."""
    df = _frame()
    blocks = list(O.fe_column_blocks(make(df), budget_bytes=2 * len(df) * 4))
    assert [b.shape for _, b in blocks] == [(2, len(df)), (1, len(df))]
    assert bool(blocks) and all(b.flags.c_contiguous and b.dtype == np.float32 for _, b in blocks)
    got = {name: row for names, b in blocks for name, row in zip(names, b)}
    assert sorted(got) == ["a", "b", "i"]
    for name, row in got.items():
        np.testing.assert_array_equal(row, df[name].to_numpy().astype(np.float32))


def test_float32_view_of_a_polars_frame_equals_the_columns():
    """The view keeps the numeric columns in float32 and the values unchanged."""
    df = _frame()
    view = O.fe_numeric_float32_view(pl.from_pandas(df))
    assert list(view.columns) == ["a", "b", "i"] and set(view.dtypes) == {np.dtype(np.float32)}
    np.testing.assert_array_equal(view["a"].to_numpy(), df["a"].to_numpy())
