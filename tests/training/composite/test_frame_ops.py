"""``append_column`` adds a column without duplicating the frame it was given."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite._frame_ops import append_column


def _frame(n: int = 200) -> pd.DataFrame:
    """A small float frame."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})


def test_writing_to_the_result_never_reaches_the_callers_frame():
    """The result is isolated from the source. Sharing the source's buffers was tried and dropped: on pandas 2.x without
    copy-on-write a frame that shares them lets a write to the result land in the caller's frame, so ``append_column``
    pays pandas' block copy instead (pandas 3, where CoW is always on, shares lazily and stays isolated)."""
    df = _frame()
    before = df.copy()
    out = append_column(df, "p", np.ones(len(df)))
    assert list(out.columns) == ["a", "b", "p"]
    out.iloc[0, 0] = -999
    out.loc[:, "b"] = 0
    pd.testing.assert_frame_equal(df, before)


def test_the_source_frame_is_left_alone():
    """The caller's frame gains neither the column nor a changed one."""
    df = _frame()
    append_column(df, "p", np.ones(len(df)))
    assert list(df.columns) == ["a", "b"]


def test_an_existing_column_is_replaced_in_the_result_only():
    """Appending over a name that is already there overwrites it in the new frame, not in the source."""
    df = _frame()
    df["p"] = 0.0
    out = append_column(df, "p", np.full(len(df), 5.0))
    np.testing.assert_allclose(out["p"].to_numpy(), 5.0)
    np.testing.assert_allclose(df["p"].to_numpy(), 0.0)
    assert list(out.columns) == ["a", "b", "p"]


def test_a_numpy_input_gets_the_values_as_its_last_column():
    """An array input keeps working: the values become the trailing column."""
    out = append_column(np.zeros((4, 2)), "p", np.arange(4))
    assert out.shape == (4, 3)
    np.testing.assert_allclose(out[:, -1], np.arange(4))


def test_a_polars_frame_goes_through_with_columns():
    """The polars path returns a polars frame with the column attached."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    out = append_column(df, "p", np.array([4.0, 5.0, 6.0]))
    assert isinstance(out, pl.DataFrame) and out.columns == ["a", "p"]
    assert df.columns == ["a"]
