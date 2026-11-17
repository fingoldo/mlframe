"""The Series-free column accessors of ``_fe_frame_ops`` answer exactly what ``X[c].to_numpy()`` and ``is_numeric_dtype(X[c])`` answer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._fe_frame_ops import fe_column_is_numeric, fe_column_values, fe_is_numeric_col


@pytest.fixture
def frame() -> pd.DataFrame:
    """One column per storage kind the FE families meet: float block, lone float64, bool, strings, categorical, nullable integer."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((50, 3)).astype(np.float32), columns=["f0", "f1", "f2"])
    df["d"] = rng.standard_normal(50)
    df["b"] = rng.random(50) > 0.5
    df["s"] = np.where(df["b"], "u", "v")
    df["c"] = pd.Categorical(df["s"])
    df["i"] = pd.array([None if k % 7 == 0 else k for k in range(50)], dtype="Int64")
    return df


def test_column_values_equal_series_to_numpy(frame: pd.DataFrame) -> None:
    """Every storage kind returns the dtype, shape, missing mask and values of the Series route."""
    for c in frame.columns:
        got, want = fe_column_values(frame, c), frame[c].to_numpy()
        assert got.dtype == want.dtype and got.shape == want.shape, c
        assert np.array_equal(pd.isna(got), pd.isna(want)), c
        keep = ~pd.isna(want)
        assert (got[keep] == want[keep]).all(), c


def test_numpy_backed_column_is_a_read_only_view_of_the_frame(frame: pd.DataFrame) -> None:
    """A numpy-backed column shares the frame's memory and cannot be written through, as under copy-on-write."""
    got = fe_column_values(frame, "f1")
    assert np.shares_memory(got, frame["f1"].to_numpy())
    assert not got.flags.writeable
    with pytest.raises(ValueError):
        got[0] = 1.0


def test_numeric_test_equals_pandas(frame: pd.DataFrame) -> None:
    """The dtype-only numeric test agrees with ``is_numeric_dtype`` on every column and is false for an absent one."""
    for c in frame.columns:
        assert fe_column_is_numeric(frame, c) == pd.api.types.is_numeric_dtype(frame[c]), c
    assert not fe_column_is_numeric(frame, "absent")


def test_duplicated_name_takes_the_series_route(frame: pd.DataFrame) -> None:
    """A name carried by two columns is not numeric (ambiguous) and its values keep the two-column shape."""
    dup = pd.concat([frame[["f0"]], frame[["f0"]]], axis=1)
    assert not fe_is_numeric_col(dup, "f0")  # ambiguous: two columns answer to the name
    assert fe_column_values(dup, "f0").shape == (50, 2)
