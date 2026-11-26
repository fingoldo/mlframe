"""Cluster 24 regression: the pandas fallback of nearest_past_join keeps user columns named "index" and a named index (audit 2026-11-20, finding 24-F58).

The fallback used reset_index(), which creates a column literally named "index". With a user column of that name, or a named left index, the
merge or the set_index step picked the wrong column. The test forces the ImportError branch by hiding polars and checks that the result matches
the polars path row for row and keeps the caller's index.
"""

import sys

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_engineering.nearest_past_join import nearest_past_join


def _frames():
    """Left frame with a user column named ``index`` and a named index; right frame with one past value per entity."""
    left = pd.DataFrame(
        {"index": [10, 20, 30, 40], "t": [5.0, 1.0, 9.0, 3.0], "ent": ["a", "a", "b", "b"]},
        index=pd.Index([7, 3, 9, 1], name="rowkey"),
    )
    right = pd.DataFrame({"t": [0.0, 4.0, 2.0, 8.0], "ent": ["a", "a", "b", "b"], "val": [1.0, 2.0, 3.0, 4.0]})
    return left, right


def test_fallback_matches_polars_and_keeps_index_when_column_named_index(monkeypatch):
    """Hiding polars forces the pandas fallback; its output equals the polars path, with the caller's index and the user's "index" column (24-F58)."""
    pytest.importorskip("polars")
    left, right = _frames()
    expected = nearest_past_join(left, right, on="t", by=["ent"], right_value_cols=["val"])

    monkeypatch.setitem(sys.modules, "polars", None)
    got = nearest_past_join(left, right, on="t", by=["ent"], right_value_cols=["val"])

    assert list(got.index) == list(left.index)
    assert got.index.name == "rowkey"
    assert "index" in got.columns and "__nearest_past_row__" not in got.columns
    np.testing.assert_array_equal(got["index"].to_numpy(), left["index"].to_numpy())
    np.testing.assert_allclose(got["val"].to_numpy(dtype=float), expected["val"].to_numpy(dtype=float), equal_nan=True)
