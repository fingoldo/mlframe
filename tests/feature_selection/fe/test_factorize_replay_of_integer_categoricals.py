"""A pandas Categorical with integer categories replays through its code map, not through the category values."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.engineered_recipes._recipe_extract import _coerce_to_int_with_nan_handling, build_category_code_map


def test_integer_categories_replay_to_their_fit_codes():
    """Categories 10 / 20 / 30 were coded 0 / 1 / 2 at fit; the replay used to return the values clipped to the top bin, a constant feature."""
    ser = pd.Series(pd.Categorical([10, 20, 30, 20, 10]))
    cmap = build_category_code_map(ser)
    vals = ser.to_numpy()
    assert np.issubdtype(vals.dtype, np.integer)
    got = _coerce_to_int_with_nan_handling(vals, 3, "r", "c", "clip", cat_code_map=cmap)
    assert got.tolist() == [0, 1, 2, 1, 0]


def test_integer_categories_that_arrive_as_float_still_match():
    """With a NaN elsewhere the column is promoted to float (10.0): the lookup falls back to the integer spelling."""
    ser = pd.Series(pd.Categorical([10, 20, 30]))
    cmap = build_category_code_map(ser)
    got = _coerce_to_int_with_nan_handling(np.array([10.0, 20.0, 30.0]), 3, "r", "c", "clip", cat_code_map=cmap)
    assert got.tolist() == [0, 1, 2]


def test_plain_numeric_column_without_a_map_is_unchanged():
    """No map: the old integer cast path."""
    got = _coerce_to_int_with_nan_handling(np.array([0, 1, 2]), 3, "r", "c", "clip", cat_code_map=None)
    assert got.tolist() == [0, 1, 2]


def test_pandas_na_in_a_string_column_replays_to_the_nan_code():
    """A pandas string column holds pd.NA for a missing cell; fit gave it code 0 (NaN shift), and replay used to send it to the sentinel bin instead."""
    ser = pd.Series(["a", pd.NA, "b"], dtype="string")
    cmap = build_category_code_map(ser)
    got = _coerce_to_int_with_nan_handling(ser.to_numpy(dtype=object), 3, "r", "c", "clip", cat_code_map=cmap)
    assert got.tolist() == [1, 0, 2]
