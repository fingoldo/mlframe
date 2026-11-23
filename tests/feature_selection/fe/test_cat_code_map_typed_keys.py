"""A categorical column that holds both the integer 1 and the string "1" keeps two codes in the replay map (AUD23-07).

The plain ``str(value)`` key merged them (one entry, the code of the later one), so a transform gave the integer 1 the code of the string "1" while fit had given them different codes. The map now uses typed keys
(``int:1``, ``str:1``) with a flag, only for a column where the plain keys would collide; every other column keeps the plain keys, so existing recipes replay unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.engineered_recipes._recipe_extract import _TYPED_KEYS_FLAG, _codes_from_cat_map, build_category_code_map


def _replay(values: list, cat_map: dict) -> list:
    """Codes of ``values`` through the stored map, the way a transform does."""
    return _codes_from_cat_map(np.array(values, dtype=object), cat_map, n_bins=10, recipe_name="r", col_name="c", unknown_strategy="raise").tolist()


def test_integer_one_and_string_one_keep_two_codes():
    """Fit codes of ``[1, "1", 2]`` are 0, 1, 2; the replay gives the same, not 1, 1, 2."""
    train = pd.Series([1, "1", 2, 1, "1"], dtype=object)
    fit_codes = pd.factorize(train)[0].tolist()
    cat_map = build_category_code_map(train)
    assert _TYPED_KEYS_FLAG in cat_map
    assert _replay([1, "1", 2, 1, "1"], cat_map) == fit_codes


def test_a_float_that_holds_an_integer_still_finds_the_integer_category():
    """A column promoted to float by a NaN elsewhere delivers 1.0 for the category 1: the typed lookup falls back to the integer key."""
    cat_map = build_category_code_map(pd.Series([1, "1", 2], dtype=object))
    assert _replay([1.0], cat_map) == [0]


def test_a_column_without_a_collision_keeps_plain_keys():
    """Contrast: no collision, no flag, the keys are the old ``str(value)`` ones, so recipes stored before this change replay the same."""
    cat_map = build_category_code_map(pd.Series(["a", "b", "a"], dtype=object))
    assert _TYPED_KEYS_FLAG not in cat_map
    assert cat_map == {"a": 0, "b": 1}
    assert _replay(["b", "a"], cat_map) == [1, 0]


def test_nan_shift_is_kept_with_typed_keys():
    """The ``+1`` shift of a NaN-bearing column applies to typed keys and the NaN cell maps to 0."""
    train = pd.Series([1, "1", np.nan, 2], dtype=object)
    cat_map = build_category_code_map(train)
    assert _TYPED_KEYS_FLAG in cat_map
    assert _replay([np.nan, 1, "1", 2], cat_map) == [0, 1, 2, 3]
