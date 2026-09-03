"""Tests for ``mlframe.feature_selection.filters._grouped_coerce_shared`` -- shared recipe-replay coercion
helpers for the grouped-aggregate FE family (_grouped_agg_fe, _grouped_quantile_fe, _ratio_delta_fe),
consolidated specifically to prevent independently-duplicated copies from drifting apart. Previously had
zero test coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._grouped_coerce_shared import (
    auto_detect_num_cols_plain,
    auto_detect_num_cols_skip_grp,
    broadcast_lookup,
    coerce_X_for_grouped,
)


class TestBroadcastLookup:
    """Groups tests covering broadcast_lookup's group-key -> value mapping."""

    def test_maps_each_key_via_the_lookup_dict(self):
        """Maps each key via the lookup dict."""
        g_keys = np.array(["a", "b", "a", "c"])
        lookup = {"a": 1.0, "b": 2.0, "c": 3.0}
        out = broadcast_lookup(g_keys, lookup, glob=-1.0)
        np.testing.assert_array_equal(out, [1.0, 2.0, 1.0, 3.0])

    def test_unseen_key_falls_back_to_global(self):
        """Unseen key falls back to global."""
        g_keys = np.array(["a", "z"])
        lookup = {"a": 1.0}
        out = broadcast_lookup(g_keys, lookup, glob=99.0)
        np.testing.assert_array_equal(out, [1.0, 99.0])

    def test_matches_per_row_reference_mapping(self):
        """Matches per row reference mapping."""
        rng = np.random.default_rng(0)
        g_keys = rng.choice(["a", "b", "c", "d"], size=500)
        lookup = {"a": 1.0, "b": 2.0, "c": 3.0}  # "d" deliberately unseen
        got = broadcast_lookup(g_keys, lookup, glob=-5.0)
        expected = np.array([lookup.get(str(k), -5.0) for k in g_keys], dtype=np.float64)
        np.testing.assert_array_equal(got, expected)

    def test_integer_group_keys(self):
        """broadcast_lookup must handle integer (not just string) group keys via str() coercion."""
        g_keys = np.array([1, 2, 1, 3])
        lookup = {"1": 10.0, "2": 20.0}
        out = broadcast_lookup(g_keys, lookup, glob=0.0)
        np.testing.assert_array_equal(out, [10.0, 20.0, 10.0, 0.0])

    def test_nan_and_inf_lookup_values_are_replaced_with_global(self):
        """A lookup value that happens to be NaN/inf must be replaced with the global fallback, not
        propagated -- the function's own nan_to_num call at the end."""
        g_keys = np.array(["a", "b", "c"])
        lookup = {"a": float("nan"), "b": float("inf"), "c": float("-inf")}
        out = broadcast_lookup(g_keys, lookup, glob=7.0)
        np.testing.assert_array_equal(out, [7.0, 7.0, 7.0])

    def test_mixed_type_keys_fall_back_to_per_row_path(self):
        """Object-dtype keys that np.unique can't sort (mixed int/str) must still resolve correctly via
        the except-branch per-row fallback, not raise."""
        g_keys = np.array(["a", 1, "b", 1], dtype=object)
        lookup = {"a": 1.0, "1": 2.0, "b": 3.0}
        out = broadcast_lookup(g_keys, lookup, glob=-1.0)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0, 2.0])


class TestAutoDetectNumColsPlain:
    """Groups tests covering auto_detect_num_cols_plain's column-selection rules."""

    def test_excludes_group_cols(self):
        """Excludes group cols."""
        df = pd.DataFrame({"g": [1, 2, 3], "a": [1.0, 2.0, 3.0]})
        out = auto_detect_num_cols_plain(df, group_cols=["g"])
        assert out == ["a"]

    def test_float_columns_always_qualify(self):
        """Float columns always qualify."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        out = auto_detect_num_cols_plain(df, group_cols=[])
        assert set(out) == {"a", "b"}

    def test_low_cardinality_int_column_excluded(self):
        """An integer column with <=500 uniques is treated as categorical, not a numeric candidate."""
        df = pd.DataFrame({"cat_like_int": [1, 2, 1, 2, 3] * 20})
        out = auto_detect_num_cols_plain(df, group_cols=[])
        assert out == []

    def test_high_cardinality_int_column_included(self):
        """An integer column with >500 uniques is treated as genuinely numeric."""
        df = pd.DataFrame({"high_card_int": list(range(600))})
        out = auto_detect_num_cols_plain(df, group_cols=[])
        assert out == ["high_card_int"]

    def test_non_numeric_columns_excluded(self):
        """String/categorical columns must never be selected."""
        df = pd.DataFrame({"s": ["x", "y", "z"], "a": [1.0, 2.0, 3.0]})
        out = auto_detect_num_cols_plain(df, group_cols=[])
        assert out == ["a"]

    def test_respects_max_cols_cap(self):
        """Respects max cols cap."""
        df = pd.DataFrame({f"c{i}": [1.0, 2.0, 3.0] for i in range(10)})
        out = auto_detect_num_cols_plain(df, group_cols=[], max_cols=3)
        assert len(out) == 3

    def test_no_grp_prefix_exclusion(self):
        """Unlike the skip_grp sibling, a 'grp'-prefixed float column MUST still be selected here."""
        df = pd.DataFrame({"grp_stat": [1.0, 2.0, 3.0]})
        out = auto_detect_num_cols_plain(df, group_cols=[])
        assert out == ["grp_stat"]


class TestAutoDetectNumColsSkipGrp:
    """Groups tests covering auto_detect_num_cols_skip_grp's additional grp-prefix exclusion."""

    def test_grp_prefixed_column_excluded(self):
        """A 'grp'-prefixed float column must be excluded (would build a nested, non-replayable recipe)."""
        df = pd.DataFrame({"grp_mean_x": [1.0, 2.0, 3.0], "a": [4.0, 5.0, 6.0]})
        out = auto_detect_num_cols_skip_grp(df, group_cols=[])
        assert out == ["a"]

    def test_non_grp_prefixed_column_still_included(self):
        """Non grp prefixed column still included."""
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        out = auto_detect_num_cols_skip_grp(df, group_cols=[])
        assert set(out) == {"a", "b"}

    def test_excludes_group_cols_too(self):
        """Excludes group cols too."""
        df = pd.DataFrame({"g": [1.0, 2.0], "a": [3.0, 4.0]})
        out = auto_detect_num_cols_skip_grp(df, group_cols=["g"])
        assert out == ["a"]


class TestCoerceXForGrouped:
    """Groups tests covering coerce_X_for_grouped's input-type dispatch."""

    def test_pandas_input_returned_unchanged(self):
        """A pandas DataFrame input must be returned as-is (identity), not copied."""
        df = pd.DataFrame({"g": [1, 2], "n": [1.0, 2.0]})
        out = coerce_X_for_grouped(df, "g", "n", "recipe_x")
        assert out is df

    def test_polars_input_extracts_narrow_frame(self):
        """A polars DataFrame input must be converted to a narrow 2-column pandas frame."""
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"g": [1, 2, 3], "n": [10.0, 20.0, 30.0], "other": [0, 0, 0]})
        out = coerce_X_for_grouped(df, "g", "n", "recipe_x")
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["g", "n"]
        np.testing.assert_array_equal(out["g"].to_numpy(), [1, 2, 3])
        np.testing.assert_array_equal(out["n"].to_numpy(), [10.0, 20.0, 30.0])

    def test_structured_ndarray_input_extracts_narrow_frame(self):
        """A structured numpy array input (dtype.names is not None) must extract the two named fields."""
        arr = np.array([(1, 10.0), (2, 20.0)], dtype=[("g", "i4"), ("n", "f8")])
        out = coerce_X_for_grouped(arr, "g", "n", "recipe_x")
        assert isinstance(out, pd.DataFrame)
        np.testing.assert_array_equal(out["g"].to_numpy(), [1, 2])
        np.testing.assert_array_equal(out["n"].to_numpy(), [10.0, 20.0])

    def test_unsupported_type_raises_type_error_naming_the_recipe(self):
        """An unrecognized input type must raise TypeError naming both the columns and the recipe --
        useful for debugging a broken transform-time call site."""
        with pytest.raises(TypeError, match="recipe_x"):
            coerce_X_for_grouped([1, 2, 3], "g", "n", "recipe_x")

    def test_plain_unstructured_ndarray_raises(self):
        """A plain (non-structured) ndarray has no named fields to extract from and must raise."""
        arr = np.zeros((3, 2))
        with pytest.raises(TypeError):
            coerce_X_for_grouped(arr, "g", "n", "recipe_y")
