"""
Tests for mlframe training utilities.

Tests cover save/load model functions, DataFrame conversions,
and special value processing functions.
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
import tempfile
import os
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from mlframe.training.utils import (
    get_pandas_view_of_polars_df,
    _NESTED_DTYPE_WARN_SEEN,
    drop_columns_from_dataframe,
    save_series_or_df,
    process_nans,
    process_nulls,
    process_infinities,
    remove_constant_columns,
)
from mlframe.training.io import (
    save_mlframe_model,
    load_mlframe_model,
)

# ================================================================================================
# Save/Load Model Tests
# ================================================================================================


class TestSaveLoadModel:
    """Tests for save_mlframe_model and load_mlframe_model."""

    def test_save_and_load_simple_dict(self, tmp_path):
        """Test saving and loading a simple dictionary."""
        model = {"weights": [1, 2, 3], "bias": 0.5}
        file_path = str(tmp_path / "model.zst")

        result = save_mlframe_model(model, file_path, verbose=0)
        assert result is True
        assert os.path.exists(file_path)

        loaded = load_mlframe_model(file_path)
        assert loaded == model

    def test_save_and_load_numpy_array(self, tmp_path):
        """Test saving and loading numpy arrays."""
        model = {"array": np.random.randn(100, 50)}
        file_path = str(tmp_path / "numpy_model.zst")

        save_mlframe_model(model, file_path, verbose=0)
        loaded = load_mlframe_model(file_path)

        np.testing.assert_array_equal(loaded["array"], model["array"])

    def test_save_and_load_sklearn_model(self, tmp_path):
        """Test saving and loading sklearn model."""
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        model.fit(X, y)

        file_path = str(tmp_path / "sklearn_model.zst")
        save_mlframe_model(model, file_path, verbose=0)
        loaded = load_mlframe_model(file_path)

        # Check predictions match
        np.testing.assert_array_almost_equal(model.predict(X), loaded.predict(X))

    def test_save_with_custom_compression(self, tmp_path):
        """Test saving with custom zstd compression settings."""
        model = {"data": list(range(1000))}
        file_path = str(tmp_path / "compressed.zst")

        # High compression level
        zstd_kwargs = {"level": 19, "threads": 1}
        result = save_mlframe_model(model, file_path, zstd_kwargs=zstd_kwargs, verbose=0)

        assert result is True
        loaded = load_mlframe_model(file_path)
        assert loaded == model

    def test_save_returns_false_on_invalid_path(self, tmp_path):
        """Test that save returns False for invalid paths."""
        model = {"test": 1}
        invalid_path = str(tmp_path / "nonexistent" / "dir" / "model.zst")

        result = save_mlframe_model(model, invalid_path, verbose=0)
        assert result is False

    def test_load_returns_none_for_missing_file(self):
        """Test that load returns None for missing file."""
        result = load_mlframe_model("/nonexistent/path/model.zst")
        assert result is None

    def test_load_returns_none_for_corrupted_file(self, tmp_path):
        """Test that load returns None for corrupted file."""
        file_path = str(tmp_path / "corrupted.zst")
        with open(file_path, "wb") as f:
            f.write(b"not a valid zstd file")

        result = load_mlframe_model(file_path)
        assert result is None

    def test_save_logs_file_size(self, tmp_path, caplog):
        """Test that save logs file size when verbose."""
        import logging

        caplog.set_level(logging.INFO)

        model = {"data": list(range(100))}
        file_path = str(tmp_path / "model.zst")

        save_mlframe_model(model, file_path, verbose=1)

        assert "Model saved successfully" in caplog.text
        assert "Size:" in caplog.text
        assert "Mb" in caplog.text

    def test_roundtrip_complex_nested_object(self, tmp_path):
        """Test saving and loading complex nested objects."""
        model = {
            "config": {"nested": {"deep": {"value": 42}}},
            "arrays": [np.array([1, 2, 3]), np.array([4, 5, 6])],
            "metadata": pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}),
        }
        file_path = str(tmp_path / "complex.zst")

        save_mlframe_model(model, file_path, verbose=0)
        loaded = load_mlframe_model(file_path)

        assert loaded["config"]["nested"]["deep"]["value"] == 42
        np.testing.assert_array_equal(loaded["arrays"][0], model["arrays"][0])
        pd.testing.assert_frame_equal(loaded["metadata"], model["metadata"])


# ================================================================================================
# Pandas View Tests
# ================================================================================================


class TestGetPandasViewOfPolarsDF:
    """Tests for get_pandas_view_of_polars_df."""

    def test_basic_numeric_conversion(self):
        """Test conversion of numeric columns."""
        pl_df = pl.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.0, 2.0, 3.0],
            }
        )

        pd_df = get_pandas_view_of_polars_df(pl_df)

        assert isinstance(pd_df, pd.DataFrame)
        assert list(pd_df.columns) == ["int_col", "float_col"]
        assert len(pd_df) == 3

    def test_string_columns(self):
        """Test conversion of string columns."""
        pl_df = pl.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
            }
        )

        pd_df = get_pandas_view_of_polars_df(pl_df)

        assert pd_df["name"].tolist() == ["Alice", "Bob", "Charlie"]

    def test_categorical_preserved_as_pd_categorical(self):
        """Test that Polars Categorical columns become pd.Categorical (int codes
        + categories dict), not strings. Benchmarked 2026-04-17: string cast
        adds ~37% to CatBoost fit+predict and OOMs at 450k+ rows.

        Category-set equality (not order) is the contract -- the Arrow
        split-blocks bridge may surface categories in encounter-order
        on some polars / pyarrow combos and sorted-order on others; the
        downstream CatBoost / LightGBM consumers don't care about the
        dictionary order, only that the dtype is preserved.
        """
        pl_df = pl.DataFrame(
            {
                "category": pl.Series(["A", "B", "A", "C"]).cast(pl.Categorical),
            }
        )

        pd_df = get_pandas_view_of_polars_df(pl_df)

        assert isinstance(pd_df["category"].dtype, pd.CategoricalDtype)
        assert pd_df["category"].tolist() == ["A", "B", "A", "C"]
        assert set(pd_df["category"].cat.categories) == {"A", "B", "C"}

    def test_boolean_columns(self):
        """Test conversion of boolean columns."""
        pl_df = pl.DataFrame(
            {
                "bool_col": [True, False, True],
            }
        )

        pd_df = get_pandas_view_of_polars_df(pl_df)

        assert pd_df["bool_col"].tolist() == [True, False, True]

    def test_mixed_column_types(self):
        """Test conversion with mixed column types."""
        pl_df = pl.DataFrame(
            {
                "int": [1, 2, 3],
                "float": [1.1, 2.2, 3.3],
                "str": ["a", "b", "c"],
                "bool": [True, False, True],
            }
        )

        pd_df = get_pandas_view_of_polars_df(pl_df)

        assert len(pd_df.columns) == 4
        assert len(pd_df) == 3

    def test_assertion_error_on_non_polars(self):
        """Test that assertion error is raised for non-Polars input."""
        pd_df = pd.DataFrame({"col": [1, 2, 3]})

        with pytest.raises(TypeError):
            get_pandas_view_of_polars_df(pd_df)

    def test_polars_series_input(self):
        """Test conversion of Polars Series - currently not supported."""
        pl_series = pl.Series("values", [1, 2, 3])

        # Series passes assertion but fails internally - convert to DataFrame first
        with pytest.raises(AttributeError):
            get_pandas_view_of_polars_df(pl_series)

    def test_empty_dataframe(self):
        """Test conversion of empty DataFrame."""
        pl_df = pl.DataFrame({"col": []}).cast({"col": pl.Int64})

        pd_df = get_pandas_view_of_polars_df(pl_df)

        assert len(pd_df) == 0
        assert "col" in pd_df.columns

    def test_preserves_column_order(self):
        """Test that column order is preserved."""
        pl_df = pl.DataFrame(
            {
                "z": [1],
                "a": [2],
                "m": [3],
            }
        )

        pd_df = get_pandas_view_of_polars_df(pl_df)

        assert list(pd_df.columns) == ["z", "a", "m"]

    # --- Categorical edge-case coverage (2026-04-17 dict→pd.Categorical change) ---
    # These guard the optimization: polars emits dict with uint32 indices, we
    # rebuild with int32. Properties to verify differ from native pandas
    # categoricals in subtle ways.

    def test_categorical_codes_are_integer(self):
        """Polars uses uint32 dictionary indices; pyarrow refuses them in
        to_pandas. We rebuild as int32 so conversion succeeds — but pandas
        then downcasts the codes to int8/int16 based on cardinality. The
        contract is only that conversion SUCCEEDS and yields a valid
        pd.Categorical with integer codes, not a specific codes dtype."""
        pl_df = pl.DataFrame(
            {
                "c": pl.Series(["a", "b", "c"]).cast(pl.Categorical),
            }
        )
        pd_df = get_pandas_view_of_polars_df(pl_df)
        assert pd.api.types.is_integer_dtype(pd_df["c"].cat.codes.dtype)
        assert pd_df["c"].cat.codes.tolist() == [0, 1, 2]

    def test_categorical_with_nulls_becomes_nan(self):
        """A polars Categorical containing nulls round-trips into a pandas
        Categorical where nulls are represented by code == -1 (NaN)."""
        pl_df = pl.DataFrame(
            {
                "c": pl.Series(["a", None, "b", None, "a"], dtype=pl.Categorical),
            }
        )
        pd_df = get_pandas_view_of_polars_df(pl_df)
        assert isinstance(pd_df["c"].dtype, pd.CategoricalDtype)
        codes = pd_df["c"].cat.codes.tolist()
        assert codes.count(-1) == 2  # two nulls
        # Non-null codes point at the two distinct categories
        assert set(c for c in codes if c != -1) <= {0, 1}
        assert sorted(pd_df["c"].cat.categories.tolist()) == ["a", "b"]

    def test_categorical_high_cardinality(self):
        """300 distinct values (above the 255 int8 ceiling, below the int16
        ceiling). The rebuilt Categorical must still be valid and contain
        all 300 categories; pandas picks whatever codes dtype fits."""
        cats = [f"cat_{i % 300}" for i in range(1000)]
        pl_df = pl.DataFrame({"c": pl.Series(cats, dtype=pl.Categorical)})
        pd_df = get_pandas_view_of_polars_df(pl_df)
        assert isinstance(pd_df["c"].dtype, pd.CategoricalDtype)
        assert pd.api.types.is_integer_dtype(pd_df["c"].cat.codes.dtype)
        assert len(pd_df["c"].cat.categories) == 300

    def test_categorical_single_category(self):
        """Degenerate: one distinct value repeated. Codes should all be 0."""
        pl_df = pl.DataFrame(
            {
                "c": pl.Series(["only"] * 10, dtype=pl.Categorical),
            }
        )
        pd_df = get_pandas_view_of_polars_df(pl_df)
        assert isinstance(pd_df["c"].dtype, pd.CategoricalDtype)
        assert pd_df["c"].cat.codes.tolist() == [0] * 10
        assert list(pd_df["c"].cat.categories) == ["only"]

    def test_categorical_all_null(self):
        """All values are null. Pandas Categorical with 0 categories, all -1 codes."""
        pl_df = pl.DataFrame(
            {
                "c": pl.Series([None, None, None], dtype=pl.Categorical),
            }
        )
        pd_df = get_pandas_view_of_polars_df(pl_df)
        assert isinstance(pd_df["c"].dtype, pd.CategoricalDtype)
        assert pd_df["c"].cat.codes.tolist() == [-1, -1, -1]
        assert pd_df["c"].isna().all()

    def test_categorical_enum_treated_as_categorical(self):
        """Polars Enum is also emitted as a pyarrow dictionary and must
        round-trip into pd.Categorical (same path as pl.Categorical)."""
        enum_dtype = pl.Enum(["low", "mid", "high"])
        pl_df = pl.DataFrame(
            {
                "level": pl.Series(["low", "high", "mid", "low"], dtype=enum_dtype),
            }
        )
        pd_df = get_pandas_view_of_polars_df(pl_df)
        assert isinstance(pd_df["level"].dtype, pd.CategoricalDtype)
        assert pd_df["level"].tolist() == ["low", "high", "mid", "low"]
        # Enum preserves the declared category order (different from lexicographic)
        assert list(pd_df["level"].cat.categories) == ["low", "mid", "high"]

    def test_categorical_equality_comparison(self):
        """Downstream filters frequently do df['c'] == 'A' — verify this still
        works on the rebuilt Categorical, since that is the API we expose to
        sklearn/CatBoost/LGB consumers."""
        pl_df = pl.DataFrame(
            {
                "c": pl.Series(["A", "B", "A", "C"], dtype=pl.Categorical),
            }
        )
        pd_df = get_pandas_view_of_polars_df(pl_df)
        mask = pd_df["c"] == "A"
        assert mask.tolist() == [True, False, True, False]

    def test_categorical_mixed_with_numeric(self):
        """Mixed frame: the numeric columns are untouched and the categorical
        is rebuilt. Column count, dtypes, and values all match expectations."""
        pl_df = pl.DataFrame(
            {
                "num": [1.0, 2.0, 3.0],
                "cat": pl.Series(["x", "y", "x"], dtype=pl.Categorical),
                "int": [10, 20, 30],
            }
        )
        pd_df = get_pandas_view_of_polars_df(pl_df)
        assert list(pd_df.columns) == ["num", "cat", "int"]
        assert pd.api.types.is_float_dtype(pd_df["num"])
        assert isinstance(pd_df["cat"].dtype, pd.CategoricalDtype)
        assert pd.api.types.is_integer_dtype(pd_df["int"])

    def test_categorical_tolist_matches_native_pandas(self):
        """Value equivalence: ``.tolist()`` on the polars-derived Categorical
        produces the same Python list that a native pd.Categorical built from
        the same data would produce. Guards against subtle encoding differences."""
        values = ["a", "b", "a", "c", "b", "a"]
        pl_df = pl.DataFrame({"c": pl.Series(values, dtype=pl.Categorical)})
        pd_df = get_pandas_view_of_polars_df(pl_df)
        native = pd.Categorical(values)
        assert pd_df["c"].tolist() == native.tolist()
        assert set(pd_df["c"].cat.categories) == set(native.categories)

    def test_categorical_astype_str_roundtrip(self):
        """Some consumers expect to coerce the column to plain strings.
        Categorical.astype(str) must give back the original values."""
        pl_df = pl.DataFrame(
            {
                "c": pl.Series(["foo", "bar", "baz"], dtype=pl.Categorical),
            }
        )
        pd_df = get_pandas_view_of_polars_df(pl_df)
        assert pd_df["c"].astype(str).tolist() == ["foo", "bar", "baz"]

    def test_categorical_empty_frame(self):
        """Empty Polars frame with a Categorical column → empty pd.Categorical,
        zero categories, zero rows. No IndexError, no crash on from_arrays."""
        pl_df = pl.DataFrame(
            {
                "c": pl.Series([], dtype=pl.Categorical),
            }
        )
        pd_df = get_pandas_view_of_polars_df(pl_df)
        assert len(pd_df) == 0
        assert isinstance(pd_df["c"].dtype, pd.CategoricalDtype)

    def test_categorical_batched_remap_three_plus_columns(self):
        """>= 3 Categorical columns take the batched single-collect remap path.

        Each column has a DISTINCT value universe, so a bug that mis-assigned
        one column's uniques to another (or dropped/duplicated a column in the
        batched ``select(...).implode()``) would surface as wrong categories or
        wrong values here. Guards the iter470 batching against the per-column
        loop's semantics."""
        pl_df = pl.DataFrame(
            {
                "ca": pl.Series(["a1", "a2", "a1", "a3"], dtype=pl.Categorical),
                "cb": pl.Series(["b9", "b8", "b8", "b7"], dtype=pl.Categorical),
                "cc": pl.Series(["c0", "c0", "c1", "c2"], dtype=pl.Categorical),
                "num": [1.0, 2.0, 3.0, 4.0],
            }
        )
        pd_df = get_pandas_view_of_polars_df(pl_df)
        # Values round-trip exactly, per column.
        assert pd_df["ca"].astype(str).tolist() == ["a1", "a2", "a1", "a3"]
        assert pd_df["cb"].astype(str).tolist() == ["b9", "b8", "b8", "b7"]
        assert pd_df["cc"].astype(str).tolist() == ["c0", "c0", "c1", "c2"]
        # Each column's Enum domain is exactly its own unique set (no cross-column leakage).
        assert set(pd_df["ca"].cat.categories) == {"a1", "a2", "a3"}
        assert set(pd_df["cb"].cat.categories) == {"b7", "b8", "b9"}
        assert set(pd_df["cc"].cat.categories) == {"c0", "c1", "c2"}
        assert pd.api.types.is_float_dtype(pd_df["num"])

    def test_categorical_batched_remap_matches_per_column_path(self):
        """The batched (>=3 cols) and per-column (<3 cols) remap paths must
        produce identical pandas output for the same column. Build a 3-col frame
        (batched) and a 1-col frame (per-column) sharing column ``c`` and assert
        the rebuilt Categorical is bit-identical (codes + categories)."""
        vals = ["p", "q", "p", "r", "q", "p", "s"]
        batched = get_pandas_view_of_polars_df(
            pl.DataFrame(
                {
                    "c": pl.Series(vals, dtype=pl.Categorical),
                    "d": pl.Series(["x"] * 7, dtype=pl.Categorical),
                    "e": pl.Series(["y"] * 7, dtype=pl.Categorical),
                }
            )
        )
        per_col = get_pandas_view_of_polars_df(
            pl.DataFrame(
                {
                    "c": pl.Series(vals, dtype=pl.Categorical),
                }
            )
        )
        assert batched["c"].tolist() == per_col["c"].tolist()
        assert list(batched["c"].cat.categories) == list(per_col["c"].cat.categories)

    def test_nested_dtype_detected_via_isinstance(self, caplog):
        """The nested-dtype WARN fires for pl.List / pl.Struct columns (detected
        by isinstance, not str(dt)), while wide Categorical/Enum columns do NOT
        trip it -- the isinstance refactor must keep the same detection set."""
        import logging

        _NESTED_DTYPE_WARN_SEEN.clear()
        # Wide Enum (large repr) must NOT be flagged as nested.
        wide = pl.Enum([f"cat_{i}" for i in range(200)])
        df_clean = pl.DataFrame(
            {
                "level": pl.Series(["cat_1", "cat_2"], dtype=wide),
                "num": [1.0, 2.0],
            }
        )
        with caplog.at_level(logging.WARNING, logger="mlframe.training.utils"):
            get_pandas_view_of_polars_df(df_clean)
        assert "nested" not in caplog.text.lower()

        caplog.clear()
        _NESTED_DTYPE_WARN_SEEN.clear()
        # A List column IS nested and must trigger exactly one WARN.
        df_nested = pl.DataFrame(
            {
                "emb": pl.Series([[1.0, 2.0], [3.0, 4.0]], dtype=pl.List(pl.Float64)),
                "num": [1.0, 2.0],
            }
        )
        with caplog.at_level(logging.WARNING, logger="mlframe.training.utils"):
            get_pandas_view_of_polars_df(df_nested)
        assert "nested" in caplog.text.lower()
        assert "emb" in caplog.text


# ================================================================================================
# Polars slice categorical-dictionary behaviour (2026-04-19 regression sensor)
# ================================================================================================


class TestPolarsSliceDictionaryDiffers:
    """Documents Polars slice-over-Categorical semantics across versions and
    pins the only contract ``get_pandas_view_of_polars_df`` actually depends
    on: each call rebuilds the pandas Categorical from THAT frame's own Arrow
    chunk, so a sliced frame's pandas view is correct regardless of whether
    polars trims the dictionary on slice.

    History: the shared-dict optimisation attempted on 2026-04-19 assumed a
    sliced ``pl.Categorical`` keeps the parent's full dictionary; it was
    abandoned because polars (≤ 1.32, and again ≥ 1.4x — verified on polars
    1.41.2) trims each slice's dictionary to exactly the values present in
    that slice, so a shared Arrow-level cache is unsound. polars 1.33.x briefly
    preserved the parent dictionary on slice; that is version-specific and NOT
    something the converter relies on. The shared-dict codepath stays gated off
    (the ``shared_dict_cache`` parameter was never restored — see the
    2026-04-19 notes in ``get_pandas_view_of_polars_df``).
    """

    def test_high_cardinality_conversion_perf_budget(self):
        """Perf-budget regression sensor on a pl.Categorical column with
        500k unique values over 500k rows — exactly one value per row,
        stressing the dict-rebuild path. On a 2026-era dev box this
        completes in ~0.3 s; a 5 s budget is generous enough that a
        naive ``astype(str)`` regression would blow through it by 10×.
        """
        import time

        pool = np.array([f"s_{i:06d}" for i in range(500_000)])
        df = pl.DataFrame({"x": pl.Series("x", pool, dtype=pl.Categorical)})

        t0 = time.perf_counter()
        out = get_pandas_view_of_polars_df(df)
        elapsed = time.perf_counter() - t0

        assert out.shape == (500_000, 1)
        assert elapsed < 5.0, f"polars→pandas on 500k × 1 Categorical with 500k uniques took {elapsed:.1f}s — dict-rebuild path likely regressed"

    def test_empty_polars_dataframe(self):
        """Edge case: empty DF with typed column must round-trip without error."""
        df = pl.DataFrame({"a": pl.Series("a", [], dtype=pl.Int32)})
        out = get_pandas_view_of_polars_df(df)
        assert out.shape == (0, 1)

    def test_zero_column_polars_dataframe(self):
        """Edge case: 0-column DF (very unusual) must not crash."""
        df = pl.DataFrame({})
        out = get_pandas_view_of_polars_df(df)
        assert out.shape == (0, 0)

    def test_slice_categorical_dictionary_is_not_shared_across_slices(self):
        """Regression sensor for the shared-dict gate in
        ``get_pandas_view_of_polars_df``.

        The converter MUST NOT share a single Categorical ``categories`` array
        across sliced train/val/test calls, because the installed polars
        (1.41.2, verified) trims each slice's Arrow dictionary to exactly the
        values present in that slice — so the parent and its slices carry
        *different* dictionaries. The sensor:

          1. Confirms the trim behaviour is still in force (slice dict is a
             strict subset of the parent dict). If a future polars upgrade
             preserves the full parent dict on slice again (as 1.33.x did),
             this branch flips to the equal-length contract instead of red-
             failing — either way the converter's per-slice rebuild is correct.
          2. Proves the converter rebuilds each frame's pandas Categorical from
             that frame's OWN dictionary: the pandas view of a slice round-trips
             that slice's values exactly, independent of the parent.

        Renamed from ``test_slice_preserves_parent_categorical_dictionary``
        (the 2026-04-23 polars-1.33 contract): the shared-dict caching it
        guarded was never re-enabled, so the binding contract is the per-slice
        rebuild, not the polars dict-sharing detail.
        """
        rng = np.random.default_rng(0)
        pool = np.array([f"c_{i}" for i in range(200)])
        values = pool[rng.integers(0, 200, size=1000)]
        src = pl.DataFrame({"x": pl.Series("x", values).cast(pl.Categorical)})

        head = src.head(800)
        tail = src[800:]

        full_dict = src.to_arrow().column(0).chunks[0].dictionary
        head_dict = head.to_arrow().column(0).chunks[0].dictionary
        tail_dict = tail.to_arrow().column(0).chunks[0].dictionary

        # Document whichever slice-dict behaviour the installed polars has.
        # The shared-dict cache is gated off regardless, so both are fine; the
        # converter never relies on dict sharing.
        full_vals = set(full_dict.to_pylist())
        head_vals = set(head_dict.to_pylist())
        tail_vals = set(tail_dict.to_pylist())
        if len(head_dict) != len(full_dict):
            # Trim behaviour (polars ≤ 1.32 and ≥ 1.4x): each slice carries a
            # strict subset of the parent palette — shared-dict caching unsound.
            assert head_vals <= full_vals and len(head_vals) < len(full_vals)
            assert tail_vals <= full_vals
        else:
            # Preserve behaviour (polars 1.33.x): slices keep the full parent
            # palette. The converter still rebuilds per-slice, so this is fine.
            assert head_dict.equals(full_dict)
            assert tail_dict.equals(full_dict)

        # The binding contract: each frame's pandas view reflects that frame's
        # own values exactly — the converter rebuilds from the local dict and
        # does NOT splice in a shared parent palette.
        pd_full = get_pandas_view_of_polars_df(src)
        pd_head = get_pandas_view_of_polars_df(head)
        pd_tail = get_pandas_view_of_polars_df(tail)
        assert pd_full["x"].astype(str).tolist() == src["x"].cast(pl.String).to_list()
        assert pd_head["x"].astype(str).tolist() == head["x"].cast(pl.String).to_list()
        assert pd_tail["x"].astype(str).tolist() == tail["x"].cast(pl.String).to_list()
        # The converter exposes no shared-dict knob — the optimisation stays gated off.
        import inspect

        assert "shared_dict_cache" not in inspect.signature(get_pandas_view_of_polars_df).parameters


# ================================================================================================
# Drop Columns Tests
# ================================================================================================


class TestDropColumnsFromDataframe:
    """Tests for drop_columns_from_dataframe."""

    def test_drop_columns_pandas(self):
        """Test dropping columns from pandas DataFrame."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "c": [7, 8, 9],
            }
        )

        result = drop_columns_from_dataframe(df, additional_columns_to_drop=["a", "b"], verbose=0)

        assert list(result.columns) == ["c"]

    def test_drop_columns_polars(self):
        """Test dropping columns from Polars DataFrame."""
        df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
                "c": [7, 8, 9],
            }
        )

        result = drop_columns_from_dataframe(df, additional_columns_to_drop=["a", "b"], verbose=0)

        assert result.columns == ["c"]

    def test_drop_from_config(self):
        """Test dropping columns specified in config."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

        result = drop_columns_from_dataframe(df, config_drop_columns=["a"], verbose=0)

        assert "a" not in result.columns

    def test_drop_combined_sources(self):
        """Test dropping from both additional and config."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3], "d": [4]})

        result = drop_columns_from_dataframe(
            df,
            additional_columns_to_drop=["a"],
            config_drop_columns=["b"],
            verbose=0,
        )

        assert list(result.columns) == ["c", "d"]

    def test_no_columns_to_drop(self):
        """Test when no columns are specified to drop."""
        df = pd.DataFrame({"a": [1], "b": [2]})

        result = drop_columns_from_dataframe(df, verbose=0)

        assert list(result.columns) == ["a", "b"]

    def test_drop_nonexistent_column_pandas(self):
        """Test dropping non-existent column (pandas ignores it)."""
        df = pd.DataFrame({"a": [1], "b": [2]})

        result = drop_columns_from_dataframe(df, additional_columns_to_drop=["nonexistent"], verbose=0)

        assert list(result.columns) == ["a", "b"]

    def test_drop_nonexistent_column_polars(self):
        """Test dropping non-existent column (polars with strict=False)."""
        df = pl.DataFrame({"a": [1], "b": [2]})

        result = drop_columns_from_dataframe(df, additional_columns_to_drop=["nonexistent"], verbose=0)

        assert result.columns == ["a", "b"]

    def test_removes_duplicates(self):
        """Test that duplicate column names are removed."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

        result = drop_columns_from_dataframe(
            df,
            additional_columns_to_drop=["a", "a", "b"],
            config_drop_columns=["a"],
            verbose=0,
        )

        assert list(result.columns) == ["c"]


# ================================================================================================
# Save Series/DataFrame Tests
# ================================================================================================


class TestSaveSeriesOrDF:
    """Tests for save_series_or_df."""

    def test_save_pandas_dataframe(self, tmp_path):
        """Test saving pandas DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        file_path = str(tmp_path / "df.parquet")

        save_series_or_df(df, file_path)

        loaded = pd.read_parquet(file_path)
        pd.testing.assert_frame_equal(df, loaded)

    def test_save_polars_dataframe(self, tmp_path):
        """Test saving Polars DataFrame."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        file_path = str(tmp_path / "df.parquet")

        save_series_or_df(df, file_path)

        loaded = pl.read_parquet(file_path)
        assert df.equals(loaded)

    def test_save_pandas_series(self, tmp_path):
        """Test saving pandas Series (converts to DataFrame)."""
        series = pd.Series([1, 2, 3], name="values")
        file_path = str(tmp_path / "series.parquet")

        save_series_or_df(series, file_path)

        loaded = pd.read_parquet(file_path)
        assert "values" in loaded.columns

    def test_save_polars_series(self, tmp_path):
        """Test saving Polars Series (converts to DataFrame)."""
        series = pl.Series("values", [1, 2, 3])
        file_path = str(tmp_path / "series.parquet")

        save_series_or_df(series, file_path)

        loaded = pl.read_parquet(file_path)
        assert "values" in loaded.columns

    def test_save_series_with_custom_name(self, tmp_path):
        """Test saving series with custom name."""
        series = pd.Series([1, 2, 3])
        file_path = str(tmp_path / "named.parquet")

        save_series_or_df(series, file_path, name="custom_name")

        loaded = pd.read_parquet(file_path)
        assert "custom_name" in loaded.columns

    def test_custom_compression(self, tmp_path):
        """Test saving with custom compression."""
        df = pd.DataFrame({"a": list(range(1000))})
        file_path = str(tmp_path / "compressed.parquet")

        save_series_or_df(df, file_path, compression="snappy")

        loaded = pd.read_parquet(file_path)
        pd.testing.assert_frame_equal(df, loaded)


# ================================================================================================
# Process Special Values Tests
# ================================================================================================


class TestProcessNans:
    """Tests for process_nans."""

    def test_fill_nans_polars(self):
        """Test filling NaN values in Polars DataFrame."""
        df = pl.DataFrame(
            {
                "a": [1.0, float("nan"), 3.0],
                "b": [4.0, 5.0, float("nan")],
            }
        )

        result = process_nans(df, fill_value=0.0, verbose=0)

        # Check no NaNs remain
        assert result["a"].is_nan().sum() == 0
        assert result["b"].is_nan().sum() == 0

    def test_fill_nans_pandas(self):
        """Test filling NaN values in pandas DataFrame."""
        df = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0],
                "b": [4.0, 5.0, np.nan],
            }
        )

        result = process_nans(df, fill_value=-1.0, verbose=0)

        assert not result["a"].isna().any()
        assert not result["b"].isna().any()
        assert result["a"].iloc[1] == -1.0

    def test_fill_with_different_value(self):
        """Test filling NaNs with different fill values."""
        df = pl.DataFrame({"a": [float("nan"), 2.0]})

        result = process_nans(df, fill_value=999.0, verbose=0)

        assert result["a"].to_list() == [999.0, 2.0]


class TestProcessNulls:
    """Tests for process_nulls."""

    def test_fill_nulls_polars(self):
        """Test filling null values in Polars DataFrame."""
        df = pl.DataFrame(
            {
                "a": [1.0, None, 3.0],
                "b": [None, 5.0, 6.0],
            }
        )

        result = process_nulls(df, fill_value=0.0, verbose=0)

        assert result["a"].is_null().sum() == 0
        assert result["b"].is_null().sum() == 0

    def test_fill_nulls_pandas(self):
        """Test filling null values in pandas DataFrame."""
        df = pd.DataFrame(
            {
                "a": [1.0, None, 3.0],
                "b": [None, 5.0, 6.0],
            }
        )

        result = process_nulls(df, fill_value=0.0, verbose=0)

        assert not result["a"].isnull().any()
        assert not result["b"].isnull().any()


class TestProcessInfinities:
    """Tests for process_infinities."""

    def test_fill_infinities_polars(self):
        """Test filling infinite values in Polars DataFrame."""
        df = pl.DataFrame(
            {
                "a": [1.0, float("inf"), 3.0],
                "b": [float("-inf"), 5.0, 6.0],
            }
        )

        result = process_infinities(df, fill_value=0.0, verbose=0)

        assert result["a"].is_infinite().sum() == 0
        assert result["b"].is_infinite().sum() == 0

    def test_fill_infinities_pandas(self):
        """Test filling infinite values in pandas DataFrame."""
        df = pd.DataFrame(
            {
                "a": [1.0, float("inf"), 3.0],
                "b": [float("-inf"), 5.0, 6.0],
            }
        )

        result = process_infinities(df, fill_value=0.0, verbose=0)

        assert not np.isinf(result["a"]).any()
        assert not np.isinf(result["b"]).any()


class TestRemoveConstantColumns:
    """Tests for remove_constant_columns."""

    def test_remove_constant_numeric_polars(self):
        """Test removing constant numeric columns in Polars."""
        df = pl.DataFrame(
            {
                "varying": [1.0, 2.0, 3.0],
                "constant": [5.0, 5.0, 5.0],
            }
        )

        result = remove_constant_columns(df, verbose=0)

        assert "varying" in result.columns
        assert "constant" not in result.columns

    def test_remove_constant_numeric_pandas(self):
        """Test removing constant numeric columns in pandas."""
        df = pd.DataFrame(
            {
                "varying": [1.0, 2.0, 3.0],
                "constant": [5.0, 5.0, 5.0],
            }
        )

        result = remove_constant_columns(df, verbose=0)

        assert "varying" in result.columns
        assert "constant" not in result.columns

    def test_remove_constant_string_polars(self):
        """Test removing constant string columns in Polars."""
        df = pl.DataFrame(
            {
                "varying": ["a", "b", "c"],
                "constant": ["x", "x", "x"],
            }
        )

        result = remove_constant_columns(df, verbose=0)

        assert "varying" in result.columns
        assert "constant" not in result.columns

    def test_remove_constant_string_pandas(self):
        """Test removing constant string columns in pandas."""
        df = pd.DataFrame(
            {
                "varying": ["a", "b", "c"],
                "constant": ["x", "x", "x"],
            }
        )

        result = remove_constant_columns(df, verbose=0)

        assert "varying" in result.columns
        assert "constant" not in result.columns

    def test_keep_varying_columns(self):
        """Test that varying columns are kept."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        )

        result = remove_constant_columns(df, verbose=0)

        assert list(result.columns) == ["a", "b"]

    def test_remove_all_nan_columns_pandas(self):
        """Test removing columns that are all NaN."""
        df = pd.DataFrame(
            {
                "varying": [1.0, 2.0, 3.0],
                "all_nan": [np.nan, np.nan, np.nan],
            }
        )

        result = remove_constant_columns(df, verbose=0)

        assert "varying" in result.columns
        assert "all_nan" not in result.columns

    def test_mixed_constant_types(self):
        """Test with both numeric and categorical constant columns."""
        df = pd.DataFrame(
            {
                "varying_num": [1, 2, 3],
                "varying_str": ["a", "b", "c"],
                "const_num": [5, 5, 5],
                "const_str": ["x", "x", "x"],
            }
        )

        result = remove_constant_columns(df, verbose=0)

        assert "varying_num" in result.columns
        assert "varying_str" in result.columns
        assert "const_num" not in result.columns
        assert "const_str" not in result.columns


# ================================================================================================
# Hypothesis Property-Based Tests
# ================================================================================================


class TestHypothesisSaveLoad:
    """Hypothesis-based property tests for save/load functions."""

    @given(
        st.dictionaries(
            keys=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L", "N"))),
            values=st.one_of(
                st.integers(min_value=-1000000, max_value=1000000),
                st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                st.text(max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "P", "S"))),
                st.lists(st.integers(min_value=-1000, max_value=1000), max_size=10),
            ),
            min_size=1,
            max_size=5,
        )
    )
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_roundtrip_preserves_dict(self, model_data):
        """Property: save then load should return identical dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "model.zst")

            result = save_mlframe_model(model_data, file_path, verbose=0)
            assert result is True

            loaded = load_mlframe_model(file_path)
            assert loaded == model_data

    @given(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
            min_size=1,
            max_size=100,
        )
    )
    @settings(max_examples=20)
    def test_roundtrip_preserves_numpy_array(self, float_list):
        """Property: numpy arrays should be preserved after save/load."""
        arr = np.array(float_list)
        model = {"array": arr}

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "numpy_model.zst")

            save_mlframe_model(model, file_path, verbose=0)
            loaded = load_mlframe_model(file_path)

            np.testing.assert_array_almost_equal(loaded["array"], arr)


class TestHypothesisDataFrameConversion:
    """Hypothesis-based property tests for DataFrame conversions."""

    @given(st.integers(min_value=1, max_value=50), st.integers(min_value=1, max_value=5))
    @settings(max_examples=15)
    def test_polars_to_pandas_preserves_shape(self, n_rows, n_cols):
        """Property: Polars to pandas conversion should preserve shape."""
        # Generate random data
        columns = {f"col_{i}": np.random.randn(n_rows).tolist() for i in range(n_cols)}
        pl_df = pl.DataFrame(columns)

        pd_df = get_pandas_view_of_polars_df(pl_df)

        assert pd_df.shape == (n_rows, n_cols)
        assert list(pd_df.columns) == list(pl_df.columns)

    @given(
        st.lists(
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
            min_size=1,
            max_size=50,
        )
    )
    @settings(max_examples=15)
    def test_polars_to_pandas_preserves_values(self, values):
        """Property: Values should be approximately preserved after conversion."""
        pl_df = pl.DataFrame({"values": values})

        pd_df = get_pandas_view_of_polars_df(pl_df)

        # Convert to Python floats for comparison
        pd_values = pd_df["values"].to_list()
        for orig, converted in zip(values, pd_values):
            assert abs(orig - converted) < 1e-9


class TestHypothesisDropColumns:
    """Hypothesis-based property tests for drop_columns_from_dataframe."""

    @given(st.integers(min_value=2, max_value=5))
    @settings(max_examples=15)
    def test_drop_columns_removes_specified(self, n_cols):
        """Property: Specified columns should be removed."""
        col_names = [f"col_{i}" for i in range(n_cols)]

        df = pd.DataFrame({name: [1, 2, 3] for name in col_names})

        # Select subset of columns to drop (at least 1, leaving at least 1)
        n_to_drop = np.random.randint(1, n_cols)
        cols_to_drop = col_names[:n_to_drop]

        result = drop_columns_from_dataframe(df, additional_columns_to_drop=cols_to_drop, verbose=0)

        # Verify dropped columns are gone
        for col in cols_to_drop:
            assert col not in result.columns

        # Verify remaining columns exist
        for col in col_names[n_to_drop:]:
            assert col in result.columns


class TestHypothesisProcessNans:
    """Hypothesis-based property tests for process_nans."""

    @given(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        st.integers(min_value=5, max_value=50),
        st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=15)
    def test_process_nans_fills_all_with_value(self, fill_value, n_rows, n_nans):
        """Property: All NaNs should be filled with the specified value."""
        assume(n_nans < n_rows)

        # Create DataFrame with some NaNs
        values = np.random.randn(n_rows)
        nan_indices = np.random.choice(n_rows, n_nans, replace=False)
        values[nan_indices] = np.nan

        df = pd.DataFrame({"col": values})

        result = process_nans(df, fill_value=fill_value, verbose=0)

        # Verify no NaNs remain
        assert not result["col"].isna().any()

        # Verify fill value was used
        for idx in nan_indices:
            assert result["col"].iloc[idx] == fill_value


class TestHypothesisRemoveConstant:
    """Hypothesis-based property tests for remove_constant_columns."""

    @given(st.integers(min_value=3, max_value=20))
    @settings(max_examples=15)
    def test_constant_columns_removed(self, n_rows):
        """Property: Constant columns should be removed, varying preserved."""
        # Create mixed DataFrame
        df = pd.DataFrame(
            {
                "varying": np.random.randn(n_rows),
                "constant": [5.0] * n_rows,
            }
        )

        result = remove_constant_columns(df, verbose=0)

        assert "varying" in result.columns
        assert "constant" not in result.columns

    @given(st.integers(min_value=3, max_value=50))
    @settings(max_examples=10)
    def test_all_varying_columns_preserved(self, n_rows):
        """Property: All varying columns should be preserved."""
        df = pd.DataFrame(
            {
                "a": np.random.randn(n_rows),
                "b": np.random.randn(n_rows),
                "c": np.random.randn(n_rows),
            }
        )

        result = remove_constant_columns(df, verbose=0)

        # All columns should be preserved since they all vary
        assert set(result.columns) == {"a", "b", "c"}


class TestProcessNansHypothesis:
    @given(
        n_rows=st.integers(1, 100),
        n_cols=st.integers(1, 10),
        nan_fraction=st.floats(0.0, 0.5),
    )
    @settings(max_examples=30, deadline=None)
    def test_no_nans_remain(self, n_rows, n_cols, nan_fraction):
        data = np.random.randn(n_rows, n_cols)
        mask = np.random.random((n_rows, n_cols)) < nan_fraction
        data[mask] = np.nan
        df = pd.DataFrame(data, columns=[f"c{i}" for i in range(n_cols)])
        result = process_nans(df, fill_value=0.0, verbose=0)
        assert not result.isna().any().any()


class TestProcessInfinitiesHypothesis:
    @given(
        n_rows=st.integers(1, 50),
        n_cols=st.integers(1, 5),
    )
    @settings(max_examples=20, deadline=None)
    def test_no_infs_remain(self, n_rows, n_cols):
        data = np.random.randn(n_rows, n_cols)
        # Inject some infinities
        data[0, 0] = np.inf
        if n_rows > 1:
            data[1, 0] = -np.inf
        df = pd.DataFrame(data, columns=[f"c{i}" for i in range(n_cols)])
        result = process_infinities(df, fill_value=0.0, verbose=0)
        assert not np.isinf(result.values).any()


class TestRemoveConstantColumnsHypothesis:
    @given(
        n_rows=st.integers(2, 50),
        n_varying=st.integers(1, 5),
        n_constant=st.integers(0, 3),
    )
    @settings(max_examples=30, deadline=None)
    def test_only_varying_columns_remain(self, n_rows, n_varying, n_constant):
        cols = {}
        for i in range(n_varying):
            cols[f"vary_{i}"] = np.random.randn(n_rows)
        for i in range(n_constant):
            cols[f"const_{i}"] = np.full(n_rows, 42.0)
        df = pd.DataFrame(cols)
        result = remove_constant_columns(df, verbose=0)
        # All constant columns should be removed
        for col in result.columns:
            assert result[col].nunique() > 1
