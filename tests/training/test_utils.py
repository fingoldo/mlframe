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
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from mlframe.training.utils import (
    get_pandas_view_of_polars_df,
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
        np.testing.assert_array_almost_equal(
            model.predict(X), loaded.predict(X)
        )

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
        pl_df = pl.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.0, 2.0, 3.0],
        })

        pd_df = get_pandas_view_of_polars_df(pl_df)

        assert isinstance(pd_df, pd.DataFrame)
        assert list(pd_df.columns) == ["int_col", "float_col"]
        assert len(pd_df) == 3

    def test_string_columns(self):
        """Test conversion of string columns."""
        pl_df = pl.DataFrame({
            "name": ["Alice", "Bob", "Charlie"],
        })

        pd_df = get_pandas_view_of_polars_df(pl_df)

        assert pd_df["name"].tolist() == ["Alice", "Bob", "Charlie"]

    def test_categorical_to_string_conversion(self):
        """Test that categorical columns are converted to strings."""
        pl_df = pl.DataFrame({
            "category": pl.Series(["A", "B", "A", "C"]).cast(pl.Categorical),
        })

        pd_df = get_pandas_view_of_polars_df(pl_df)

        # Should be converted to string
        assert pd_df["category"].tolist() == ["A", "B", "A", "C"]

    def test_boolean_columns(self):
        """Test conversion of boolean columns."""
        pl_df = pl.DataFrame({
            "bool_col": [True, False, True],
        })

        pd_df = get_pandas_view_of_polars_df(pl_df)

        assert pd_df["bool_col"].tolist() == [True, False, True]

    def test_mixed_column_types(self):
        """Test conversion with mixed column types."""
        pl_df = pl.DataFrame({
            "int": [1, 2, 3],
            "float": [1.1, 2.2, 3.3],
            "str": ["a", "b", "c"],
            "bool": [True, False, True],
        })

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
        pl_df = pl.DataFrame({
            "z": [1],
            "a": [2],
            "m": [3],
        })

        pd_df = get_pandas_view_of_polars_df(pl_df)

        assert list(pd_df.columns) == ["z", "a", "m"]


# ================================================================================================
# Drop Columns Tests
# ================================================================================================

class TestDropColumnsFromDataframe:
    """Tests for drop_columns_from_dataframe."""

    def test_drop_columns_pandas(self):
        """Test dropping columns from pandas DataFrame."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        })

        result = drop_columns_from_dataframe(
            df, additional_columns_to_drop=["a", "b"], verbose=0
        )

        assert list(result.columns) == ["c"]

    def test_drop_columns_polars(self):
        """Test dropping columns from Polars DataFrame."""
        df = pl.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [7, 8, 9],
        })

        result = drop_columns_from_dataframe(
            df, additional_columns_to_drop=["a", "b"], verbose=0
        )

        assert result.columns == ["c"]

    def test_drop_from_config(self):
        """Test dropping columns specified in config."""
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

        result = drop_columns_from_dataframe(
            df, config_drop_columns=["a"], verbose=0
        )

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

        result = drop_columns_from_dataframe(
            df, additional_columns_to_drop=["nonexistent"], verbose=0
        )

        assert list(result.columns) == ["a", "b"]

    def test_drop_nonexistent_column_polars(self):
        """Test dropping non-existent column (polars with strict=False)."""
        df = pl.DataFrame({"a": [1], "b": [2]})

        result = drop_columns_from_dataframe(
            df, additional_columns_to_drop=["nonexistent"], verbose=0
        )

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
        df = pl.DataFrame({
            "a": [1.0, float("nan"), 3.0],
            "b": [4.0, 5.0, float("nan")],
        })

        result = process_nans(df, fill_value=0.0, verbose=0)

        # Check no NaNs remain
        assert result["a"].is_nan().sum() == 0
        assert result["b"].is_nan().sum() == 0

    def test_fill_nans_pandas(self):
        """Test filling NaN values in pandas DataFrame."""
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [4.0, 5.0, np.nan],
        })

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
        df = pl.DataFrame({
            "a": [1.0, None, 3.0],
            "b": [None, 5.0, 6.0],
        })

        result = process_nulls(df, fill_value=0.0, verbose=0)

        assert result["a"].is_null().sum() == 0
        assert result["b"].is_null().sum() == 0

    def test_fill_nulls_pandas(self):
        """Test filling null values in pandas DataFrame."""
        df = pd.DataFrame({
            "a": [1.0, None, 3.0],
            "b": [None, 5.0, 6.0],
        })

        result = process_nulls(df, fill_value=0.0, verbose=0)

        assert not result["a"].isnull().any()
        assert not result["b"].isnull().any()


class TestProcessInfinities:
    """Tests for process_infinities."""

    def test_fill_infinities_polars(self):
        """Test filling infinite values in Polars DataFrame."""
        df = pl.DataFrame({
            "a": [1.0, float("inf"), 3.0],
            "b": [float("-inf"), 5.0, 6.0],
        })

        result = process_infinities(df, fill_value=0.0, verbose=0)

        assert result["a"].is_infinite().sum() == 0
        assert result["b"].is_infinite().sum() == 0

    def test_fill_infinities_pandas(self):
        """Test filling infinite values in pandas DataFrame."""
        df = pd.DataFrame({
            "a": [1.0, float("inf"), 3.0],
            "b": [float("-inf"), 5.0, 6.0],
        })

        result = process_infinities(df, fill_value=0.0, verbose=0)

        assert not np.isinf(result["a"]).any()
        assert not np.isinf(result["b"]).any()


class TestRemoveConstantColumns:
    """Tests for remove_constant_columns."""

    def test_remove_constant_numeric_polars(self):
        """Test removing constant numeric columns in Polars."""
        df = pl.DataFrame({
            "varying": [1.0, 2.0, 3.0],
            "constant": [5.0, 5.0, 5.0],
        })

        result = remove_constant_columns(df, verbose=0)

        assert "varying" in result.columns
        assert "constant" not in result.columns

    def test_remove_constant_numeric_pandas(self):
        """Test removing constant numeric columns in pandas."""
        df = pd.DataFrame({
            "varying": [1.0, 2.0, 3.0],
            "constant": [5.0, 5.0, 5.0],
        })

        result = remove_constant_columns(df, verbose=0)

        assert "varying" in result.columns
        assert "constant" not in result.columns

    def test_remove_constant_string_polars(self):
        """Test removing constant string columns in Polars."""
        df = pl.DataFrame({
            "varying": ["a", "b", "c"],
            "constant": ["x", "x", "x"],
        })

        result = remove_constant_columns(df, verbose=0)

        assert "varying" in result.columns
        assert "constant" not in result.columns

    def test_remove_constant_string_pandas(self):
        """Test removing constant string columns in pandas."""
        df = pd.DataFrame({
            "varying": ["a", "b", "c"],
            "constant": ["x", "x", "x"],
        })

        result = remove_constant_columns(df, verbose=0)

        assert "varying" in result.columns
        assert "constant" not in result.columns

    def test_keep_varying_columns(self):
        """Test that varying columns are kept."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        })

        result = remove_constant_columns(df, verbose=0)

        assert list(result.columns) == ["a", "b"]

    def test_remove_all_nan_columns_pandas(self):
        """Test removing columns that are all NaN."""
        df = pd.DataFrame({
            "varying": [1.0, 2.0, 3.0],
            "all_nan": [np.nan, np.nan, np.nan],
        })

        result = remove_constant_columns(df, verbose=0)

        assert "varying" in result.columns
        assert "all_nan" not in result.columns

    def test_mixed_constant_types(self):
        """Test with both numeric and categorical constant columns."""
        df = pd.DataFrame({
            "varying_num": [1, 2, 3],
            "varying_str": ["a", "b", "c"],
            "const_num": [5, 5, 5],
            "const_str": ["x", "x", "x"],
        })

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

    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('L', 'N'))),
        values=st.one_of(
            st.integers(min_value=-1000000, max_value=1000000),
            st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
            st.text(max_size=50, alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'S'))),
            st.lists(st.integers(min_value=-1000, max_value=1000), max_size=10),
        ),
        min_size=1,
        max_size=5,
    ))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_roundtrip_preserves_dict(self, model_data):
        """Property: save then load should return identical dict."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "model.zst")

            result = save_mlframe_model(model_data, file_path, verbose=0)
            assert result is True

            loaded = load_mlframe_model(file_path)
            assert loaded == model_data

    @given(st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1,
        max_size=100,
    ))
    @settings(max_examples=20)
    def test_roundtrip_preserves_numpy_array(self, float_list):
        """Property: numpy arrays should be preserved after save/load."""
        import tempfile
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

    @given(st.lists(
        st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
        min_size=1,
        max_size=50,
    ))
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

        result = drop_columns_from_dataframe(
            df, additional_columns_to_drop=cols_to_drop, verbose=0
        )

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
        df = pd.DataFrame({
            "varying": np.random.randn(n_rows),
            "constant": [5.0] * n_rows,
        })

        result = remove_constant_columns(df, verbose=0)

        assert "varying" in result.columns
        assert "constant" not in result.columns

    @given(st.integers(min_value=3, max_value=50))
    @settings(max_examples=10)
    def test_all_varying_columns_preserved(self, n_rows):
        """Property: All varying columns should be preserved."""
        df = pd.DataFrame({
            "a": np.random.randn(n_rows),
            "b": np.random.randn(n_rows),
            "c": np.random.randn(n_rows),
        })

        result = remove_constant_columns(df, verbose=0)

        # All columns should be preserved since they all vary
        assert set(result.columns) == {"a", "b", "c"}
