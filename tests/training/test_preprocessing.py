"""
Tests for preprocessing functions.
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl

from mlframe.training.preprocessing import (
    load_and_prepare_dataframe,
    preprocess_dataframe,
    create_split_dataframes,
)
from mlframe.training_old import make_train_test_split
from mlframe.training.configs import PreprocessingConfig, TrainingSplitConfig
from mlframe.training.utils import process_nans, process_nulls, remove_constant_columns


class TestDataLoading:
    """Test data loading and preparation."""

    def test_load_from_dataframe(self, sample_polars_data):
        """Test loading from existing Polars DataFrame."""
        pl_df, _, _ = sample_polars_data
        config = PreprocessingConfig()

        result = load_and_prepare_dataframe(pl_df, config, verbose=0)

        assert len(result) == len(pl_df)
        assert list(result.columns) == list(pl_df.columns)

    def test_tail_selection(self, sample_polars_data):
        """Test tail selection."""
        pl_df, _, _ = sample_polars_data
        config = PreprocessingConfig(tail=100)

        result = load_and_prepare_dataframe(pl_df, config, verbose=0)

        assert len(result) == 100

    def test_column_selection(self, sample_polars_data):
        """Test column selection (at load time)."""
        pl_df, feature_names, _ = sample_polars_data
        selected_cols = feature_names[:5] + ["target"]
        config = PreprocessingConfig(columns=selected_cols)

        # Save to parquet and load with column selection
        import tempfile
        import os

        # Create temp file and explicitly close before use (Windows compatibility)
        f = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        fname = f.name
        f.close()

        try:
            pl_df.write_parquet(fname)
            result = load_and_prepare_dataframe(fname, config, verbose=0)
            assert list(result.columns) == selected_cols
        finally:
            # Clean up temp file
            if os.path.exists(fname):
                os.unlink(fname)

    def test_load_from_polars(self):
        """Test loading from Polars DataFrame."""
        import polars as pl

        df = pl.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [4, 5, 6],
            }
        )
        config = PreprocessingConfig()

        result = load_and_prepare_dataframe(df, config, verbose=0)

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 3


class TestDataCleaning:
    """Test data cleaning functions."""

    def test_process_nans(self):
        """Test NaN processing."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, np.nan, 4.0],
                "b": [np.nan, 2.0, 3.0, 4.0],
                "c": ["x", "y", "z", "w"],
            }
        )

        result = process_nans(df, fill_value=0.0, verbose=0)

        assert not result["a"].isna().any()
        assert not result["b"].isna().any()
        assert result["a"].iloc[2] == 0.0

    def test_process_nulls(self):
        """Test NULL processing."""
        df = pd.DataFrame(
            {
                "a": [1.0, 2.0, None, 4.0],
                "b": [1.0, 2.0, 3.0, 4.0],
            }
        )

        result = process_nulls(df, fill_value=-999.0, verbose=0)

        assert not result["a"].isnull().any()
        assert result.loc[2, "a"] == -999.0

    def test_remove_constant_columns(self):
        """Test constant column removal."""
        df = pd.DataFrame(
            {
                "const_num": [5.0] * 10,
                "varying_num": range(10),
                "const_cat": ["a"] * 10,
                "varying_cat": ["a", "b"] * 5,
            }
        )

        result = remove_constant_columns(df, verbose=0)

        assert "const_num" not in result.columns
        assert "const_cat" not in result.columns
        assert "varying_num" in result.columns
        assert "varying_cat" in result.columns


class TestPreprocessing:
    """Test full preprocessing pipeline."""

    def test_preprocess_dataframe_basic(self, sample_regression_data):
        """Test basic preprocessing."""
        df, _, _ = sample_regression_data

        # Add some issues
        df_messy = df.copy()
        df_messy.loc[10, "feature_0"] = np.nan
        df_messy.loc[20, "feature_1"] = np.inf
        df_messy["constant_col"] = 5.0

        config = PreprocessingConfig(
            fillna_value=0.0,
            fix_infinities=True,
        )

        result = preprocess_dataframe(df_messy, config, verbose=0)

        # Constant column should be removed
        assert "constant_col" not in result.columns
        # NaNs and infinities should be filled
        assert not result.isna().any().any()

    def test_preprocess_polars_dataframe(self, sample_polars_data):
        """Test preprocessing with Polars DataFrame."""
        pl_df, _, _ = sample_polars_data

        config = PreprocessingConfig(fillna_value=0.0)

        result = preprocess_dataframe(pl_df, config, verbose=0)

        assert isinstance(result, pl.DataFrame)


class TestTrainTestSplit:
    """Test train/val/test splitting."""

    def test_basic_split(self, sample_regression_data):
        """Test basic split without timestamps."""
        df, _, _ = sample_regression_data

        config = TrainingSplitConfig(
            test_size=0.2,
            val_size=0.1,
            shuffle_test=False,
            shuffle_val=False,
        )

        train_idx, val_idx, test_idx, _, _, _ = make_train_test_split(df, **config.model_dump(), timestamps=None)

        # Check sizes are approximately correct
        total = len(train_idx) + len(val_idx) + len(test_idx)
        assert total == len(df)
        assert len(test_idx) == pytest.approx(len(df) * 0.2, abs=2)
        # Val size is 0.1 of remaining after test is taken out
        # So actual val size ≈ 0.1 * (1 - 0.2) * len(df) = 0.08 * len(df)
        assert len(val_idx) == pytest.approx(len(df) * 0.1 * (1 - 0.2), abs=2)

    def test_split_with_timestamps(self, sample_timeseries_data):
        """Test split with timestamps."""
        df, _, dates, _ = sample_timeseries_data

        # Use sequential splitting (not shuffled)
        config = TrainingSplitConfig(
            test_size=0.2,
            val_size=0.1,
            shuffle_test=False,
            shuffle_val=False,
            wholeday_splitting=False,
        )

        # Convert DatetimeIndex to Series (make_train_test_split expects pd.Series)
        timestamps = pd.Series(dates, name='timestamp')
        train_idx, val_idx, test_idx, _, _, _ = make_train_test_split(df, **config.model_dump(), timestamps=timestamps)

        # With sequential splitting, test and val should come after train
        # Indices are sorted, so we just check that sets don't overlap
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0

    def test_split_with_shuffling(self, sample_regression_data):
        """Test split with shuffling."""
        df, _, _ = sample_regression_data

        config = TrainingSplitConfig(
            test_size=0.2,
            val_size=0.1,
            shuffle_test=True,
            shuffle_val=True,
            random_seed=42,
        )

        train_idx1, val_idx1, test_idx1, _, _, _ = make_train_test_split(df, **config.model_dump(), timestamps=None)

        # Run again with same seed
        train_idx2, val_idx2, test_idx2, _, _, _ = make_train_test_split(df, **config.model_dump(), timestamps=None)

        # Should be reproducible
        assert np.array_equal(train_idx1, train_idx2)
        assert np.array_equal(val_idx1, val_idx2)
        assert np.array_equal(test_idx1, test_idx2)

    def test_mixed_sequential_shuffled_split(self, sample_regression_data):
        """Test split with mixed sequential and shuffled portions."""
        df, _, _ = sample_regression_data

        config = TrainingSplitConfig(
            test_size=0.2,
            val_size=0.1,
            shuffle_test=False,
            shuffle_val=False,
            val_sequential_fraction=0.5,  # 50% sequential, 50% shuffled
        )

        train_idx, val_idx, test_idx, _, _, _ = make_train_test_split(df, **config.model_dump(), timestamps=None)

        assert len(val_idx) > 0
        assert len(test_idx) > 0


class TestSplitDataframes:
    """Test creating split dataframes."""

    def test_create_split_dataframes_pandas(self, sample_regression_data):
        """Test creating splits from pandas DataFrame."""
        df, _, _ = sample_regression_data

        train_idx = np.arange(0, 700)
        val_idx = np.arange(700, 850)
        test_idx = np.arange(850, 1000)

        train_df, val_df, test_df = create_split_dataframes(df, train_idx, val_idx, test_idx)

        assert len(train_df) == 700
        assert len(val_df) == 150
        assert len(test_df) == 150
        assert isinstance(train_df, pd.DataFrame)

    def test_create_split_dataframes_polars(self, sample_polars_data):
        """Test creating splits from Polars DataFrame."""
        pl_df, _, _ = sample_polars_data

        train_idx = np.arange(0, 700)
        val_idx = np.arange(700, 850)
        test_idx = np.arange(850, 1000)

        train_df, val_df, test_df = create_split_dataframes(pl_df, train_idx, val_idx, test_idx)

        assert len(train_df) == 700
        assert len(val_df) == 150
        assert len(test_df) == 150
        assert isinstance(train_df, pl.DataFrame)
