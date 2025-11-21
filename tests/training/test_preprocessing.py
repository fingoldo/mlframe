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
    save_split_artifacts,
)
import os
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


class TestSaveSplitArtifacts:
    """Test saving split artifacts to disk."""

    def test_saves_timestamps_for_all_splits(self, tmp_path):
        """Test that timestamps are saved for train/val/test splits."""
        train_idx = np.arange(0, 70)
        val_idx = np.arange(70, 85)
        test_idx = np.arange(85, 100)
        timestamps = pd.Series(pd.date_range("2020-01-01", periods=100, freq="h"))

        data_dir = str(tmp_path)
        models_dir = "models"

        save_split_artifacts(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            timestamps=timestamps,
            group_ids_raw=None,
            artifacts=None,
            data_dir=data_dir,
            models_dir=models_dir,
            target_name="target",
            model_name="test_model",
        )

        # Check files were created
        base_path = os.path.join(data_dir, models_dir, "target", "test_model")
        assert os.path.exists(os.path.join(base_path, "train_timestamps.parquet"))
        assert os.path.exists(os.path.join(base_path, "val_timestamps.parquet"))
        assert os.path.exists(os.path.join(base_path, "test_timestamps.parquet"))

    def test_saves_group_ids(self, tmp_path):
        """Test that group IDs are saved for all splits."""
        train_idx = np.arange(0, 70)
        val_idx = np.arange(70, 85)
        test_idx = np.arange(85, 100)
        # Use pandas Series for proper indexing
        group_ids = pd.Series(np.random.randint(0, 10, 100))

        data_dir = str(tmp_path)
        models_dir = "models"

        save_split_artifacts(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            timestamps=None,
            group_ids_raw=group_ids,
            artifacts=None,
            data_dir=data_dir,
            models_dir=models_dir,
            target_name="target",
            model_name="test_model",
        )

        base_path = os.path.join(data_dir, models_dir, "target", "test_model")
        assert os.path.exists(os.path.join(base_path, "train_group_ids_raw.parquet"))
        assert os.path.exists(os.path.join(base_path, "val_group_ids_raw.parquet"))
        assert os.path.exists(os.path.join(base_path, "test_group_ids_raw.parquet"))

    def test_saves_artifacts(self, tmp_path):
        """Test that artifacts are saved for all splits."""
        train_idx = np.arange(0, 70)
        val_idx = np.arange(70, 85)
        test_idx = np.arange(85, 100)
        # Use pandas Series for proper indexing
        artifacts = pd.Series(range(100), name="artifact_col")

        data_dir = str(tmp_path)
        models_dir = "models"

        save_split_artifacts(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            timestamps=None,
            group_ids_raw=None,
            artifacts=artifacts,
            data_dir=data_dir,
            models_dir=models_dir,
            target_name="target",
            model_name="test_model",
        )

        base_path = os.path.join(data_dir, models_dir, "target", "test_model")
        assert os.path.exists(os.path.join(base_path, "train_artifacts.parquet"))
        assert os.path.exists(os.path.join(base_path, "val_artifacts.parquet"))
        assert os.path.exists(os.path.join(base_path, "test_artifacts.parquet"))

    def test_skips_none_indices(self, tmp_path):
        """Test that None indices are skipped."""
        train_idx = np.arange(0, 70)
        val_idx = None  # No validation set
        test_idx = np.arange(70, 100)
        timestamps = pd.Series(pd.date_range("2020-01-01", periods=100, freq="h"))

        data_dir = str(tmp_path)
        models_dir = "models"

        save_split_artifacts(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            timestamps=timestamps,
            group_ids_raw=None,
            artifacts=None,
            data_dir=data_dir,
            models_dir=models_dir,
            target_name="target",
            model_name="test_model",
        )

        base_path = os.path.join(data_dir, models_dir, "target", "test_model")
        assert os.path.exists(os.path.join(base_path, "train_timestamps.parquet"))
        assert not os.path.exists(os.path.join(base_path, "val_timestamps.parquet"))
        assert os.path.exists(os.path.join(base_path, "test_timestamps.parquet"))

    def test_does_nothing_when_data_dir_is_none(self, tmp_path):
        """Test that nothing is saved when data_dir is None."""
        train_idx = np.arange(0, 70)
        val_idx = np.arange(70, 85)
        test_idx = np.arange(85, 100)
        timestamps = pd.Series(pd.date_range("2020-01-01", periods=100, freq="h"))

        save_split_artifacts(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            timestamps=timestamps,
            group_ids_raw=None,
            artifacts=None,
            data_dir=None,  # No data dir
            models_dir="models",
            target_name="target",
            model_name="test_model",
        )

        # No files should be created anywhere
        assert not os.path.exists(os.path.join(str(tmp_path), "models"))

    def test_does_not_overwrite_existing_files(self, tmp_path):
        """Test that existing files are not overwritten."""
        train_idx = np.arange(0, 70)
        val_idx = np.arange(70, 85)
        test_idx = np.arange(85, 100)
        timestamps = pd.Series(pd.date_range("2020-01-01", periods=100, freq="h"))

        data_dir = str(tmp_path)
        models_dir = "models"

        # First save
        save_split_artifacts(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            timestamps=timestamps,
            group_ids_raw=None,
            artifacts=None,
            data_dir=data_dir,
            models_dir=models_dir,
            target_name="target",
            model_name="test_model",
        )

        base_path = os.path.join(data_dir, models_dir, "target", "test_model")
        file_path = os.path.join(base_path, "train_timestamps.parquet")
        mtime_before = os.path.getmtime(file_path)

        # Wait a tiny bit to ensure mtime would change
        import time
        time.sleep(0.01)

        # Second save with different data
        new_timestamps = pd.Series(pd.date_range("2021-01-01", periods=100, freq="h"))
        save_split_artifacts(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            timestamps=new_timestamps,
            group_ids_raw=None,
            artifacts=None,
            data_dir=data_dir,
            models_dir=models_dir,
            target_name="target",
            model_name="test_model",
        )

        mtime_after = os.path.getmtime(file_path)
        assert mtime_before == mtime_after  # File should not be modified

    def test_slugifies_target_and_model_names(self, tmp_path):
        """Test that target and model names are slugified."""
        train_idx = np.arange(0, 70)
        val_idx = np.arange(70, 85)
        test_idx = np.arange(85, 100)
        timestamps = pd.Series(pd.date_range("2020-01-01", periods=100, freq="h"))

        data_dir = str(tmp_path)
        models_dir = "models"

        save_split_artifacts(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            timestamps=timestamps,
            group_ids_raw=None,
            artifacts=None,
            data_dir=data_dir,
            models_dir=models_dir,
            target_name="My Target Name",  # Needs slugification
            model_name="Model With Spaces",  # Needs slugification
        )

        # Check slugified path exists
        base_path = os.path.join(data_dir, models_dir, "my-target-name", "model-with-spaces")
        assert os.path.exists(os.path.join(base_path, "train_timestamps.parquet"))

    def test_handles_polars_series(self, tmp_path):
        """Test that Polars Series are handled correctly."""
        train_idx = np.arange(0, 70)
        val_idx = np.arange(70, 85)
        test_idx = np.arange(85, 100)
        timestamps = pl.Series("ts", pd.date_range("2020-01-01", periods=100, freq="h").tolist())

        data_dir = str(tmp_path)
        models_dir = "models"

        save_split_artifacts(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            timestamps=timestamps,
            group_ids_raw=None,
            artifacts=None,
            data_dir=data_dir,
            models_dir=models_dir,
            target_name="target",
            model_name="test_model",
        )

        base_path = os.path.join(data_dir, models_dir, "target", "test_model")
        assert os.path.exists(os.path.join(base_path, "train_timestamps.parquet"))

    def test_custom_compression(self, tmp_path):
        """Test saving with custom compression."""
        train_idx = np.arange(0, 70)
        val_idx = np.arange(70, 85)
        test_idx = np.arange(85, 100)
        timestamps = pd.Series(pd.date_range("2020-01-01", periods=100, freq="h"))

        data_dir = str(tmp_path)
        models_dir = "models"

        save_split_artifacts(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            timestamps=timestamps,
            group_ids_raw=None,
            artifacts=None,
            data_dir=data_dir,
            models_dir=models_dir,
            target_name="target",
            model_name="test_model",
            compression="snappy",
        )

        base_path = os.path.join(data_dir, models_dir, "target", "test_model")
        assert os.path.exists(os.path.join(base_path, "train_timestamps.parquet"))
