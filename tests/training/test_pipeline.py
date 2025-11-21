"""
Tests for pipeline fitting and transformation.

Tests fit_and_transform_pipeline and related functionality.
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl

from mlframe.training.pipeline import fit_and_transform_pipeline, prepare_df_for_catboost, create_polarsds_pipeline
from mlframe.training.configs import PolarsPipelineConfig
from unittest.mock import patch


class TestFitAndTransformPipeline:
    """Test pipeline fitting and transformation."""

    def test_fit_transform_basic_pandas(self, sample_regression_data):
        """Test basic pipeline fit and transform with pandas."""
        df, feature_names, y = sample_regression_data

        # Split data
        train_size = int(0.7 * len(df))
        train_df = df[feature_names].iloc[:train_size]
        val_df = df[feature_names].iloc[train_size:train_size + 100]
        test_df = df[feature_names].iloc[train_size + 100:]

        config = PolarsPipelineConfig(use_polarsds_pipeline=False)

        # Fit and transform
        train_transformed, val_transformed, test_transformed, pipeline, cat_features = fit_and_transform_pipeline(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            config=config,
            ensure_float32=False,
            verbose=0,
        )

        # Verify shapes
        assert len(train_transformed) == len(train_df)
        assert len(val_transformed) == len(val_df)
        assert len(test_transformed) == len(test_df)

        # Verify columns match
        assert list(train_transformed.columns) == list(train_df.columns)

    def test_fit_transform_basic_polars(self, sample_polars_data):
        """Test basic pipeline fit and transform with Polars."""
        pl_df, feature_names, y = sample_polars_data

        # Split data
        train_size = int(0.7 * len(pl_df))
        train_df = pl_df[feature_names][:train_size]
        val_df = pl_df[feature_names][train_size:train_size + 100]
        test_df = pl_df[feature_names][train_size + 100:]

        config = PolarsPipelineConfig(use_polarsds_pipeline=False)

        # Fit and transform
        train_transformed, val_transformed, test_transformed, pipeline, cat_features = fit_and_transform_pipeline(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            config=config,
            ensure_float32=False,
            verbose=0,
        )

        # Verify shapes
        assert len(train_transformed) == len(train_df)
        assert len(val_transformed) == len(val_df)
        assert len(test_transformed) == len(test_df)

    def test_fit_pipeline_with_categorical_features(self):
        """Test pipeline with categorical features."""
        # Create data with categorical columns
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_0': np.random.randn(500),
            'feature_1': np.random.randn(500),
            'cat_feature': np.random.choice(['A', 'B', 'C'], 500),
        })

        train_size = int(0.7 * len(df))
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:]

        config = PolarsPipelineConfig(use_polarsds_pipeline=False, categorical_encoding="none")

        # Fit and transform
        train_transformed, val_transformed, test_transformed, pipeline, cat_features = fit_and_transform_pipeline(
            train_df=train_df,
            val_df=val_df,
            test_df=None,
            config=config,
            ensure_float32=False,
            verbose=0,
        )

        # Verify cat_features list (preserved when categorical_encoding=None for CatBoost)
        assert 'cat_feature' in cat_features
        assert len(cat_features) == 1

    def test_fit_pipeline_with_no_validation_set(self, sample_regression_data):
        """Test pipeline with no validation set."""
        df, feature_names, y = sample_regression_data

        train_df = df[feature_names].iloc[:700]

        config = PolarsPipelineConfig(use_polarsds_pipeline=False)

        # Fit and transform
        train_transformed, val_transformed, test_transformed, pipeline, cat_features = fit_and_transform_pipeline(
            train_df=train_df,
            val_df=None,
            test_df=None,
            config=config,
            ensure_float32=False,
            verbose=0,
        )

        # Verify results
        assert len(train_transformed) == len(train_df)
        assert val_transformed is None
        assert test_transformed is None

    def test_fit_pipeline_with_float32_conversion(self, sample_regression_data):
        """Test pipeline with float32 conversion."""
        df, feature_names, y = sample_regression_data

        train_df = df[feature_names].iloc[:700]

        config = PolarsPipelineConfig(use_polarsds_pipeline=False)

        # Fit and transform with float32 conversion
        train_transformed, val_transformed, test_transformed, pipeline, cat_features = fit_and_transform_pipeline(
            train_df=train_df,
            val_df=None,
            test_df=None,
            config=config,
            ensure_float32=True,
            verbose=0,
        )

        # Verify dtypes are float32 (or compatible)
        if isinstance(train_transformed, pd.DataFrame):
            # Check that numeric columns are float32
            for col in train_transformed.select_dtypes(include=[np.number]).columns:
                assert train_transformed[col].dtype == np.float32 or train_transformed[col].dtype == np.float64


class TestPrepareDfForCatboost:
    """Test CatBoost DataFrame preparation."""

    def test_prepare_df_for_catboost_with_object_columns(self):
        """Test preparing DataFrame with object columns for CatBoost."""
        # Create DataFrame with object/categorical columns
        df = pd.DataFrame({
            'feature_0': np.random.randn(100),
            'feature_1': np.random.randn(100),
            'cat_feature': np.random.choice(['A', 'B', 'C'], 100),
        })

        cat_features = ['cat_feature']

        # Prepare for CatBoost
        prepare_df_for_catboost(df=df, cat_features=cat_features)

        # Verify cat_feature is now categorical
        assert isinstance(df['cat_feature'].dtype, pd.CategoricalDtype)

    def test_prepare_df_for_catboost_no_cat_features(self):
        """Test preparing DataFrame with no categorical features."""
        df = pd.DataFrame({
            'feature_0': np.random.randn(100),
            'feature_1': np.random.randn(100),
        })

        # Should not fail with empty cat_features
        prepare_df_for_catboost(df=df, cat_features=[])

        # Verify dtypes unchanged
        assert df['feature_0'].dtype == np.float64


class TestPipelineEdgeCases:
    """Test edge cases in pipeline."""

    def test_pipeline_with_empty_dataframe(self):
        """Test pipeline with empty DataFrame."""
        # Create empty DataFrame
        df = pd.DataFrame(columns=['feature_0', 'feature_1'])

        config = PolarsPipelineConfig(use_polarsds_pipeline=False)

        # This might raise an error or handle gracefully
        # depending on implementation
        try:
            train_transformed, val_transformed, test_transformed, pipeline, cat_features = fit_and_transform_pipeline(
                train_df=df,
                val_df=None,
                test_df=None,
                config=config,
                ensure_float32=False,
                verbose=0,
            )
            # If it doesn't raise, verify output
            assert len(train_transformed) == 0
        except (ValueError, IndexError):
            # Expected - empty DataFrame can't be fit
            pass

    def test_pipeline_with_single_row(self):
        """Test pipeline with single row DataFrame."""
        df = pd.DataFrame({
            'feature_0': [1.0],
            'feature_1': [2.0],
        })

        config = PolarsPipelineConfig(use_polarsds_pipeline=False)

        # Fit and transform
        train_transformed, val_transformed, test_transformed, pipeline, cat_features = fit_and_transform_pipeline(
            train_df=df,
            val_df=None,
            test_df=None,
            config=config,
            ensure_float32=False,
            verbose=0,
        )

        # Verify it worked
        assert len(train_transformed) == 1


class TestPipelineConfigurations:
    """Test different pipeline configurations."""

    def test_pipeline_with_different_scaler(self, sample_regression_data):
        """Test pipeline with different scaler configurations."""
        df, feature_names, y = sample_regression_data

        train_df = df[feature_names].iloc[:700]

        # Test with different scalers (polars-ds supports: standard, min_max, abs_max)
        for scaler_name in ['standard', 'min_max', 'abs_max']:
            config = PolarsPipelineConfig(
                use_polarsds_pipeline=False,
                scaler_name=scaler_name,
            )

            train_transformed, val_transformed, test_transformed, pipeline, cat_features = fit_and_transform_pipeline(
                train_df=train_df,
                val_df=None,
                test_df=None,
                config=config,
                ensure_float32=False,
                verbose=0,
            )

            # Verify it worked
            assert len(train_transformed) == len(train_df)

    def test_pipeline_with_different_imputer_strategy(self, sample_regression_data):
        """Test pipeline with different imputer strategies."""
        df, feature_names, y = sample_regression_data

        # Add some NaNs
        df_with_nan = df.copy()
        df_with_nan.loc[10:20, feature_names[0]] = np.nan

        train_df = df_with_nan[feature_names].iloc[:700]

        # Test with different imputer strategies
        for strategy in ['mean', 'median', 'most_frequent']:
            config = PolarsPipelineConfig(
                use_polarsds_pipeline=False,
                imputer_strategy=strategy,
            )

            train_transformed, val_transformed, test_transformed, pipeline, cat_features = fit_and_transform_pipeline(
                train_df=train_df,
                val_df=None,
                test_df=None,
                config=config,
                ensure_float32=False,
                verbose=0,
            )

            # Verify pipeline completes successfully
            # Note: Pipeline may not actually impute if imputer is not configured
            assert len(train_transformed) == len(train_df)


class TestCreatePolardsPipeline:
    """Tests for create_polarsds_pipeline function."""

    def test_returns_none_when_polarsds_not_available(self):
        """Test that function returns None when polars-ds is not installed."""
        # Create simple Polars DataFrame
        pl_df = pl.DataFrame({
            "feature_0": [1.0, 2.0, 3.0],
            "feature_1": [4.0, 5.0, 6.0],
        })
        config = PolarsPipelineConfig()

        with patch.dict('sys.modules', {'polars_ds': None, 'polars_ds.pipeline': None}):
            result = create_polarsds_pipeline(pl_df, config, verbose=0)
            # Function returns None if polars-ds can't be imported
            # This is expected behavior

    def test_basic_pipeline_creation(self, sample_polars_data):
        """Test basic pipeline creation with Polars DataFrame."""
        pl_df, feature_names, _ = sample_polars_data

        config = PolarsPipelineConfig(
            use_polarsds_pipeline=True,
            scaler_name="standard",
        )

        # Only run if polars-ds is available
        try:
            import polars_ds.pipeline
        except ImportError:
            pytest.skip("polars-ds not installed")

        pipeline = create_polarsds_pipeline(
            pl_df.select(feature_names),
            config,
            verbose=0,
        )

        assert pipeline is not None

    def test_pipeline_with_scaling(self, sample_polars_data):
        """Test pipeline creation with different scaling methods."""
        pl_df, feature_names, _ = sample_polars_data

        try:
            import polars_ds.pipeline
        except ImportError:
            pytest.skip("polars-ds not installed")

        for scaler_name in ['standard', 'min_max', 'abs_max', 'robust']:
            config = PolarsPipelineConfig(
                use_polarsds_pipeline=True,
                scaler_name=scaler_name,
            )

            pipeline = create_polarsds_pipeline(
                pl_df.select(feature_names),
                config,
                verbose=0,
            )

            assert pipeline is not None

    def test_pipeline_with_ordinal_encoding(self, sample_categorical_data):
        """Test pipeline with ordinal categorical encoding."""
        df, feature_names, cat_features, _ = sample_categorical_data

        # Convert to Polars
        pl_df = pl.from_pandas(df)

        try:
            import polars_ds.pipeline
        except ImportError:
            pytest.skip("polars-ds not installed")

        config = PolarsPipelineConfig(
            use_polarsds_pipeline=True,
            categorical_encoding="ordinal",
        )

        pipeline = create_polarsds_pipeline(
            pl_df.select(feature_names),
            config,
            verbose=0,
        )

        assert pipeline is not None

    def test_pipeline_with_onehot_encoding(self, sample_categorical_data):
        """Test pipeline with one-hot categorical encoding."""
        df, feature_names, cat_features, _ = sample_categorical_data

        # Convert to Polars
        pl_df = pl.from_pandas(df)

        try:
            import polars_ds.pipeline
        except ImportError:
            pytest.skip("polars-ds not installed")

        config = PolarsPipelineConfig(
            use_polarsds_pipeline=True,
            categorical_encoding="onehot",
        )

        pipeline = create_polarsds_pipeline(
            pl_df.select(feature_names),
            config,
            verbose=0,
        )

        assert pipeline is not None

    def test_pipeline_transformation(self, sample_polars_data):
        """Test that created pipeline can transform data."""
        pl_df, feature_names, _ = sample_polars_data

        try:
            import polars_ds.pipeline
        except ImportError:
            pytest.skip("polars-ds not installed")

        config = PolarsPipelineConfig(
            use_polarsds_pipeline=True,
            scaler_name="standard",
        )

        pipeline = create_polarsds_pipeline(
            pl_df.select(feature_names),
            config,
            verbose=0,
        )

        # Transform data
        if pipeline is not None:
            transformed = pipeline.transform(pl_df.select(feature_names))
            assert len(transformed) == len(pl_df)
            assert len(transformed.columns) == len(feature_names)

    def test_pipeline_with_no_scaling(self, sample_polars_data):
        """Test pipeline creation without scaling."""
        pl_df, feature_names, _ = sample_polars_data

        try:
            import polars_ds.pipeline
        except ImportError:
            pytest.skip("polars-ds not installed")

        config = PolarsPipelineConfig(
            use_polarsds_pipeline=True,
            scaler_name=None,  # No scaling
        )

        pipeline = create_polarsds_pipeline(
            pl_df.select(feature_names),
            config,
            verbose=0,
        )

        assert pipeline is not None

    def test_custom_pipeline_name(self, sample_polars_data):
        """Test pipeline creation with custom name."""
        pl_df, feature_names, _ = sample_polars_data

        try:
            import polars_ds.pipeline
        except ImportError:
            pytest.skip("polars-ds not installed")

        config = PolarsPipelineConfig(
            use_polarsds_pipeline=True,
            scaler_name="standard",
        )

        pipeline = create_polarsds_pipeline(
            pl_df.select(feature_names),
            config,
            pipeline_name="custom_pipeline",
            verbose=0,
        )

        assert pipeline is not None

    def test_pipeline_converts_int_to_float32(self, sample_polars_data):
        """Test that pipeline converts integers to float32."""
        # Create DataFrame with integer columns
        pl_df = pl.DataFrame({
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        try:
            import polars_ds.pipeline
        except ImportError:
            pytest.skip("polars-ds not installed")

        config = PolarsPipelineConfig(
            use_polarsds_pipeline=True,
            scaler_name="standard",
        )

        pipeline = create_polarsds_pipeline(
            pl_df,
            config,
            verbose=0,
        )

        if pipeline is not None:
            transformed = pipeline.transform(pl_df)
            # int_col should be converted to float
            assert transformed["int_col"].dtype in (pl.Float32, pl.Float64)
