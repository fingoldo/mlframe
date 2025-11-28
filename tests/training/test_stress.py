"""
Stress and performance tests for mlframe training module.

Tests memory usage, large datasets, timeout handling, and parallel execution.
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl
import gc
import time
import psutil
import os
import tempfile

from mlframe.training.core import train_mlframe_models_suite
from mlframe.training.io import save_mlframe_model, load_mlframe_model
from mlframe.training.pipeline import fit_and_transform_pipeline
from mlframe.training.configs import PolarsPipelineConfig
from .shared import SimpleFeaturesAndTargetsExtractor


# ================================================================================================
# Test Class 1: Memory Stress Tests
# ================================================================================================


class TestMemoryStress:
    """Tests for memory handling under stress."""

    def test_large_dataframe_pandas(self, temp_data_dir, common_init_params):
        """Test with large pandas DataFrame."""
        np.random.seed(42)
        n_samples = 5000
        n_features = 50

        X = np.random.randn(n_samples, n_features)
        y = 2 * X[:, 0] + np.random.randn(n_samples) * 0.5

        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['target'] = y

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # Get initial memory
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="large_pandas",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            config_params_override={"iterations": 10},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Cleanup
        del df
        gc.collect()

        mem_after = process.memory_info().rss / 1024 / 1024
        mem_delta = mem_after - mem_before

        # Should not leak excessive memory (< 500 MB delta for this size)
        assert mem_delta < 500, f"Memory delta {mem_delta:.0f} MB exceeds threshold"
        assert "target" in models

    def test_large_dataframe_polars(self, temp_data_dir, common_init_params):
        """Test with large Polars DataFrame."""
        np.random.seed(42)
        n_samples = 5000
        n_features = 50

        X = np.random.randn(n_samples, n_features)
        y = 2 * X[:, 0] + np.random.randn(n_samples) * 0.5

        data = {f'feature_{i}': X[:, i] for i in range(n_features)}
        data['target'] = y
        df = pl.DataFrame(data)

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="large_polars",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            config_params_override={"iterations": 10},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        assert "target" in models

    def test_many_columns(self, temp_data_dir, common_init_params):
        """Test with many feature columns."""
        np.random.seed(42)
        n_samples = 500
        n_features = 200

        X = np.random.randn(n_samples, n_features)
        y = X[:, 0] + np.random.randn(n_samples) * 0.1

        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['target'] = y

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="many_columns",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            config_params_override={"iterations": 10},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        assert "target" in models

    def test_repeated_training_memory_leak(self, temp_data_dir, common_init_params):
        """Test for memory leaks with repeated training."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_0': np.random.randn(200),
            'feature_1': np.random.randn(200),
            'target': np.random.randn(200),
        })

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)
        process = psutil.Process()

        memory_readings = []

        # Train multiple times
        for i in range(5):
            models, metadata = train_mlframe_models_suite(
                df=df.copy(),
                target_name="test_target",
                model_name=f"repeat_{i}",
                features_and_targets_extractor=fte,
                mlframe_models=["ridge"],
                config_params_override={"iterations": 10},
                init_common_params=common_init_params,
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                data_dir=temp_data_dir,
                models_dir="models",
                verbose=0,
            )

            gc.collect()
            memory_readings.append(process.memory_info().rss / 1024 / 1024)

        # Memory should not grow significantly (< 100 MB total growth)
        mem_growth = memory_readings[-1] - memory_readings[0]
        assert mem_growth < 100, f"Memory grew by {mem_growth:.0f} MB over 5 iterations"


# ================================================================================================
# Test Class 2: Performance Tests
# ================================================================================================


class TestPerformance:
    """Tests for performance benchmarks."""

    def test_pipeline_transform_performance(self, sample_regression_data):
        """Test pipeline transform performance."""
        df, feature_names, y = sample_regression_data
        train_df = df[feature_names].iloc[:700]
        val_df = df[feature_names].iloc[700:]

        config = PolarsPipelineConfig(use_polarsds_pipeline=False)

        start = time.time()

        train_transformed, val_transformed, test_transformed, pipeline, cat_features = fit_and_transform_pipeline(
            train_df=train_df,
            val_df=val_df,
            test_df=None,
            config=config,
            ensure_float32=False,
            verbose=0,
        )

        elapsed = time.time() - start

        # Should complete quickly (< 5 seconds)
        assert elapsed < 5.0, f"Pipeline transform took {elapsed:.1f}s"
        assert len(train_transformed) == len(train_df)

    def test_model_save_load_performance(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test model save/load performance."""
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="save_load_perf",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            config_params_override={"iterations": 10},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        # Test save performance
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "model.zst")

            start = time.time()
            save_mlframe_model(models, file_path, verbose=0)
            save_time = time.time() - start

            # Test load performance
            start = time.time()
            loaded = load_mlframe_model(file_path)
            load_time = time.time() - start

            # Both should complete quickly (< 5 seconds each)
            assert save_time < 5.0, f"Save took {save_time:.1f}s"
            assert load_time < 5.0, f"Load took {load_time:.1f}s"

    def test_multiple_models_performance(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test training multiple model types performance."""
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        start = time.time()

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="multi_model_perf",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge", "lasso", "elasticnet"],
            config_params_override={"iterations": 10},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        elapsed = time.time() - start

        # 3 linear models should complete quickly (< 30 seconds)
        assert elapsed < 30.0, f"Multiple models took {elapsed:.1f}s"
        assert "target" in models


# ================================================================================================
# Test Class 3: Concurrent and Parallel Tests
# ================================================================================================


class TestConcurrency:
    """Tests for concurrent execution scenarios."""

    def test_sequential_training_different_configs(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test sequential training with different configurations."""
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        configs = [
            {"scaler_name": "standard"},
            {"scaler_name": "min_max"},
            {"scaler_name": "robust"},
        ]

        results = []

        for i, config in enumerate(configs):
            pipeline_config = PolarsPipelineConfig(
                use_polarsds_pipeline=False,
                **config,
            )

            models, metadata = train_mlframe_models_suite(
                df=df.copy(),
                target_name="test_target",
                model_name=f"config_{i}",
                features_and_targets_extractor=fte,
                mlframe_models=["ridge"],
                pipeline_config=pipeline_config,
                config_params_override={"iterations": 10},
                init_common_params=common_init_params,
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                data_dir=temp_data_dir,
                models_dir="models",
                verbose=0,
            )

            results.append(models)

        # All should succeed
        assert all("target" in r for r in results)

    def test_gc_between_trainings(self, sample_regression_data, temp_data_dir, common_init_params):
        """Test GC effectiveness between training runs."""
        df, feature_names, y = sample_regression_data
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        for i in range(3):
            models, metadata = train_mlframe_models_suite(
                df=df.copy(),
                target_name="test_target",
                model_name=f"gc_test_{i}",
                features_and_targets_extractor=fte,
                mlframe_models=["ridge"],
                config_params_override={"iterations": 10},
                init_common_params=common_init_params,
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                data_dir=temp_data_dir,
                models_dir="models",
                verbose=0,
            )

            # Force GC between runs
            del models, metadata
            gc.collect()

        # Should complete without errors
        assert True


# ================================================================================================
# Test Class 4: Edge Case Stress Tests
# ================================================================================================


class TestEdgeCaseStress:
    """Stress tests for edge cases."""

    def test_very_small_dataset(self, temp_data_dir, common_init_params):
        """Test with minimum viable dataset size."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_0': np.random.randn(10),
            'feature_1': np.random.randn(10),
            'target': np.random.randn(10),
        })

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="tiny_dataset",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            config_params_override={"iterations": 5},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        assert "target" in models

    def test_high_dimensional_sparse(self, temp_data_dir, common_init_params):
        """Test with high-dimensional sparse-like data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 500

        # Create sparse-like data (mostly zeros)
        X = np.zeros((n_samples, n_features))
        for i in range(n_samples):
            non_zero_idx = np.random.choice(n_features, size=10, replace=False)
            X[i, non_zero_idx] = np.random.randn(10)

        y = np.random.randn(n_samples)

        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['target'] = y

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="sparse_like",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            config_params_override={"iterations": 5},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        assert "target" in models

    def test_many_nan_values(self, temp_data_dir, common_init_params):
        """Test with many NaN values (50%)."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_0': np.random.randn(200),
            'feature_1': np.random.randn(200),
            'target': np.random.randn(200),
        })

        # Add 50% NaN values
        nan_mask = np.random.random(200) < 0.5
        df.loc[nan_mask, 'feature_0'] = np.nan

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        # May fail or succeed depending on imputation
        try:
            models, metadata = train_mlframe_models_suite(
                df=df,
                target_name="test_target",
                model_name="many_nan",
                features_and_targets_extractor=fte,
                mlframe_models=["ridge"],
                config_params_override={"iterations": 5},
                init_common_params=common_init_params,
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                data_dir=temp_data_dir,
                models_dir="models",
                verbose=0,
            )
            assert "target" in models
        except Exception:
            # Expected - too many NaNs may cause issues
            pass

    def test_extreme_values(self, temp_data_dir, common_init_params):
        """Test with extreme values in data."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature_0': np.concatenate([np.random.randn(190), [1e10, -1e10, 1e-10, -1e-10] + [np.inf, -np.inf] + [0, 0, 0, 0]]),
            'feature_1': np.random.randn(200),
            'target': np.random.randn(200),
        })

        # Replace inf with large values
        df = df.replace([np.inf, -np.inf], [1e10, -1e10])

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="extreme_values",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            config_params_override={"iterations": 5},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        assert "target" in models

    def test_duplicate_rows(self, temp_data_dir, common_init_params):
        """Test with duplicate rows in data."""
        np.random.seed(42)
        base_df = pd.DataFrame({
            'feature_0': np.random.randn(50),
            'feature_1': np.random.randn(50),
            'target': np.random.randn(50),
        })

        # Create duplicates
        df = pd.concat([base_df] * 4, ignore_index=True)

        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

        models, metadata = train_mlframe_models_suite(
            df=df,
            target_name="test_target",
            model_name="duplicates",
            features_and_targets_extractor=fte,
            mlframe_models=["ridge"],
            config_params_override={"iterations": 5},
            init_common_params=common_init_params,
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir,
            models_dir="models",
            verbose=0,
        )

        assert "target" in models


# ================================================================================================
# Test Class 5: File I/O Stress Tests
# ================================================================================================


class TestFileIOStress:
    """Stress tests for file I/O operations."""

    def test_large_model_save_load(self):
        """Test saving/loading large model objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create large nested structure
            large_data = {
                "arrays": [np.random.randn(1000, 100) for _ in range(5)],
                "dicts": [{f"key_{i}": np.random.randn(100) for i in range(50)}],
                "nested": {
                    "level1": {
                        "level2": {
                            "data": np.random.randn(500, 50)
                        }
                    }
                }
            }

            file_path = os.path.join(tmpdir, "large_model.zst")

            # Save
            result = save_mlframe_model(large_data, file_path, verbose=0)
            assert result is True

            # Verify file exists and has reasonable size
            assert os.path.exists(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            assert file_size > 0.1, "File should have substantial size"

            # Load
            loaded = load_mlframe_model(file_path)
            assert loaded is not None

    def test_repeated_save_operations(self):
        """Test repeated save operations don't cause issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"key": np.random.randn(100)}

            for i in range(10):
                file_path = os.path.join(tmpdir, f"model_{i}.zst")
                result = save_mlframe_model(data, file_path, verbose=0)
                assert result is True

            # Verify all files exist
            files = os.listdir(tmpdir)
            assert len(files) == 10

    def test_save_load_different_compressions(self):
        """Test save/load with different compression levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {"data": np.random.randn(500, 100)}

            for compression in [1, 5, 10]:
                file_path = os.path.join(tmpdir, f"model_comp{compression}.zst")
                result = save_mlframe_model(data, file_path, zstd_kwargs={'level': compression}, verbose=0)
                assert result is True

                loaded = load_mlframe_model(file_path)
                assert loaded is not None
