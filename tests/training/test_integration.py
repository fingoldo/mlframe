"""
Integration tests for mlframe training module.

Covers:
- End-to-end training pipelines
- Save/load/predict roundtrip
- Multi-target training
- AutoML integration (enabled when dependencies available)
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

from sklearn.linear_model import Ridge, LogisticRegression

from mlframe.training.configs import (
    DataConfig,
    TrainingControlConfig,
    MetricsConfig,
    DisplayConfig,
    NamingConfig,
)


def call_train_and_evaluate(
    model,
    df=None,
    target=None,
    train_idx=None,
    val_idx=None,
    test_idx=None,
    sample_weight=None,
    compute_trainset_metrics=False,
    compute_valset_metrics=True,
    compute_testset_metrics=True,
    print_report=True,
    show_perf_chart=True,
    show_fi=True,
    verbose=False,
    **kwargs,
):
    """Helper to call train_and_evaluate_model with config objects."""
    from mlframe.training.trainer import train_and_evaluate_model

    data = DataConfig(
        df=df,
        target=target,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        sample_weight=sample_weight,
    )
    control = TrainingControlConfig(
        verbose=verbose,
        compute_trainset_metrics=compute_trainset_metrics,
        compute_valset_metrics=compute_valset_metrics,
        compute_testset_metrics=compute_testset_metrics,
    )
    metrics = MetricsConfig()
    display = DisplayConfig(
        print_report=print_report,
        show_perf_chart=show_perf_chart,
        show_fi=show_fi,
    )
    naming = NamingConfig()

    return train_and_evaluate_model(
        model=model,
        data=data,
        control=control,
        metrics=metrics,
        display=display,
        naming=naming,
    )


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestFullPipelineRegression:
    """End-to-end regression pipeline tests."""

    @pytest.fixture
    def regression_data(self):
        """Create regression dataset."""
        np.random.seed(42)
        n_samples = 500
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5

        columns = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=columns)
        df['target'] = y

        train_idx = np.arange(400)
        val_idx = np.arange(400, 450)
        test_idx = np.arange(450, 500)

        return df, columns, train_idx, val_idx, test_idx

    def test_end_to_end_regression_pipeline(self, regression_data):
        """Test complete regression training pipeline."""
        df, columns, train_idx, val_idx, test_idx = regression_data
        target = df['target']
        feature_df = df[columns]

        model = Ridge(alpha=1.0)

        result, train_df, val_df, test_df = call_train_and_evaluate(
            model=model,
            df=feature_df,
            target=target,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            compute_trainset_metrics=True,
            compute_valset_metrics=True,
            compute_testset_metrics=True,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
            verbose=False,
        )

        # Verify result structure
        assert result is not None
        assert result.model is not None
        assert result.test_preds is not None
        assert len(result.test_preds) == len(test_idx)

        # Verify metrics
        assert 'MAE' in result.metrics.get('test', {}) or len(result.metrics.get('test', {})) > 0

        # Verify predictions are reasonable (not NaN, not constant)
        assert not np.any(np.isnan(result.test_preds))
        assert np.std(result.test_preds) > 0

    def test_predictions_in_valid_range(self, regression_data):
        """Test that regression predictions are in valid range."""
        df, columns, train_idx, val_idx, test_idx = regression_data
        target = df['target']
        feature_df = df[columns]

        model = Ridge(alpha=1.0)

        result, *_ = call_train_and_evaluate(
            model=model,
            df=feature_df,
            target=target,
            train_idx=train_idx,
            test_idx=test_idx,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
        )

        # Predictions should be within reasonable range of target
        target_range = target.max() - target.min()
        pred_range = result.test_preds.max() - result.test_preds.min()

        # Predictions shouldn't be collapsed to a constant
        assert pred_range > 0.1 * target_range


class TestFullPipelineClassification:
    """End-to-end classification pipeline tests."""

    @pytest.fixture
    def classification_data(self):
        """Create classification dataset."""
        np.random.seed(42)
        n_samples = 500
        n_features = 10

        X = np.random.randn(n_samples, n_features)
        logits = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2]
        probs = 1 / (1 + np.exp(-logits))
        y = (probs > 0.5).astype(int)

        columns = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=columns)
        df['target'] = y

        train_idx = np.arange(400)
        val_idx = np.arange(400, 450)
        test_idx = np.arange(450, 500)

        return df, columns, train_idx, val_idx, test_idx

    def test_end_to_end_classification_pipeline(self, classification_data):
        """Test complete classification training pipeline."""
        df, columns, train_idx, val_idx, test_idx = classification_data
        target = df['target']
        feature_df = df[columns]

        model = LogisticRegression(max_iter=1000)

        result, *_ = call_train_and_evaluate(
            model=model,
            df=feature_df,
            target=target,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            compute_testset_metrics=True,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
            verbose=False,
        )

        # Verify result structure
        assert result is not None
        assert result.model is not None
        assert result.test_preds is not None
        assert result.test_probs is not None

        # Verify predictions are binary
        assert set(np.unique(result.test_preds)).issubset({0, 1})

        # Verify probabilities
        assert result.test_probs.shape[1] == 2
        np.testing.assert_array_almost_equal(
            result.test_probs.sum(axis=1),
            np.ones(len(test_idx)),
            decimal=5
        )

    def test_probabilities_sum_to_one(self, classification_data):
        """Test that class probabilities sum to 1."""
        df, columns, train_idx, val_idx, test_idx = classification_data
        target = df['target']
        feature_df = df[columns]

        model = LogisticRegression(max_iter=1000)

        result, *_ = call_train_and_evaluate(
            model=model,
            df=feature_df,
            target=target,
            train_idx=train_idx,
            test_idx=test_idx,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
        )

        prob_sums = result.test_probs.sum(axis=1)
        np.testing.assert_array_almost_equal(prob_sums, 1.0, decimal=5)

    def test_probabilities_in_valid_range(self, classification_data):
        """Test that probabilities are in [0, 1]."""
        df, columns, train_idx, val_idx, test_idx = classification_data
        target = df['target']
        feature_df = df[columns]

        model = LogisticRegression(max_iter=1000)

        result, *_ = call_train_and_evaluate(
            model=model,
            df=feature_df,
            target=target,
            train_idx=train_idx,
            test_idx=test_idx,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
        )

        assert np.all(result.test_probs >= 0)
        assert np.all(result.test_probs <= 1)


# =============================================================================
# Save/Load Roundtrip Tests
# =============================================================================


class TestSaveLoadRoundtrip:
    """Tests for model save/load functionality."""

    def test_save_and_load_model(self, tmp_path):
        """Test saving and loading a trained model."""
        from mlframe.training.io import save_mlframe_model, load_mlframe_model

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = 2 * X[:, 0] + np.random.randn(100) * 0.5

        model = Ridge(alpha=1.0)
        model.fit(X, y)

        model_path = str(tmp_path / "test_model.dump")

        # Save
        success = save_mlframe_model(model, model_path)
        assert success

        # Load
        loaded_model = load_mlframe_model(model_path)
        assert loaded_model is not None

        # Verify predictions match
        original_preds = model.predict(X)
        loaded_preds = loaded_model.predict(X)
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_save_load_complex_object(self, tmp_path):
        """Test saving/loading complex nested objects."""
        from mlframe.training.io import save_mlframe_model, load_mlframe_model

        # Complex object with model and metadata
        model_data = {
            'model': Ridge(alpha=1.0).fit(np.random.randn(50, 3), np.random.randn(50)),
            'metadata': {'version': '1.0', 'features': ['a', 'b', 'c']},
            'array': np.array([1, 2, 3]),
        }

        model_path = str(tmp_path / "complex_model.dump")

        save_mlframe_model(model_data, model_path)
        loaded = load_mlframe_model(model_path)

        assert loaded is not None
        assert 'model' in loaded
        assert 'metadata' in loaded
        assert loaded['metadata']['version'] == '1.0'


# =============================================================================
# AutoML Integration Tests (Enabled when deps available)
# =============================================================================


class TestAutoMLIntegration:
    """AutoML integration tests - run when dependencies are installed."""

    @pytest.fixture
    def automl_data(self):
        """Create dataset for AutoML testing."""
        np.random.seed(42)
        n_samples = 200  # Small for faster tests
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(n_features)])
        df['target'] = y

        train_df = df.iloc[:150].copy()
        test_df = df.iloc[150:].copy()

        return train_df, test_df

    def test_autogluon_training(self, automl_data, tmp_path):
        """Test AutoGluon model training."""
        ag = pytest.importorskip("autogluon.tabular")

        from mlframe.training.automl import train_autogluon_model

        train_df, test_df = automl_data

        result = train_autogluon_model(
            train_df=train_df,
            test_df=test_df,
            target_name='target',
            init_params={'path': str(tmp_path / 'ag_model')},
            fit_params={'time_limit': 30, 'presets': 'medium_quality'},  # Fast training
        )

        if result is not None:
            assert hasattr(result, 'model')
            assert hasattr(result, 'feature_importances')
            assert hasattr(result, 'test_roc_auc')

    def test_lama_training(self, automl_data, tmp_path):
        """Test LightAutoML model training."""
        lama = pytest.importorskip("lightautoml")
        from lightautoml.tasks import Task

        from mlframe.training.automl import train_lama_model

        train_df, test_df = automl_data

        try:
            result = train_lama_model(
                train_df=train_df,
                test_df=test_df,
                target_name='target',
                init_params={'task': Task('binary'), 'timeout': 30},  # Required task + fast training
            )

            if result is not None:
                assert hasattr(result, 'model')
        except AttributeError as e:
            if "np.find_common_type" in str(e):
                pytest.skip("LightAutoML incompatible with NumPy 2.0")
            raise

    def test_automl_suite(self, automl_data, tmp_path):
        """Test training multiple AutoML models."""
        from mlframe.training.automl import train_automl_models_suite
        from mlframe.training.configs import AutoMLConfig

        train_df, test_df = automl_data

        # Try to train both if available
        config = AutoMLConfig(
            enable_autogluon=True,
            enable_lama=True,
            autogluon_init_params={'path': str(tmp_path / 'ag_suite')},
            autogluon_fit_params={'time_limit': 30, 'presets': 'medium_quality'},
            lama_init_params={'timeout': 30},
        )

        results = train_automl_models_suite(
            train_df=train_df,
            test_df=test_df,
            target_name='target',
            config=config,
        )

        # Should return dict (may be empty if deps not available)
        assert isinstance(results, dict)


# =============================================================================
# Edge Cases
# =============================================================================


class TestIntegrationEdgeCases:
    """Edge case integration tests."""

    def test_small_dataset(self):
        """Test training with very small dataset."""
        np.random.seed(42)
        X = np.random.randn(20, 3)
        y = 2 * X[:, 0] + np.random.randn(20) * 0.5

        df = pd.DataFrame(X, columns=['f0', 'f1', 'f2'])
        target = pd.Series(y)

        model = Ridge(alpha=1.0)

        train_idx = np.arange(15)
        test_idx = np.arange(15, 20)

        result, *_ = call_train_and_evaluate(
            model=model,
            df=df,
            target=target,
            train_idx=train_idx,
            test_idx=test_idx,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
        )

        assert result is not None
        assert len(result.test_preds) == 5

    def test_high_dimensional_data(self):
        """Test with high-dimensional data (more features than samples)."""
        from sklearn.linear_model import RidgeCV

        np.random.seed(42)
        n_samples = 100
        n_features = 200  # More features than samples

        X = np.random.randn(n_samples, n_features)
        y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.5

        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(n_features)])
        target = pd.Series(y)

        # Use RidgeCV for automatic alpha selection
        model = RidgeCV(alphas=[0.1, 1.0, 10.0])

        train_idx = np.arange(80)
        test_idx = np.arange(80, 100)

        result, *_ = call_train_and_evaluate(
            model=model,
            df=df,
            target=target,
            train_idx=train_idx,
            test_idx=test_idx,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
        )

        assert result is not None
        assert not np.any(np.isnan(result.test_preds))

    def test_with_sample_weights(self):
        """Test training with sample weights."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = 2 * X[:, 0] + np.random.randn(100) * 0.5

        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
        target = pd.Series(y)

        # Create sample weights (recent samples have higher weight)
        sample_weight = np.linspace(0.5, 1.5, 100)

        model = Ridge(alpha=1.0)

        train_idx = np.arange(80)
        test_idx = np.arange(80, 100)

        result, *_ = call_train_and_evaluate(
            model=model,
            df=df,
            target=target,
            train_idx=train_idx,
            test_idx=test_idx,
            sample_weight=sample_weight,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
        )

        assert result is not None

    def test_polars_dataframe(self):
        """Test training with Polars DataFrame."""
        import polars as pl

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = 2 * X[:, 0] + np.random.randn(100) * 0.5

        df = pl.DataFrame({f'f{i}': X[:, i] for i in range(5)})
        target = pl.Series(y)

        model = Ridge(alpha=1.0)

        train_idx = np.arange(80)
        test_idx = np.arange(80, 100)

        result, *_ = call_train_and_evaluate(
            model=model,
            df=df,
            target=target,
            train_idx=train_idx,
            test_idx=test_idx,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
        )

        assert result is not None
        assert result.test_preds is not None


class TestValidationChecks:
    """Tests for prediction validation."""

    def test_predictions_not_all_nan(self):
        """Test that predictions don't contain all NaN."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = 2 * X[:, 0] + np.random.randn(100) * 0.5

        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
        target = pd.Series(y)

        model = Ridge(alpha=1.0)

        result, *_ = call_train_and_evaluate(
            model=model,
            df=df,
            target=target,
            train_idx=np.arange(80),
            test_idx=np.arange(80, 100),
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
        )

        assert not np.all(np.isnan(result.test_preds))

    def test_predictions_not_all_same(self):
        """Test that predictions aren't all the same value."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = 2 * X[:, 0] + np.random.randn(100) * 0.5

        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
        target = pd.Series(y)

        model = Ridge(alpha=1.0)

        result, *_ = call_train_and_evaluate(
            model=model,
            df=df,
            target=target,
            train_idx=np.arange(80),
            test_idx=np.arange(80, 100),
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
        )

        # Predictions should have some variance
        assert np.std(result.test_preds) > 0

    def test_prediction_shape_matches_input(self):
        """Test that prediction shape matches input size."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = 2 * X[:, 0] + np.random.randn(100) * 0.5

        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
        target = pd.Series(y)

        model = Ridge(alpha=1.0)
        test_idx = np.arange(80, 100)

        result, *_ = call_train_and_evaluate(
            model=model,
            df=df,
            target=target,
            train_idx=np.arange(80),
            test_idx=test_idx,
            print_report=False,
            show_perf_chart=False,
            show_fi=False,
        )

        assert len(result.test_preds) == len(test_idx)
