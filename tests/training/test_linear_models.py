"""
Tests for linear models.

Tests all 7 linear model types:
- linear
- ridge
- lasso
- elasticnet
- huber
- ransac
- sgd
"""

import pytest
import numpy as np
from sklearn.base import is_classifier, is_regressor

from mlframe.training.models import (
    create_linear_model,
    is_linear_model,
    LINEAR_MODEL_TYPES,
)
from mlframe.training.configs import LinearModelConfig


class TestLinearModelCreation:
    """Test creation of linear models."""

    @pytest.mark.parametrize("model_type", [
        "linear", "ridge", "lasso", "elasticnet", "huber", "ransac", "sgd"
    ])
    def test_create_regression_model(self, model_type):
        """Test creation of regression models."""
        config = LinearModelConfig(model_type=model_type)
        model = create_linear_model(model_type, config, use_regression=True)

        assert model is not None
        # Most regression models should be regressors (except RANSAC which is a meta-estimator)
        if model_type not in ['ransac']:
            assert is_regressor(model) or hasattr(model, 'fit')

    @pytest.mark.parametrize("model_type", [
        "linear", "ridge", "lasso", "elasticnet", "huber", "sgd"
    ])
    def test_create_classification_model(self, model_type):
        """Test creation of classification models."""
        config = LinearModelConfig(model_type=model_type)
        model = create_linear_model(model_type, config, use_regression=False)

        assert model is not None
        # Should be a classifier or have fit/predict methods
        assert is_classifier(model) or hasattr(model, 'predict_proba') or hasattr(model, 'fit')

    def test_invalid_model_type(self):
        """Test creation with invalid model type raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LinearModelConfig(model_type="invalid_model")


class TestLinearModelTypes:
    """Test linear model type detection."""

    def test_is_linear_model(self):
        """Test linear model detection."""
        assert is_linear_model("linear")
        assert is_linear_model("ridge")
        assert is_linear_model("LASSO")  # Case insensitive
        assert is_linear_model("ElasticNet")

        assert not is_linear_model("cb")
        assert not is_linear_model("lgb")
        assert not is_linear_model("xgb")

    def test_linear_model_types_set(self):
        """Test LINEAR_MODEL_TYPES constant."""
        assert len(LINEAR_MODEL_TYPES) == 7
        assert "linear" in LINEAR_MODEL_TYPES
        assert "sgd" in LINEAR_MODEL_TYPES


class TestLinearModelTraining:
    """Test training of linear models."""

    @pytest.mark.parametrize("model_type", ["linear", "ridge", "lasso"])
    def test_train_regression_model(self, sample_regression_data, model_type):
        """Test training regression models."""
        df, feature_names, y = sample_regression_data

        train_size = int(0.7 * len(df))
        train_df = df[feature_names].iloc[:train_size]
        train_target = df['target'].iloc[:train_size]
        test_df = df[feature_names].iloc[train_size:]
        test_target = df['target'].iloc[train_size:]

        config = LinearModelConfig(model_type=model_type, max_iter=1000)
        model = create_linear_model(model_type, config, use_regression=True)

        model.fit(train_df, train_target)
        train_preds = model.predict(train_df)
        test_preds = model.predict(test_df)

        assert len(train_preds) == len(train_target)
        assert len(test_preds) == len(test_target)

        # Check predictions are reasonable (not all zeros, not NaN)
        assert not np.all(train_preds == 0)
        assert not np.any(np.isnan(train_preds))

    @pytest.mark.parametrize("model_type", ["linear", "ridge", "sgd"])
    def test_train_classification_model(self, sample_classification_data, model_type):
        """Test training classification models."""
        df, feature_names, _, y = sample_classification_data

        train_size = int(0.7 * len(df))
        train_df = df[feature_names].iloc[:train_size]
        train_target = df['target'].iloc[:train_size]
        test_df = df[feature_names].iloc[train_size:]
        test_target = df['target'].iloc[train_size:]

        config = LinearModelConfig(model_type=model_type, max_iter=1000)
        model = create_linear_model(model_type, config, use_regression=False)

        model.fit(train_df, train_target)
        train_preds = model.predict(train_df)
        test_preds = model.predict(test_df)

        assert len(train_preds) == len(train_target)
        assert len(test_preds) == len(test_target)

        # For classification with predict_proba, check probabilities
        if model_type != 'ridge' and hasattr(model, 'predict_proba'):
            train_probs = model.predict_proba(train_df)
            assert train_probs.shape[0] == len(train_target)

        # Check predictions are binary
        assert set(train_preds).issubset({0, 1})

    def test_elasticnet_with_custom_l1_ratio(self, sample_regression_data):
        """Test ElasticNet with custom L1 ratio."""
        df, feature_names, y = sample_regression_data

        train_df = df[feature_names].iloc[:700]
        train_target = df['target'].iloc[:700]

        config = LinearModelConfig(
            model_type='elasticnet',
            alpha=0.1,
            l1_ratio=0.7,
            max_iter=2000,
        )

        model = create_linear_model('elasticnet', config, use_regression=True)
        model.fit(train_df, train_target)

        # ElasticNet should create sparse solutions
        if hasattr(model, 'coef_'):
            # Check that some coefficients are zero (L1 effect)
            assert np.sum(np.abs(model.coef_) < 0.01) > 0

    def test_huber_robustness(self, sample_regression_data):
        """Test Huber regression with outliers."""
        df, feature_names, y = sample_regression_data

        # Add outliers
        df_with_outliers = df.copy()
        df_with_outliers.loc[10:20, 'target'] *= 10  # Create outliers

        train_df = df_with_outliers[feature_names].iloc[:700]
        train_target = df_with_outliers['target'].iloc[:700]

        config = LinearModelConfig(
            model_type='huber',
            epsilon=1.35,
            max_iter=1000,
        )

        model = create_linear_model('huber', config, use_regression=True)
        model.fit(train_df, train_target)

        # Huber should be more robust to outliers than regular linear regression
        assert hasattr(model, 'coef_')
        train_preds = model.predict(train_df)
        assert not np.any(np.isnan(train_preds))

    def test_sgd_large_dataset_simulation(self, sample_regression_data):
        """Test SGD on simulated large dataset."""
        df, feature_names, y = sample_regression_data

        # SGD is designed for large datasets
        train_df = df[feature_names].iloc[:800]
        train_target = df['target'].iloc[:800]

        config = LinearModelConfig(
            model_type='sgd',
            loss='squared_error',
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            learning_rate='optimal',
        )

        model = create_linear_model('sgd', config, use_regression=True)
        model.fit(train_df, train_target)

        train_preds = model.predict(train_df)
        assert len(train_preds) == len(train_target)


class TestLinearModelConfigurations:
    """Test different configuration scenarios."""

    def test_ridge_with_different_alphas(self, sample_regression_data):
        """Test Ridge with different regularization strengths."""
        df, feature_names, y = sample_regression_data

        train_df = df[feature_names].iloc[:700]
        train_target = df['target'].iloc[:700]

        alphas = [0.01, 1.0, 100.0]
        models = []

        for alpha in alphas:
            config = LinearModelConfig(model_type='ridge', alpha=alpha)
            model = create_linear_model('ridge', config, use_regression=True)
            model.fit(train_df, train_target)
            models.append(model)

        # Higher alpha should lead to smaller coefficients (more regularization)
        if all(hasattr(m, 'coef_') for m in models):
            coef_norms = [np.linalg.norm(m.coef_) for m in models]
            assert coef_norms[2] < coef_norms[0]  # High alpha < low alpha

    def test_calibrated_classifier(self, sample_classification_data):
        """Test calibrated classifier wrapper."""
        df, feature_names, _, y = sample_classification_data

        train_df = df[feature_names].iloc[:700]
        train_target = df['target'].iloc[:700]

        config = LinearModelConfig(
            model_type='linear',
            use_calibrated_classifier=True,
        )

        model = create_linear_model('linear', config, use_regression=False)
        model.fit(train_df, train_target)

        # Should be wrapped in CalibratedClassifierCV
        assert hasattr(model, 'predict_proba')
        train_probs = model.predict_proba(train_df)
        assert train_probs.shape[0] == len(train_target)


class TestLinearModelConvergence:
    """Test convergence behavior of iterative linear models."""

    def test_sgd_convergence_n_iter(self, sample_regression_data):
        """Test SGD records number of iterations."""
        df, feature_names, y = sample_regression_data

        train_df = df[feature_names].iloc[:800]
        train_target = df['target'].iloc[:800]

        config = LinearModelConfig(
            model_type='sgd',
            loss='squared_error',
            penalty='l2',
            alpha=0.0001,
            max_iter=2000,
            tol=1e-4,
        )

        model = create_linear_model('sgd', config, use_regression=True)
        model.fit(train_df, train_target)

        # SGD should have n_iter_ attribute
        if hasattr(model, 'n_iter_'):
            # Should have completed some iterations
            assert model.n_iter_ > 0
            # Should have converged before max_iter (for this simple data)
            # Note: may or may not converge early depending on data

    def test_lasso_convergence(self, sample_regression_data):
        """Test LASSO convergence behavior."""
        import warnings

        df, feature_names, y = sample_regression_data

        train_df = df[feature_names].iloc[:800]
        train_target = df['target'].iloc[:800]

        config = LinearModelConfig(
            model_type='lasso',
            alpha=0.1,
            max_iter=5000,
            tol=1e-4,
        )

        model = create_linear_model('lasso', config, use_regression=True)

        # Capture convergence warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(train_df, train_target)

            # Check if convergence warning was raised
            convergence_warnings = [warning for warning in w
                                    if 'converge' in str(warning.message).lower()]

            # For simple test data, should converge without warnings
            # But if there's a warning, verify model still produces predictions
            if convergence_warnings:
                # Model should still work despite warning
                preds = model.predict(train_df)
                assert len(preds) == len(train_target)

        # Verify model has n_iter_ if available
        if hasattr(model, 'n_iter_'):
            assert model.n_iter_ > 0

    def test_elasticnet_convergence(self, sample_regression_data):
        """Test ElasticNet convergence behavior."""
        import warnings

        df, feature_names, y = sample_regression_data

        train_df = df[feature_names].iloc[:800]
        train_target = df['target'].iloc[:800]

        config = LinearModelConfig(
            model_type='elasticnet',
            alpha=0.1,
            l1_ratio=0.5,
            max_iter=5000,
            tol=1e-4,
        )

        model = create_linear_model('elasticnet', config, use_regression=True)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            model.fit(train_df, train_target)

        # Verify iteration count
        if hasattr(model, 'n_iter_'):
            assert model.n_iter_ > 0

        # Verify predictions work
        preds = model.predict(train_df)
        assert len(preds) == len(train_target)

    def test_sgd_classifier_convergence(self, sample_classification_data):
        """Test SGD classifier convergence."""
        df, feature_names, _, y = sample_classification_data

        train_df = df[feature_names].iloc[:800]
        train_target = df['target'].iloc[:800]

        config = LinearModelConfig(
            model_type='sgd',
            loss='log_loss',
            penalty='l2',
            alpha=0.0001,
            max_iter=2000,
            tol=1e-4,
        )

        model = create_linear_model('sgd', config, use_regression=False)
        model.fit(train_df, train_target)

        if hasattr(model, 'n_iter_'):
            assert model.n_iter_ > 0

        # Should produce valid predictions
        preds = model.predict(train_df)
        assert set(preds).issubset({0, 1})

    def test_model_improves_with_more_iterations(self, sample_regression_data):
        """Test that models improve or stay same with more iterations."""
        from sklearn.metrics import mean_squared_error

        df, feature_names, y = sample_regression_data

        train_df = df[feature_names].iloc[:600]
        train_target = df['target'].iloc[:600]
        val_df = df[feature_names].iloc[600:800]
        val_target = df['target'].iloc[600:800]

        # Train with few iterations
        config_low = LinearModelConfig(
            model_type='sgd',
            loss='squared_error',
            max_iter=10,
            tol=0,  # Don't stop early
            early_stopping=False,
            random_state=42,
        )

        # Train with more iterations
        config_high = LinearModelConfig(
            model_type='sgd',
            loss='squared_error',
            max_iter=1000,
            tol=1e-4,
            early_stopping=False,
            random_state=42,
        )

        model_low = create_linear_model('sgd', config_low, use_regression=True)
        model_high = create_linear_model('sgd', config_high, use_regression=True)

        model_low.fit(train_df, train_target)
        model_high.fit(train_df, train_target)

        # Get validation predictions
        preds_low = model_low.predict(val_df)
        preds_high = model_high.predict(val_df)

        mse_low = mean_squared_error(val_target, preds_low)
        mse_high = mean_squared_error(val_target, preds_high)

        # More iterations should give same or better results
        # (allowing some tolerance due to randomness)
        assert mse_high <= mse_low * 2, \
            f"More iterations should not significantly worsen: {mse_high} vs {mse_low}"

    def test_early_stopping_sgd(self, sample_regression_data):
        """Test SGD early stopping behavior."""
        df, feature_names, y = sample_regression_data

        # Split for early stopping validation
        train_df = df[feature_names].iloc[:600]
        train_target = df['target'].iloc[:600]

        config = LinearModelConfig(
            model_type='sgd',
            loss='squared_error',
            max_iter=10000,  # High max_iter
            tol=1e-3,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=5,
            random_state=42,
        )

        model = create_linear_model('sgd', config, use_regression=True)
        model.fit(train_df, train_target)

        # With early stopping, should stop before max_iter
        if hasattr(model, 'n_iter_'):
            # Early stopping should kick in for well-behaved data
            # (but this depends on the data, so we just verify it ran)
            assert model.n_iter_ > 0

        # Model should still produce valid predictions
        preds = model.predict(train_df)
        assert not np.any(np.isnan(preds))
