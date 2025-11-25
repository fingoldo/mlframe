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
        """Test creation with invalid model type."""
        config = LinearModelConfig(model_type="invalid_model")

        with pytest.raises(ValueError):
            create_linear_model("invalid_model", config, use_regression=True)


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
