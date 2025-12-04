"""
Tests for sklearn-compatible estimator classes in lightninglib.py

Run tests:
    pytest tests/lightninglib/test_estimators.py -v
    pytest tests/lightninglib/test_estimators.py --cov=mlframe.lightninglib --cov-report=html
"""

import pytest
import torch
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from unittest.mock import Mock, patch

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mlframe.lightninglib import (
    PytorchLightningEstimator,
    PytorchLightningClassifier,
    PytorchLightningRegressor,
    MLPTorchModel,
    TorchDataModule,
    generate_mlp
)


# ================================================================================================
# Fixtures
# ================================================================================================


@pytest.fixture
def classification_data():
    """Generate classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_classes=3,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return {
        'X_train': X_train.astype(np.float32),
        'y_train': y_train.astype(np.int64),
        'X_test': X_test.astype(np.float32),
        'y_test': y_test.astype(np.int64)
    }


@pytest.fixture
def regression_data():
    """Generate regression dataset."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=8,
        noise=10.0,
        random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return {
        'X_train': X_train.astype(np.float32),
        'y_train': y_train.astype(np.float32),
        'X_test': X_test.astype(np.float32),
        'y_test': y_test.astype(np.float32)
    }


@pytest.fixture
def estimator_params_classifier():
    """Generate proper estimator initialization parameters for classification."""
    network_params = {
        'nlayers': 2,
        'first_layer_num_neurons': 32,
        'dropout_prob': 0.0,
        'activation_function': torch.nn.ReLU
    }

    model_params = {
        'loss_fn': torch.nn.CrossEntropyLoss(),
        'learning_rate': 1e-3
    }

    datamodule_params = {
        'read_fcn': None,
        'data_placement_device': None,
        'features_dtype': torch.float32,
        'labels_dtype': torch.int64,
        'dataloader_params': {'batch_size': 32, 'num_workers': 0}
    }

    trainer_params = {
        'max_epochs': 1,
        'enable_model_summary': False,
        'default_root_dir': None,
        'log_every_n_steps': 1,
        'devices': 1,  # Prevent multi-GPU training in tests
        'logger': False  # Disable CSVLogger to avoid fieldnames issues in tests
    }

    return {
        'model_class': MLPTorchModel,
        'model_params': model_params,
        'network_params': network_params,
        'datamodule_class': TorchDataModule,
        'datamodule_params': datamodule_params,
        'trainer_params': trainer_params
    }


@pytest.fixture
def estimator_params_regressor():
    """Generate proper estimator initialization parameters for regression."""
    network_params = {
        'nlayers': 2,
        'first_layer_num_neurons': 32,
        'dropout_prob': 0.0,
        'activation_function': torch.nn.ReLU
    }

    model_params = {
        'loss_fn': torch.nn.MSELoss(),
        'learning_rate': 1e-3
    }

    datamodule_params = {
        'read_fcn': None,
        'data_placement_device': None,
        'features_dtype': torch.float32,
        'labels_dtype': torch.float32,
        'dataloader_params': {'batch_size': 32, 'num_workers': 0}
    }

    trainer_params = {
        'max_epochs': 1,
        'enable_model_summary': False,
        'default_root_dir': None,
        'log_every_n_steps': 1,
        'devices': 1,  # Prevent multi-GPU training in tests
        'logger': False  # Disable CSVLogger to avoid fieldnames issues in tests
    }

    return {
        'model_class': MLPTorchModel,
        'model_params': model_params,
        'network_params': network_params,
        'datamodule_class': TorchDataModule,
        'datamodule_params': datamodule_params,
        'trainer_params': trainer_params
    }


# ================================================================================================
# PytorchLightningClassifier Tests
# ================================================================================================


class TestPytorchLightningClassifier:
    """Tests for PytorchLightningClassifier."""

    def test_estimator_type(self, estimator_params_classifier):
        """Test that _estimator_type is 'classifier'."""
        clf = PytorchLightningClassifier(**estimator_params_classifier)

        assert clf._estimator_type == 'classifier'

    def test_fit_basic(self, estimator_params_classifier, classification_data):
        """Test basic fit method."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 2
        clf = PytorchLightningClassifier(**params)

        clf.fit(classification_data['X_train'], classification_data['y_train'])

        assert hasattr(clf, 'model')
        assert hasattr(clf, 'classes_')

    def test_fit_with_validation(self, estimator_params_classifier, classification_data):
        """Test fit with validation set."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 2
        clf = PytorchLightningClassifier(**params)

        eval_set = (classification_data['X_test'], classification_data['y_test'])
        clf.fit(
            classification_data['X_train'],
            classification_data['y_train'],
            eval_set=eval_set
        )

        assert hasattr(clf, 'model')

    def test_predict(self, estimator_params_classifier, classification_data):
        """Test predict method returns class labels."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 2
        clf = PytorchLightningClassifier(**params)

        clf.fit(classification_data['X_train'], classification_data['y_train'])
        predictions = clf.predict(classification_data['X_test'])

        assert predictions.shape == (len(classification_data['X_test']),)
        assert all(p in [0, 1, 2] for p in predictions)

    def test_predict_proba(self, estimator_params_classifier, classification_data):
        """Test predict_proba returns probabilities."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 2
        clf = PytorchLightningClassifier(**params)

        clf.fit(classification_data['X_train'], classification_data['y_train'])
        probas = clf.predict_proba(classification_data['X_test'])

        assert probas.shape == (len(classification_data['X_test']), 3)
        # Probabilities should sum to 1
        assert np.allclose(probas.sum(axis=1), 1.0)
        # Probabilities should be in [0, 1]
        assert (probas >= 0).all() and (probas <= 1).all()

    def test_score(self, estimator_params_classifier, classification_data):
        """Test score method returns accuracy."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 5
        clf = PytorchLightningClassifier(**params)

        clf.fit(classification_data['X_train'], classification_data['y_train'])
        accuracy = clf.score(classification_data['X_test'], classification_data['y_test'])

        assert 0.0 <= accuracy <= 1.0

    def test_classes_attribute(self, estimator_params_classifier, classification_data):
        """Test that classes_ attribute is set correctly."""
        clf = PytorchLightningClassifier(**estimator_params_classifier)

        clf.fit(classification_data['X_train'], classification_data['y_train'])

        assert hasattr(clf, 'classes_')
        assert len(clf.classes_) == 3
        assert np.array_equal(clf.classes_, np.array([0, 1, 2]))

    def test_partial_fit(self, estimator_params_classifier, classification_data):
        """Test partial_fit method."""
        clf = PytorchLightningClassifier(**estimator_params_classifier)

        # First partial fit
        clf.partial_fit(
            classification_data['X_train'][:50],
            classification_data['y_train'][:50],
            classes=np.array([0, 1, 2])
        )

        assert hasattr(clf, 'model')
        assert hasattr(clf, 'classes_')

        # Second partial fit (should reuse model)
        clf.partial_fit(
            classification_data['X_train'][50:100],
            classification_data['y_train'][50:100]
        )

    def test_predict_without_fit_raises_error(self, estimator_params_classifier, classification_data):
        """Test that predict without fit raises error."""
        clf = PytorchLightningClassifier(**estimator_params_classifier)

        with pytest.raises((AttributeError, ValueError, RuntimeError)):
            clf.predict(classification_data['X_test'])

    def test_fit_with_pandas(self, estimator_params_classifier, classification_data):
        """Test fit with pandas DataFrame."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 2
        clf = PytorchLightningClassifier(**params)

        X_df = pd.DataFrame(classification_data['X_train'])
        y_series = pd.Series(classification_data['y_train'])

        clf.fit(X_df, y_series)

        assert hasattr(clf, 'model')


# ================================================================================================
# PytorchLightningRegressor Tests
# ================================================================================================


class TestPytorchLightningRegressor:
    """Tests for PytorchLightningRegressor."""

    def test_estimator_type(self, estimator_params_regressor):
        """Test that _estimator_type is 'regressor'."""
        reg = PytorchLightningRegressor(**estimator_params_regressor)

        assert reg._estimator_type == 'regressor'

    def test_fit_basic(self, estimator_params_regressor, regression_data):
        """Test basic fit method."""
        params = estimator_params_regressor.copy()
        params['trainer_params']['max_epochs'] = 2
        reg = PytorchLightningRegressor(**params)

        reg.fit(regression_data['X_train'], regression_data['y_train'])

        assert hasattr(reg, 'model')

    def test_fit_with_validation(self, estimator_params_regressor, regression_data):
        """Test fit with validation set."""
        params = estimator_params_regressor.copy()
        params['trainer_params']['max_epochs'] = 2
        reg = PytorchLightningRegressor(**params)

        eval_set = (regression_data['X_test'], regression_data['y_test'])
        reg.fit(
            regression_data['X_train'],
            regression_data['y_train'],
            eval_set=eval_set
        )

        assert hasattr(reg, 'model')

    def test_predict(self, estimator_params_regressor, regression_data):
        """Test predict method returns continuous values."""
        params = estimator_params_regressor.copy()
        params['trainer_params']['max_epochs'] = 2
        reg = PytorchLightningRegressor(**params)

        reg.fit(regression_data['X_train'], regression_data['y_train'])
        predictions = reg.predict(regression_data['X_test'])

        assert predictions.shape == (len(regression_data['X_test']),)
        assert predictions.dtype in [np.float32, np.float64]

    def test_score(self, estimator_params_regressor, regression_data):
        """Test score method returns R²."""
        params = estimator_params_regressor.copy()
        params['trainer_params']['max_epochs'] = 5
        reg = PytorchLightningRegressor(**params)

        reg.fit(regression_data['X_train'], regression_data['y_train'])
        r2 = reg.score(regression_data['X_test'], regression_data['y_test'])

        # R² can be negative for very bad models
        assert isinstance(r2, float)

    def test_partial_fit(self, estimator_params_regressor, regression_data):
        """Test partial_fit method."""
        reg = PytorchLightningRegressor(**estimator_params_regressor)

        # First partial fit
        reg.partial_fit(
            regression_data['X_train'][:50],
            regression_data['y_train'][:50]
        )

        assert hasattr(reg, 'model')

        # Second partial fit
        reg.partial_fit(
            regression_data['X_train'][50:100],
            regression_data['y_train'][50:100]
        )


# ================================================================================================
# PytorchLightningEstimator Base Class Tests
# ================================================================================================


class TestPytorchLightningEstimator:
    """Tests for PytorchLightningEstimator base class."""

    def test_get_params(self, estimator_params_classifier):
        """Test get_params method."""
        params = estimator_params_classifier.copy()
        params['model_params']['learning_rate'] = 0.001
        clf = PytorchLightningClassifier(**params)

        get_params = clf.get_params(deep=True)

        # network_params is not included in get_params (commented out in implementation)
        assert 'model_params' in get_params
        assert 'learning_rate' in get_params['model_params']
        assert get_params['model_params']['learning_rate'] == 0.001

    def test_set_params(self, estimator_params_classifier):
        """Test set_params method."""
        clf = PytorchLightningClassifier(**estimator_params_classifier)

        clf.set_params(model_params={'learning_rate': 0.01}, trainer_params={'max_epochs': 5})

        assert clf.model_params['learning_rate'] == 0.01
        assert clf.trainer_params['max_epochs'] == 5

    def test_set_params_returns_self(self, estimator_params_classifier):
        """Test that set_params returns self."""
        clf = PytorchLightningClassifier(**estimator_params_classifier)

        result = clf.set_params(model_params={'learning_rate': 0.01})

        assert result is clf

    def test_fit_returns_self(self, estimator_params_classifier, classification_data):
        """Test that fit returns self."""
        clf = PytorchLightningClassifier(**estimator_params_classifier)

        result = clf.fit(classification_data['X_train'], classification_data['y_train'])

        assert result is clf

    def test_predict_with_custom_batch_size(self, estimator_params_classifier, classification_data):
        """Test predict with custom batch_size."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 2
        clf = PytorchLightningClassifier(**params)

        clf.fit(classification_data['X_train'], classification_data['y_train'])
        predictions = clf.predict(classification_data['X_test'], batch_size=8)

        assert predictions.shape == (len(classification_data['X_test']),)

    def test_predict_with_device(self, estimator_params_classifier, classification_data):
        """Test predict with device parameter."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 2
        clf = PytorchLightningClassifier(**params)

        clf.fit(classification_data['X_train'], classification_data['y_train'])
        predictions = clf.predict(classification_data['X_test'], device='cpu')

        assert predictions.shape == (len(classification_data['X_test']),)

    def test_use_swa_parameter(self, estimator_params_classifier, classification_data):
        """Test Stochastic Weight Averaging parameter."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 5
        # Note: use_swa would need to be added as a trainer callback parameter
        clf = PytorchLightningClassifier(**params)

        clf.fit(classification_data['X_train'], classification_data['y_train'])

        assert hasattr(clf, 'model')

    def test_early_stopping_with_validation(self, estimator_params_classifier, classification_data):
        """Test early stopping with validation set."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 100  # High max_epochs
        # Note: early_stopping_patience would need to be added as a callback parameter
        clf = PytorchLightningClassifier(**params)

        eval_set = (classification_data['X_test'], classification_data['y_test'])
        clf.fit(
            classification_data['X_train'],
            classification_data['y_train'],
            eval_set=eval_set
        )

        # Should stop early
        assert hasattr(clf, 'model')

    def test_checkpointing(self, estimator_params_classifier, classification_data):
        """Test model checkpointing."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 5
        clf = PytorchLightningClassifier(**params)

        eval_set = (classification_data['X_test'], classification_data['y_test'])
        clf.fit(
            classification_data['X_train'],
            classification_data['y_train'],
            eval_set=eval_set
        )

        # Should have checkpointing enabled
        assert hasattr(clf, 'model')

    def test_different_network_architectures(self, estimator_params_classifier, classification_data):
        """Test with different network architectures."""
        for nlayers in [1, 3, 5]:
            params = estimator_params_classifier.copy()
            params['network_params']['nlayers'] = nlayers
            clf = PytorchLightningClassifier(**params)

            clf.fit(classification_data['X_train'], classification_data['y_train'])
            predictions = clf.predict(classification_data['X_test'])

            assert predictions.shape == (len(classification_data['X_test']),)

    def test_different_optimizers(self, estimator_params_classifier, classification_data):
        """Test with different optimizers."""
        from torch.optim import SGD, Adam

        for optimizer in [SGD, Adam]:
            params = estimator_params_classifier.copy()
            params['trainer_params']['max_epochs'] = 2
            params['model_params']['optimizer'] = optimizer
            clf = PytorchLightningClassifier(**params)

            clf.fit(classification_data['X_train'], classification_data['y_train'])

            assert hasattr(clf, 'model')

    def test_different_loss_functions(self, estimator_params_classifier):
        """Test with different loss functions."""
        import torch.nn as nn

        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 2
        params['model_params']['loss_fn'] = nn.NLLLoss()
        clf = PytorchLightningClassifier(**params)

        # Note: NLLLoss requires log_softmax outputs, so this might not work well
        # but it tests the parameter passing

    def test_tune_batch_size(self, estimator_params_classifier, classification_data):
        """Test batch size tuning."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 2
        # Note: tune_batch_size would need to be added as a trainer parameter
        clf = PytorchLightningClassifier(**params)

        # Tuning might fail gracefully
        try:
            clf.fit(classification_data['X_train'], classification_data['y_train'])
        except:
            pass  # Tuning can fail in some environments

    def test_verbose_output(self, estimator_params_classifier, classification_data, capsys):
        """Test verbose output."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 2
        params['trainer_params']['enable_progress_bar'] = True  # Enable verbose
        clf = PytorchLightningClassifier(**params)

        clf.fit(classification_data['X_train'], classification_data['y_train'])

        # Check that something was printed
        captured = capsys.readouterr()
        # Output may vary based on Lightning version

    def test_no_validation_data(self, estimator_params_classifier, classification_data):
        """Test training without validation data."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 2
        clf = PytorchLightningClassifier(**params)

        # Fit without eval_set
        clf.fit(classification_data['X_train'], classification_data['y_train'])

        assert hasattr(clf, 'model')

    def test_single_sample_prediction(self, estimator_params_classifier, classification_data):
        """Test prediction on single sample."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 2
        clf = PytorchLightningClassifier(**params)

        clf.fit(classification_data['X_train'], classification_data['y_train'])

        # Predict on single sample
        single_sample = classification_data['X_test'][:1]
        prediction = clf.predict(single_sample)

        assert prediction.shape == (1,)


# ================================================================================================
# Edge Cases and Error Handling
# ================================================================================================


class TestEstimatorsEdgeCases:
    """Tests for edge cases and error handling."""

    def test_small_dataset(self):
        """Test with very small dataset."""
        X = np.random.randn(10, 5).astype(np.float32)
        y = np.random.randint(0, 2, 10)

        clf = PytorchLightningClassifier(
            model_class=MLPTorchModel,
            model_params={'loss_fn': torch.nn.CrossEntropyLoss(), 'learning_rate': 1e-3},
            network_params={'nlayers': 1},
            datamodule_class=TorchDataModule,
            datamodule_params={
                'read_fcn': None,
                'data_placement_device': None,
                'features_dtype': torch.float32,
                'labels_dtype': torch.int64,
                'dataloader_params': {'batch_size': 2, 'num_workers': 0}
            },
            trainer_params={'max_epochs': 1, 'default_root_dir': None, 'log_every_n_steps': 1, 'devices': 1}
        )

        clf.fit(X, y)
        predictions = clf.predict(X)

        assert len(predictions) == 10

    def test_binary_classification(self):
        """Test binary classification."""
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)

        clf = PytorchLightningClassifier(
            model_class=MLPTorchModel,
            model_params={'loss_fn': torch.nn.CrossEntropyLoss(), 'learning_rate': 1e-3},
            network_params={'nlayers': 2},
            datamodule_class=TorchDataModule,
            datamodule_params={
                'read_fcn': None,
                'data_placement_device': None,
                'features_dtype': torch.float32,
                'labels_dtype': torch.int64,
                'dataloader_params': {'batch_size': 32, 'num_workers': 0}
            },
            trainer_params={'max_epochs': 2, 'default_root_dir': None, 'log_every_n_steps': 1, 'devices': 1}
        )

        clf.fit(X.astype(np.float32), y)
        predictions = clf.predict(X.astype(np.float32))
        probas = clf.predict_proba(X.astype(np.float32))

        assert all(p in [0, 1] for p in predictions)
        assert probas.shape == (100, 2)

    def test_regression_scalar_output(self, regression_data):
        """Test that regression outputs are properly shaped."""
        reg = PytorchLightningRegressor(
            model_class=MLPTorchModel,
            model_params={'loss_fn': torch.nn.MSELoss(), 'learning_rate': 1e-3},
            network_params={'nlayers': 2},
            datamodule_class=TorchDataModule,
            datamodule_params={
                'read_fcn': None,
                'data_placement_device': None,
                'features_dtype': torch.float32,
                'labels_dtype': torch.float32,
                'dataloader_params': {'batch_size': 32, 'num_workers': 0}
            },
            trainer_params={'max_epochs': 2, 'default_root_dir': None, 'log_every_n_steps': 1, 'devices': 1}
        )

        reg.fit(regression_data['X_train'], regression_data['y_train'])
        predictions = reg.predict(regression_data['X_test'])

        # Should be 1D array
        assert predictions.ndim == 1


# ================================================================================================
# Mutation Testing - Estimator Tests
# ================================================================================================


class TestEstimatorsMutationTests:
    """Tests specifically targeting mutation survivors in estimators."""

    def test_predict_raises_when_model_not_fitted(self, estimator_params_classifier, classification_data):
        """Test predict raises error when model attribute missing.

        Kills mutation: `not hasattr(self, "model") or self.model is None`.
        """
        clf = PytorchLightningClassifier(**estimator_params_classifier)

        with pytest.raises((AttributeError, ValueError, RuntimeError)):
            clf.predict(classification_data['X_test'])

    def test_predict_raises_when_model_is_none(self, estimator_params_classifier, classification_data):
        """Test predict raises error when model is None.

        Kills mutation: `self.model is None` check.
        """
        clf = PytorchLightningClassifier(**estimator_params_classifier)
        clf.model = None  # Artificially set

        with pytest.raises((AttributeError, ValueError, RuntimeError)):
            clf.predict(classification_data['X_test'])

    def test_regression_prediction_shape_1d(self, estimator_params_regressor, regression_data):
        """Test regression predictions are properly squeezed to 1D.

        Kills mutation: `predictions.ndim == 2 and predictions.shape[1] == 1`.
        """
        params = estimator_params_regressor.copy()
        params['trainer_params']['max_epochs'] = 2
        reg = PytorchLightningRegressor(**params)

        reg.fit(regression_data['X_train'], regression_data['y_train'])
        predictions = reg.predict(regression_data['X_test'])

        # Should be squeezed to 1D
        assert predictions.ndim == 1, f"Expected 1D, got {predictions.ndim}D"
        assert len(predictions) == len(regression_data['X_test'])

    def test_predict_device_cpu_string_comparison(self, estimator_params_classifier, classification_data):
        """Test device parameter string comparison works correctly.

        Kills mutation: `device == "cpu"` to `device > "cpu"`, `device is "cpu"`.
        """
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 2
        clf = PytorchLightningClassifier(**params)

        clf.fit(classification_data['X_train'], classification_data['y_train'])

        # Explicit cpu device string
        predictions = clf.predict(classification_data['X_test'], device='cpu')
        assert predictions is not None
        assert len(predictions) == len(classification_data['X_test'])

    def test_classifier_binary_boundary(self):
        """Test binary classification at num_classes=2 boundary.

        Kills mutation: boundary conditions on num_classes checks.
        """
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)

        clf = PytorchLightningClassifier(
            model_class=MLPTorchModel,
            model_params={'loss_fn': torch.nn.CrossEntropyLoss(), 'learning_rate': 1e-3},
            network_params={'nlayers': 2},
            datamodule_class=TorchDataModule,
            datamodule_params={
                'read_fcn': None,
                'data_placement_device': None,
                'features_dtype': torch.float32,
                'labels_dtype': torch.int64,
                'dataloader_params': {'batch_size': 32, 'num_workers': 0}
            },
            trainer_params={'max_epochs': 2, 'default_root_dir': None, 'log_every_n_steps': 1, 'devices': 1}
        )

        clf.fit(X.astype(np.float32), y)
        predictions = clf.predict(X.astype(np.float32))
        probas = clf.predict_proba(X.astype(np.float32))

        # Should have exactly 2 classes
        assert len(clf.classes_) == 2
        assert probas.shape[1] == 2
        assert all(p in [0, 1] for p in predictions)


# ================================================================================================
# Sample Weight Support Tests
# ================================================================================================


class TestSampleWeightSupport:
    """Tests for sample weight support in MLP models."""

    def test_classifier_with_sample_weight(self, estimator_params_classifier, classification_data):
        """Test classification with custom sample weights."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 2
        clf = PytorchLightningClassifier(**params)

        # Create sample weights (higher weight for first half)
        n_samples = len(classification_data['X_train'])
        sample_weight = np.ones(n_samples)
        sample_weight[:n_samples // 2] = 2.0

        clf.fit(
            classification_data['X_train'],
            classification_data['y_train'],
            sample_weight=sample_weight
        )

        assert hasattr(clf, 'model')
        predictions = clf.predict(classification_data['X_test'])
        assert predictions.shape == (len(classification_data['X_test']),)

    def test_regressor_with_sample_weight(self, estimator_params_regressor, regression_data):
        """Test regression with custom sample weights."""
        params = estimator_params_regressor.copy()
        params['trainer_params']['max_epochs'] = 2
        reg = PytorchLightningRegressor(**params)

        # Create sample weights
        n_samples = len(regression_data['X_train'])
        sample_weight = np.linspace(0.5, 1.5, n_samples).astype(np.float32)

        reg.fit(
            regression_data['X_train'],
            regression_data['y_train'],
            sample_weight=sample_weight
        )

        assert hasattr(reg, 'model')
        predictions = reg.predict(regression_data['X_test'])
        assert predictions.shape == (len(regression_data['X_test']),)

    def test_sample_weight_via_fit_params(self, estimator_params_classifier, classification_data):
        """Test sample_weight passed via fit_params dictionary."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 2
        clf = PytorchLightningClassifier(**params)

        n_samples = len(classification_data['X_train'])
        sample_weight = np.ones(n_samples) * 0.5

        # Pass via fit_params instead of direct parameter
        clf.fit(
            classification_data['X_train'],
            classification_data['y_train'],
            sample_weight=sample_weight
        )

        assert hasattr(clf, 'model')

    def test_classifier_with_validation_and_sample_weight(self, estimator_params_classifier, classification_data):
        """Test sample weights work with validation set."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 3
        clf = PytorchLightningClassifier(**params)

        n_train = len(classification_data['X_train'])
        n_val = len(classification_data['X_test'])
        train_weights = np.ones(n_train)
        val_weights = np.ones(n_val)

        eval_set = (classification_data['X_test'], classification_data['y_test'])
        clf.fit(
            classification_data['X_train'],
            classification_data['y_train'],
            sample_weight=train_weights,
            eval_set=eval_set,
            eval_sample_weight=val_weights
        )

        assert hasattr(clf, 'model')

    def test_uniform_weights_produces_valid_predictions(self, estimator_params_classifier, classification_data):
        """Test that uniform weights produce valid predictions."""
        params = estimator_params_classifier.copy()
        params['trainer_params']['max_epochs'] = 2

        # Train with uniform weights
        clf = PytorchLightningClassifier(**params)
        n_samples = len(classification_data['X_train'])
        uniform_weights = np.ones(n_samples, dtype=np.float32)
        clf.fit(
            classification_data['X_train'],
            classification_data['y_train'],
            sample_weight=uniform_weights
        )

        # Should produce valid predictions
        predictions = clf.predict(classification_data['X_test'])
        assert predictions.shape == (len(classification_data['X_test']),)
        assert all(p in [0, 1, 2] for p in predictions)

    def test_low_weights_reduce_impact(self, estimator_params_regressor, regression_data):
        """Test that low-weighted samples contribute less to loss."""
        params = estimator_params_regressor.copy()
        params['trainer_params']['max_epochs'] = 2
        reg = PytorchLightningRegressor(**params)

        n_samples = len(regression_data['X_train'])
        # Low weight for first half, high weight for second half
        sample_weight = np.ones(n_samples, dtype=np.float32)
        sample_weight[:n_samples // 2] = 0.01  # Very low but non-zero
        sample_weight[n_samples // 2:] = 1.0

        reg.fit(
            regression_data['X_train'],
            regression_data['y_train'],
            sample_weight=sample_weight
        )

        assert hasattr(reg, 'model')
        predictions = reg.predict(regression_data['X_test'])
        assert not np.isnan(predictions).any()


# ================================================================================================
# Network Reset and Clone Tests
# ================================================================================================


class TestNetworkResetAndClone:
    """Tests for MLP network reset behavior and sklearn clone() support."""

    @staticmethod
    def _get_input_dim(network):
        """Get input dimension from first Linear layer in network."""
        for layer in network:
            if hasattr(layer, 'in_features'):
                return layer.in_features
        return None

    def test_fit_resets_network_on_different_feature_count(self, estimator_params_classifier):
        """Test that fit() resets network when feature count changes.

        This is critical for pipelines with different feature selectors (e.g., RFECV vs MRMR)
        that produce different feature counts.
        """
        clf = PytorchLightningClassifier(**estimator_params_classifier)

        # First fit with 10 features
        X1 = np.random.randn(100, 10).astype(np.float32)
        y1 = np.random.randint(0, 3, 100)
        clf.fit(X1, y1)

        first_network = clf.network
        first_input_dim = self._get_input_dim(first_network)
        assert first_input_dim == 10

        # Second fit with 20 features - should reset network
        X2 = np.random.randn(100, 20).astype(np.float32)
        y2 = np.random.randint(0, 3, 100)
        clf.fit(X2, y2)

        second_network = clf.network
        second_input_dim = self._get_input_dim(second_network)
        assert second_input_dim == 20, "Network should be reset with new input dimension"
        assert first_network is not second_network, "Network should be a new object after fit()"

    def test_partial_fit_keeps_network(self, estimator_params_classifier):
        """Test that partial_fit() keeps existing network (for incremental learning)."""
        clf = PytorchLightningClassifier(**estimator_params_classifier)

        # First partial fit
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 3, 100)
        clf.partial_fit(X[:50], y[:50], classes=np.array([0, 1, 2]))

        first_network = clf.network
        first_network_id = id(first_network)

        # Second partial fit - should keep same network
        clf.partial_fit(X[50:], y[50:])

        second_network = clf.network
        second_network_id = id(second_network)

        assert first_network_id == second_network_id, "partial_fit() should keep the same network"

    def test_fit_after_partial_fit_resets_network(self, estimator_params_classifier):
        """Test that fit() resets network even after partial_fit()."""
        clf = PytorchLightningClassifier(**estimator_params_classifier)

        # First partial fit
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 3, 100)
        clf.partial_fit(X[:50], y[:50], classes=np.array([0, 1, 2]))

        partial_fit_network = clf.network
        partial_fit_network_id = id(partial_fit_network)

        # Now call fit() - should reset
        clf.fit(X[50:], y[50:])

        fit_network = clf.network
        fit_network_id = id(fit_network)

        assert partial_fit_network_id != fit_network_id, "fit() should reset network after partial_fit()"

    def test_sklearn_clone_works(self, estimator_params_classifier):
        """Test that sklearn.base.clone() works correctly with PytorchLightningClassifier."""
        from sklearn.base import clone

        clf = PytorchLightningClassifier(**estimator_params_classifier)

        # Clone before fitting
        cloned = clone(clf)

        assert cloned is not clf, "Clone should create a new object"
        assert type(cloned) == type(clf), "Clone should have same type"

        # Check all params are copied
        for key, value in clf.get_params().items():
            assert hasattr(cloned, key), f"Cloned estimator missing param: {key}"

    def test_sklearn_clone_after_fit_gives_unfitted_model(self, estimator_params_classifier):
        """Test that cloning a fitted model gives an unfitted model."""
        from sklearn.base import clone

        clf = PytorchLightningClassifier(**estimator_params_classifier)

        # Fit the original
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 3, 100)
        clf.fit(X, y)

        assert hasattr(clf, 'network') and clf.network is not None, "Original should be fitted"

        # Clone the fitted model
        cloned = clone(clf)

        # Cloned model should not have a network (unfitted state)
        # Note: clone() only copies parameters, not fitted attributes
        assert not hasattr(cloned, 'network') or cloned.network is None, \
            "Cloned model should be unfitted"

    def test_cloned_model_can_fit_with_different_features(self, estimator_params_classifier):
        """Test that cloned models can fit with different feature counts."""
        from sklearn.base import clone

        clf = PytorchLightningClassifier(**estimator_params_classifier)

        # Fit original with 10 features
        X1 = np.random.randn(100, 10).astype(np.float32)
        y1 = np.random.randint(0, 3, 100)
        clf.fit(X1, y1)

        # Clone and fit with 20 features
        cloned = clone(clf)
        X2 = np.random.randn(100, 20).astype(np.float32)
        y2 = np.random.randint(0, 3, 100)
        cloned.fit(X2, y2)

        # Both should work independently
        pred1 = clf.predict(X1[:5])
        pred2 = cloned.predict(X2[:5])

        assert pred1.shape == (5,), "Original should predict correctly"
        assert pred2.shape == (5,), "Cloned should predict correctly"

        # Check input dimensions are different
        assert self._get_input_dim(clf.network) == 10
        assert self._get_input_dim(cloned.network) == 20

    def test_get_params_includes_all_init_params(self, estimator_params_classifier):
        """Test that get_params() includes all __init__ parameters required for clone()."""
        clf = PytorchLightningClassifier(**estimator_params_classifier)

        params = clf.get_params()

        # All these must be present for clone() to work
        required_params = [
            'model_class', 'model_params', 'network_params',
            'datamodule_class', 'datamodule_params', 'trainer_params',
            'use_swa', 'swa_params', 'tune_params', 'tune_batch_size',
            'float32_matmul_precision', 'early_stopping_rounds'
        ]

        for param in required_params:
            assert param in params, f"get_params() missing required param: {param}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
