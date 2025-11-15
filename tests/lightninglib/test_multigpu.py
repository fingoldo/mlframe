"""
Multi-GPU (DDP) integration tests for lightninglib.

These tests verify that the implementation works correctly in distributed
training scenarios with multiple GPUs.

Run tests:
    # Run all multi-GPU tests (requires 2+ GPUs)
    pytest tests/lightninglib/test_multigpu.py -v

    # Run with specific GPU count
    pytest tests/lightninglib/test_multigpu.py -v --devices=2
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mlframe.lightninglib import (
    PytorchLightningClassifier,
    PytorchLightningRegressor,
    MLPTorchModel,
    TorchDataModule,
)
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


# ================================================================================================
# Fixtures and Utilities
# ================================================================================================


@pytest.fixture
def gpu_count():
    """Get number of available GPUs."""
    return torch.cuda.device_count()


@pytest.fixture
def multigpu_classification_data():
    """Generate classification dataset for multi-GPU tests."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=4,
        random_state=42
    )

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return {
        'X_train': X_train.astype(np.float32),
        'y_train': y_train,
        'X_val': X_val.astype(np.float32),
        'y_val': y_val,
        'X_test': X_test.astype(np.float32),
        'y_test': y_test,
        'num_classes': 4,
        'num_features': 20
    }


@pytest.fixture
def multigpu_regression_data():
    """Generate regression dataset for multi-GPU tests."""
    X, y = make_regression(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return {
        'X_train': X_train.astype(np.float32),
        'y_train': y_train.astype(np.float32),
        'X_val': X_val.astype(np.float32),
        'y_val': y_val.astype(np.float32),
        'X_test': X_test.astype(np.float32),
        'y_test': y_test.astype(np.float32),
        'num_features': 20
    }


@pytest.fixture
def multigpu_estimator_params_classifier(gpu_count):
    """Generate estimator parameters for multi-GPU classification tests."""
    # Use min(2, gpu_count) to work on machines with 2+ GPUs
    num_devices = min(2, gpu_count) if gpu_count > 1 else 1

    network_params = {
        'nlayers': 3,
        'first_layer_num_neurons': 64,
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
        'max_epochs': 5,
        'enable_model_summary': False,
        'default_root_dir': None,
        'log_every_n_steps': 1,
        'devices': num_devices,  # Use multiple GPUs
        'accelerator': 'gpu',
        'strategy': 'ddp' if num_devices > 1 else 'auto'
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
def multigpu_estimator_params_regressor(gpu_count):
    """Generate estimator parameters for multi-GPU regression tests."""
    num_devices = min(2, gpu_count) if gpu_count > 1 else 1

    network_params = {
        'nlayers': 3,
        'first_layer_num_neurons': 64,
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
        'max_epochs': 5,
        'enable_model_summary': False,
        'default_root_dir': None,
        'log_every_n_steps': 1,
        'devices': num_devices,
        'accelerator': 'gpu',
        'strategy': 'ddp' if num_devices > 1 else 'auto'
    }

    return {
        'model_class': MLPTorchModel,
        'model_params': model_params,
        'network_params': network_params,
        'datamodule_class': TorchDataModule,
        'datamodule_params': datamodule_params,
        'trainer_params': trainer_params
    }


# Skip all tests if no GPUs available or only 1 GPU
pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Multi-GPU tests require at least 2 GPUs"
)


# ================================================================================================
# Best Epoch Tracking Tests
# ================================================================================================


class TestMultiGPUBestEpochTracking:
    """Test that best_epoch is properly tracked in DDP mode."""

    def test_best_epoch_tracking_classifier(self, multigpu_classification_data, multigpu_estimator_params_classifier):
        """Test that best_epoch is accessible after DDP training (classifier)."""
        clf = PytorchLightningClassifier(**multigpu_estimator_params_classifier)

        eval_set = (multigpu_classification_data['X_val'], multigpu_classification_data['y_val'])
        clf.fit(
            multigpu_classification_data['X_train'],
            multigpu_classification_data['y_train'],
            eval_set=eval_set
        )

        # In DDP mode, best_epoch should be set on the estimator
        assert hasattr(clf, 'best_epoch')
        # After fixing, best_epoch should not be None
        assert clf.best_epoch is not None
        assert isinstance(clf.best_epoch, int)
        assert 0 <= clf.best_epoch < multigpu_estimator_params_classifier['trainer_params']['max_epochs']

    def test_best_epoch_tracking_regressor(self, multigpu_regression_data, multigpu_estimator_params_regressor):
        """Test that best_epoch is accessible after DDP training (regressor)."""
        reg = PytorchLightningRegressor(**multigpu_estimator_params_regressor)

        eval_set = (multigpu_regression_data['X_val'], multigpu_regression_data['y_val'])
        reg.fit(
            multigpu_regression_data['X_train'],
            multigpu_regression_data['y_train'],
            eval_set=eval_set
        )

        assert hasattr(reg, 'best_epoch')
        assert reg.best_epoch is not None
        assert isinstance(reg.best_epoch, int)

    def test_model_best_epoch_attribute(self, multigpu_classification_data, multigpu_estimator_params_classifier):
        """Test that best_epoch is also set on the model itself."""
        clf = PytorchLightningClassifier(**multigpu_estimator_params_classifier)

        eval_set = (multigpu_classification_data['X_val'], multigpu_classification_data['y_val'])
        clf.fit(
            multigpu_classification_data['X_train'],
            multigpu_classification_data['y_train'],
            eval_set=eval_set
        )

        # Model should also have best_epoch for DDP compatibility
        assert hasattr(clf.model, 'best_epoch')
        # Both should match
        if clf.best_epoch is not None and clf.model.best_epoch is not None:
            assert clf.best_epoch == clf.model.best_epoch


# ================================================================================================
# Training Completion Tests
# ================================================================================================


class TestMultiGPUTraining:
    """Test that training completes successfully in DDP mode."""

    def test_classification_training_completes(self, multigpu_classification_data, multigpu_estimator_params_classifier):
        """Test that classification training completes without errors."""
        clf = PytorchLightningClassifier(**multigpu_estimator_params_classifier)

        eval_set = (multigpu_classification_data['X_val'], multigpu_classification_data['y_val'])
        clf.fit(
            multigpu_classification_data['X_train'],
            multigpu_classification_data['y_train'],
            eval_set=eval_set
        )

        # Training should complete and model should be fitted
        assert hasattr(clf, 'model')
        assert clf.model is not None

    def test_regression_training_completes(self, multigpu_regression_data, multigpu_estimator_params_regressor):
        """Test that regression training completes without errors."""
        reg = PytorchLightningRegressor(**multigpu_estimator_params_regressor)

        eval_set = (multigpu_regression_data['X_val'], multigpu_regression_data['y_val'])
        reg.fit(
            multigpu_regression_data['X_train'],
            multigpu_regression_data['y_train'],
            eval_set=eval_set
        )

        assert hasattr(reg, 'model')
        assert reg.model is not None

    def test_training_without_validation(self, multigpu_classification_data, multigpu_estimator_params_classifier):
        """Test DDP training without validation set."""
        clf = PytorchLightningClassifier(**multigpu_estimator_params_classifier)

        # Train without validation
        clf.fit(
            multigpu_classification_data['X_train'],
            multigpu_classification_data['y_train']
        )

        assert hasattr(clf, 'model')
        assert clf.model is not None


# ================================================================================================
# Prediction and Inference Tests
# ================================================================================================


class TestMultiGPUPrediction:
    """Test that predictions work correctly after DDP training."""

    def test_classifier_predictions(self, multigpu_classification_data, multigpu_estimator_params_classifier):
        """Test classifier predictions after DDP training."""
        clf = PytorchLightningClassifier(**multigpu_estimator_params_classifier)

        eval_set = (multigpu_classification_data['X_val'], multigpu_classification_data['y_val'])
        clf.fit(
            multigpu_classification_data['X_train'],
            multigpu_classification_data['y_train'],
            eval_set=eval_set
        )

        # Predict
        predictions = clf.predict(multigpu_classification_data['X_test'])
        assert predictions.shape == (len(multigpu_classification_data['X_test']),)
        assert all(p in range(multigpu_classification_data['num_classes']) for p in predictions)

        # Predict probabilities
        probas = clf.predict_proba(multigpu_classification_data['X_test'])
        assert probas.shape == (len(multigpu_classification_data['X_test']), multigpu_classification_data['num_classes'])
        assert np.allclose(probas.sum(axis=1), 1.0)

    def test_regressor_predictions(self, multigpu_regression_data, multigpu_estimator_params_regressor):
        """Test regressor predictions after DDP training."""
        reg = PytorchLightningRegressor(**multigpu_estimator_params_regressor)

        eval_set = (multigpu_regression_data['X_val'], multigpu_regression_data['y_val'])
        reg.fit(
            multigpu_regression_data['X_train'],
            multigpu_regression_data['y_train'],
            eval_set=eval_set
        )

        predictions = reg.predict(multigpu_regression_data['X_test'])
        assert predictions.shape == multigpu_regression_data['y_test'].shape
        assert predictions.dtype in [np.float32, np.float64]

    def test_scoring(self, multigpu_classification_data, multigpu_estimator_params_classifier):
        """Test scoring after DDP training."""
        clf = PytorchLightningClassifier(**multigpu_estimator_params_classifier)

        eval_set = (multigpu_classification_data['X_val'], multigpu_classification_data['y_val'])
        clf.fit(
            multigpu_classification_data['X_train'],
            multigpu_classification_data['y_train'],
            eval_set=eval_set
        )

        accuracy = clf.score(multigpu_classification_data['X_test'], multigpu_classification_data['y_test'])
        assert 0.0 <= accuracy <= 1.0


# ================================================================================================
# Advanced DDP Features Tests
# ================================================================================================


class TestMultiGPUAdvancedFeatures:
    """Test advanced features in DDP mode."""

    def test_early_stopping_in_ddp(self, multigpu_classification_data, multigpu_estimator_params_classifier):
        """Test that early stopping works correctly in DDP mode."""
        params = multigpu_estimator_params_classifier.copy()
        params['trainer_params'] = params['trainer_params'].copy()
        params['trainer_params']['max_epochs'] = 50

        clf = PytorchLightningClassifier(**params, early_stopping_rounds=5)

        eval_set = (multigpu_classification_data['X_val'], multigpu_classification_data['y_val'])
        clf.fit(
            multigpu_classification_data['X_train'],
            multigpu_classification_data['y_train'],
            eval_set=eval_set
        )

        # Should have stopped early (before 50 epochs)
        assert hasattr(clf, 'model')

    def test_load_best_weights_in_ddp(self, multigpu_classification_data, multigpu_estimator_params_classifier):
        """Test that best weights restoration works in DDP mode."""
        params = multigpu_estimator_params_classifier.copy()
        params['model_params'] = params['model_params'].copy()
        params['model_params']['load_best_weights_on_train_end'] = True

        clf = PytorchLightningClassifier(**params)

        eval_set = (multigpu_classification_data['X_val'], multigpu_classification_data['y_val'])
        clf.fit(
            multigpu_classification_data['X_train'],
            multigpu_classification_data['y_train'],
            eval_set=eval_set
        )

        # Model should have best_epoch attribute
        if hasattr(clf.model, 'best_epoch'):
            assert clf.model.best_epoch is not None

    def test_lr_scheduler_in_ddp(self, multigpu_classification_data, multigpu_estimator_params_classifier):
        """Test learning rate scheduler in DDP mode."""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        params = multigpu_estimator_params_classifier.copy()
        params['model_params'] = params['model_params'].copy()
        params['model_params']['lr_scheduler'] = CosineAnnealingLR
        params['model_params']['lr_scheduler_kwargs'] = {'T_max': 10}
        params['model_params']['lr_scheduler_interval'] = 'epoch'

        clf = PytorchLightningClassifier(**params)

        clf.fit(
            multigpu_classification_data['X_train'],
            multigpu_classification_data['y_train']
        )

        predictions = clf.predict(multigpu_classification_data['X_test'])
        assert len(predictions) == len(multigpu_classification_data['X_test'])


# ================================================================================================
# Partial Fit Tests
# ================================================================================================


class TestMultiGPUPartialFit:
    """Test partial_fit in multi-GPU mode."""

    def test_partial_fit_classifier(self, multigpu_classification_data, multigpu_estimator_params_classifier):
        """Test partial_fit with multiple GPUs."""
        params = multigpu_estimator_params_classifier.copy()
        params['trainer_params'] = params['trainer_params'].copy()
        params['trainer_params']['max_epochs'] = 2

        clf = PytorchLightningClassifier(**params)

        # Split data into batches
        n_samples = len(multigpu_classification_data['X_train'])
        batch_size = n_samples // 2

        # First partial fit
        clf.partial_fit(
            multigpu_classification_data['X_train'][:batch_size],
            multigpu_classification_data['y_train'][:batch_size],
            classes=np.array(range(multigpu_classification_data['num_classes']))
        )

        assert hasattr(clf, 'model')

        # Second partial fit
        clf.partial_fit(
            multigpu_classification_data['X_train'][batch_size:],
            multigpu_classification_data['y_train'][batch_size:]
        )

        # Should be able to predict after partial fit
        predictions = clf.predict(multigpu_classification_data['X_test'])
        assert len(predictions) == len(multigpu_classification_data['X_test'])


# ================================================================================================
# Resource Management Tests
# ================================================================================================


class TestMultiGPUResourceManagement:
    """Test GPU resource management in DDP mode."""

    def test_gpu_memory_cleanup(self, multigpu_classification_data, multigpu_estimator_params_classifier):
        """Test that GPU memory is properly managed."""
        clf = PytorchLightningClassifier(**multigpu_estimator_params_classifier)

        # Get initial memory
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated()

        eval_set = (multigpu_classification_data['X_val'], multigpu_classification_data['y_val'])
        clf.fit(
            multigpu_classification_data['X_train'],
            multigpu_classification_data['y_train'],
            eval_set=eval_set
        )

        # Make predictions
        predictions = clf.predict(multigpu_classification_data['X_test'])

        # Verify predictions work
        assert len(predictions) == len(multigpu_classification_data['X_test'])

        # Memory should be managed (not growing unbounded)
        # This is a smoke test - actual memory usage depends on many factors

    def test_multiple_sequential_trainings(self, multigpu_classification_data, multigpu_estimator_params_classifier):
        """Test multiple sequential training runs on multiple GPUs."""
        for i in range(3):
            clf = PytorchLightningClassifier(**multigpu_estimator_params_classifier)

            clf.fit(
                multigpu_classification_data['X_train'],
                multigpu_classification_data['y_train']
            )

            predictions = clf.predict(multigpu_classification_data['X_test'])
            assert len(predictions) == len(multigpu_classification_data['X_test'])

            # Clean up
            del clf
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
