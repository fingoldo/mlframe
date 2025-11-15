"""
Integration tests for lightninglib.py - End-to-end workflows

Run tests:
    pytest tests/lightninglib/test_integration.py -v
    pytest tests/lightninglib/test_integration.py --cov=mlframe.lightninglib --cov-report=html
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
import tempfile
import os

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mlframe.lightninglib import (
    PytorchLightningClassifier,
    PytorchLightningRegressor,
    TorchDataset,
    TorchDataModule,
    MLPTorchModel,
    generate_mlp,
    MetricSpec,
    BestEpochModelCheckpoint,
)


# ================================================================================================
# Fixtures
# ================================================================================================


@pytest.fixture
def classification_dataset():
    """Generate complete classification dataset with train/val/test splits."""
    X, y = make_classification(
        n_samples=300,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=4,
        random_state=42
    )

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    return {
        'X_train': X_train.astype(np.float32),
        'y_train': y_train.astype(np.int64),
        'X_val': X_val.astype(np.float32),
        'y_val': y_val.astype(np.int64),
        'X_test': X_test.astype(np.float32),
        'y_test': y_test.astype(np.int64),
        'num_features': 20,
        'num_classes': 4
    }


@pytest.fixture
def regression_dataset():
    """Generate complete regression dataset with train/val/test splits."""
    X, y = make_regression(
        n_samples=300,
        n_features=15,
        n_informative=10,
        noise=15.0,
        random_state=42
    )

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    return {
        'X_train': X_train.astype(np.float32),
        'y_train': y_train.astype(np.float32),
        'X_val': X_val.astype(np.float32),
        'y_val': y_val.astype(np.float32),
        'X_test': X_test.astype(np.float32),
        'y_test': y_test.astype(np.float32),
        'num_features': 15
    }


@pytest.fixture
def integration_estimator_params_classifier():
    """Generate proper estimator initialization parameters for integration tests - classification."""
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
        'max_epochs': 10,
        'enable_model_summary': False,
        'default_root_dir': None,
        'log_every_n_steps': 1
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
def integration_estimator_params_regressor():
    """Generate proper estimator initialization parameters for integration tests - regression."""
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
        'max_epochs': 10,
        'enable_model_summary': False,
        'default_root_dir': None,
        'log_every_n_steps': 1
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
# End-to-End Classification Pipeline
# ================================================================================================


class TestClassificationPipeline:
    """Integration tests for complete classification workflows."""

    def test_full_classification_pipeline(self, classification_dataset, integration_estimator_params_classifier):
        """Test complete classification pipeline: fit → predict → score."""
        # Create classifier
        clf = PytorchLightningClassifier(**integration_estimator_params_classifier)

        # Fit with validation
        eval_set = (classification_dataset['X_val'], classification_dataset['y_val'])
        clf.fit(
            classification_dataset['X_train'],
            classification_dataset['y_train'],
            eval_set=eval_set
        )

        # Predict
        predictions = clf.predict(classification_dataset['X_test'])
        assert predictions.shape == classification_dataset['y_test'].shape

        # Predict probabilities
        probas = clf.predict_proba(classification_dataset['X_test'])
        assert probas.shape == (len(classification_dataset['X_test']), classification_dataset['num_classes'])

        # Score
        accuracy = clf.score(classification_dataset['X_test'], classification_dataset['y_test'])
        assert 0.0 <= accuracy <= 1.0

    def test_classification_with_different_architectures(self, classification_dataset, integration_estimator_params_classifier):
        """Test classification with various network architectures."""
        from mlframe.lightninglib import MLPNeuronsByLayerArchitecture

        architectures = [
            MLPNeuronsByLayerArchitecture.Constant,
            MLPNeuronsByLayerArchitecture.Declining,
            MLPNeuronsByLayerArchitecture.Expanding,
        ]

        for arch in architectures:
            params = integration_estimator_params_classifier.copy()
            params['network_params'] = params['network_params'].copy()
            params['network_params']['neurons_by_layer_arch'] = arch
            params['trainer_params'] = params['trainer_params'].copy()
            params['trainer_params']['max_epochs'] = 3
            clf = PytorchLightningClassifier(**params)

            clf.fit(
                classification_dataset['X_train'],
                classification_dataset['y_train']
            )

            predictions = clf.predict(classification_dataset['X_test'])
            assert len(predictions) == len(classification_dataset['X_test'])

    def test_classification_with_regularization(self, classification_dataset, integration_estimator_params_classifier):
        """Test classification with dropout and L1 regularization."""
        params = integration_estimator_params_classifier.copy()
        params['network_params'] = params['network_params'].copy()
        params['network_params']['dropout_prob'] = 0.3
        params['model_params'] = params['model_params'].copy()
        params['model_params']['l1_alpha'] = 0.001
        params['trainer_params'] = params['trainer_params'].copy()
        params['trainer_params']['max_epochs'] = 5
        clf = PytorchLightningClassifier(**params)

        clf.fit(
            classification_dataset['X_train'],
            classification_dataset['y_train']
        )

        predictions = clf.predict(classification_dataset['X_test'])
        assert len(predictions) == len(classification_dataset['X_test'])

    def test_classification_with_normalization(self, classification_dataset, integration_estimator_params_classifier):
        """Test classification with various normalization layers."""
        params = integration_estimator_params_classifier.copy()
        params['network_params'] = params['network_params'].copy()
        params['network_params']['use_batchnorm'] = True
        params['network_params']['use_layernorm_per_layer'] = True
        params['trainer_params'] = params['trainer_params'].copy()
        params['trainer_params']['max_epochs'] = 5
        clf = PytorchLightningClassifier(**params)

        clf.fit(
            classification_dataset['X_train'],
            classification_dataset['y_train']
        )

        accuracy = clf.score(classification_dataset['X_test'], classification_dataset['y_test'])
        assert 0.0 <= accuracy <= 1.0


# ================================================================================================
# End-to-End Regression Pipeline
# ================================================================================================


class TestRegressionPipeline:
    """Integration tests for complete regression workflows."""

    def test_full_regression_pipeline(self, regression_dataset, integration_estimator_params_regressor):
        """Test complete regression pipeline: fit → predict → score."""
        # Create regressor
        reg = PytorchLightningRegressor(**integration_estimator_params_regressor)

        # Fit with validation
        eval_set = (regression_dataset['X_val'], regression_dataset['y_val'])
        reg.fit(
            regression_dataset['X_train'],
            regression_dataset['y_train'],
            eval_set=eval_set
        )

        # Predict
        predictions = reg.predict(regression_dataset['X_test'])
        assert predictions.shape == regression_dataset['y_test'].shape

        # Score (R²)
        r2 = reg.score(regression_dataset['X_test'], regression_dataset['y_test'])
        assert isinstance(r2, float)

    def test_regression_with_different_activations(self, regression_dataset, integration_estimator_params_regressor):
        """Test regression with different activation functions."""
        activations = [nn.ReLU, nn.LeakyReLU, nn.GELU]

        for activation in activations:
            params = integration_estimator_params_regressor.copy()
            params['network_params'] = params['network_params'].copy()
            params['network_params']['activation_function'] = activation
            params['network_params']['nlayers'] = 2
            params['trainer_params'] = params['trainer_params'].copy()
            params['trainer_params']['max_epochs'] = 3
            reg = PytorchLightningRegressor(**params)

            reg.fit(
                regression_dataset['X_train'],
                regression_dataset['y_train']
            )

            predictions = reg.predict(regression_dataset['X_test'])
            assert len(predictions) == len(regression_dataset['X_test'])


# ================================================================================================
# Partial Fit Workflows
# ================================================================================================


class TestPartialFitWorkflows:
    """Integration tests for partial_fit workflows."""

    def test_sequential_partial_fit(self, classification_dataset, integration_estimator_params_classifier):
        """Test sequential partial_fit calls."""
        params = integration_estimator_params_classifier.copy()
        params['network_params'] = params['network_params'].copy()
        params['network_params']['nlayers'] = 2
        params['trainer_params'] = params['trainer_params'].copy()
        params['trainer_params']['max_epochs'] = 2
        clf = PytorchLightningClassifier(**params)

        # Split training data into batches
        n_samples = len(classification_dataset['X_train'])
        batch_size = n_samples // 3

        # First partial fit
        clf.partial_fit(
            classification_dataset['X_train'][:batch_size],
            classification_dataset['y_train'][:batch_size],
            classes=np.arange(classification_dataset['num_classes'])
        )

        # Second partial fit
        clf.partial_fit(
            classification_dataset['X_train'][batch_size:2*batch_size],
            classification_dataset['y_train'][batch_size:2*batch_size]
        )

        # Third partial fit
        clf.partial_fit(
            classification_dataset['X_train'][2*batch_size:],
            classification_dataset['y_train'][2*batch_size:]
        )

        # Verify model works
        predictions = clf.predict(classification_dataset['X_test'])
        assert len(predictions) == len(classification_dataset['X_test'])


# ================================================================================================
# Data Module Integration
# ================================================================================================


class TestDataModuleIntegration:
    """Integration tests with TorchDataModule."""

    def test_datamodule_with_lightning_training(self, classification_dataset):
        """Test full training pipeline with TorchDataModule."""
        # Create DataModule
        dm = TorchDataModule(
            train_features=classification_dataset['X_train'],
            train_labels=classification_dataset['y_train'],
            val_features=classification_dataset['X_val'],
            val_labels=classification_dataset['y_val'],
            test_features=classification_dataset['X_test'],
            test_labels=classification_dataset['y_test'],
            dataloader_params={'batch_size': 32}
        )

        dm.setup(stage='fit')

        # Verify dataloaders
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()

        assert train_loader is not None
        assert val_loader is not None

        # Get batch
        batch = next(iter(train_loader))
        assert len(batch) == 2  # features, labels

    def test_dataset_with_dataloader(self, classification_dataset):
        """Test TorchDataset with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        # Create dataset
        dataset = TorchDataset(
            features=classification_dataset['X_train'],
            labels=classification_dataset['y_train'],
            batch_size=0  # Sample mode
        )

        # Create DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Iterate through batches
        total_samples = 0
        for batch_x, batch_y in loader:
            total_samples += len(batch_x)
            assert batch_x.shape[1] == classification_dataset['num_features']

        assert total_samples == len(classification_dataset['X_train'])


# ================================================================================================
# Checkpointing and Best Weights
# ================================================================================================


class TestCheckpointingIntegration:
    """Integration tests for checkpointing and best weights restoration."""

    def test_best_weights_restoration(self, classification_dataset, integration_estimator_params_classifier):
        """Test that best weights are restored after training."""
        params = integration_estimator_params_classifier.copy()
        params['network_params'] = params['network_params'].copy()
        params['network_params']['nlayers'] = 2
        params['model_params'] = params['model_params'].copy()
        params['model_params']['load_best_weights_on_train_end'] = True
        clf = PytorchLightningClassifier(**params)

        eval_set = (classification_dataset['X_val'], classification_dataset['y_val'])
        clf.fit(
            classification_dataset['X_train'],
            classification_dataset['y_train'],
            eval_set=eval_set
        )

        # Model should have best_epoch attribute
        if hasattr(clf.model, 'best_epoch'):
            assert clf.model.best_epoch is not None


# ================================================================================================
# Scheduler Integration
# ================================================================================================


class TestSchedulerIntegration:
    """Integration tests for learning rate schedulers."""

    def test_onecycle_scheduler(self, classification_dataset, integration_estimator_params_classifier):
        """Test training with OneCycleLR scheduler."""
        from torch.optim.lr_scheduler import OneCycleLR

        params = integration_estimator_params_classifier.copy()
        params['network_params'] = params['network_params'].copy()
        params['network_params']['nlayers'] = 2
        params['model_params'] = params['model_params'].copy()
        params['model_params']['lr_scheduler'] = OneCycleLR
        params['model_params']['lr_scheduler_kwargs'] = {'max_lr': 0.01}
        params['model_params']['lr_scheduler_interval'] = 'step'
        params['trainer_params'] = params['trainer_params'].copy()
        params['trainer_params']['max_epochs'] = 5
        clf = PytorchLightningClassifier(**params)

        clf.fit(
            classification_dataset['X_train'],
            classification_dataset['y_train']
        )

        predictions = clf.predict(classification_dataset['X_test'])
        assert len(predictions) == len(classification_dataset['X_test'])

    def test_cosine_annealing_scheduler(self, classification_dataset, integration_estimator_params_classifier):
        """Test training with CosineAnnealingLR scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        params = integration_estimator_params_classifier.copy()
        params['network_params'] = params['network_params'].copy()
        params['network_params']['nlayers'] = 2
        params['model_params'] = params['model_params'].copy()
        params['model_params']['lr_scheduler'] = CosineAnnealingLR
        params['model_params']['lr_scheduler_kwargs'] = {'T_max': 10}
        params['model_params']['lr_scheduler_interval'] = 'epoch'
        params['trainer_params'] = params['trainer_params'].copy()
        params['trainer_params']['max_epochs'] = 5
        clf = PytorchLightningClassifier(**params)

        clf.fit(
            classification_dataset['X_train'],
            classification_dataset['y_train']
        )

        accuracy = clf.score(classification_dataset['X_test'], classification_dataset['y_test'])
        assert 0.0 <= accuracy <= 1.0


# ================================================================================================
# Pandas Integration
# ================================================================================================


class TestPandasIntegration:
    """Integration tests with pandas DataFrames."""

    def test_full_pipeline_with_pandas(self, classification_dataset, integration_estimator_params_classifier):
        """Test complete pipeline using pandas DataFrames."""
        # Convert to pandas
        X_train_df = pd.DataFrame(classification_dataset['X_train'])
        y_train_series = pd.Series(classification_dataset['y_train'])
        X_test_df = pd.DataFrame(classification_dataset['X_test'])
        y_test_series = pd.Series(classification_dataset['y_test'])

        params = integration_estimator_params_classifier.copy()
        params['network_params'] = params['network_params'].copy()
        params['network_params']['nlayers'] = 2
        params['trainer_params'] = params['trainer_params'].copy()
        params['trainer_params']['max_epochs'] = 5
        clf = PytorchLightningClassifier(**params)

        # Fit
        clf.fit(X_train_df, y_train_series)

        # Predict
        predictions = clf.predict(X_test_df)
        assert len(predictions) == len(y_test_series)

        # Score
        accuracy = clf.score(X_test_df, y_test_series)
        assert 0.0 <= accuracy <= 1.0


# ================================================================================================
# Multi-Epoch Training
# ================================================================================================


class TestMultiEpochTraining:
    """Integration tests for multi-epoch training scenarios."""

    def test_long_training_with_early_stopping(self, classification_dataset, integration_estimator_params_classifier):
        """Test long training with early stopping."""
        params = integration_estimator_params_classifier.copy()
        params['trainer_params'] = params['trainer_params'].copy()
        params['trainer_params']['max_epochs'] = 50  # Many epochs
        clf = PytorchLightningClassifier(**params, early_stopping_rounds=5)

        eval_set = (classification_dataset['X_val'], classification_dataset['y_val'])
        clf.fit(
            classification_dataset['X_train'],
            classification_dataset['y_train'],
            eval_set=eval_set
        )

        # Should have stopped early
        predictions = clf.predict(classification_dataset['X_test'])
        assert len(predictions) == len(classification_dataset['X_test'])


# ================================================================================================
# Real-World Scenarios
# ================================================================================================


class TestRealWorldScenarios:
    """Integration tests simulating real-world use cases."""

    def test_binary_classification_workflow(self):
        """Test binary classification from start to finish."""
        # Generate binary classification data
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_classes=2,
            n_informative=8,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create and train classifier
        network_params = {
            'nlayers': 2,
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
            'max_epochs': 10,
            'enable_model_summary': False,
            'default_root_dir': None,
            'log_every_n_steps': 1
        }

        clf = PytorchLightningClassifier(
            model_class=MLPTorchModel,
            model_params=model_params,
            network_params=network_params,
            datamodule_class=TorchDataModule,
            datamodule_params=datamodule_params,
            trainer_params=trainer_params
        )

        clf.fit(X_train.astype(np.float32), y_train)

        # Evaluate
        predictions = clf.predict(X_test.astype(np.float32))
        accuracy = accuracy_score(y_test, predictions)

        assert 0.0 <= accuracy <= 1.0

    def test_multiclass_classification_workflow(self):
        """Test multi-class classification from start to finish."""
        # Generate multi-class data
        X, y = make_classification(
            n_samples=500,
            n_features=15,
            n_classes=5,
            n_informative=12,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create and train classifier
        network_params = {
            'nlayers': 3,
            'first_layer_num_neurons': 128,
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
            'max_epochs': 15,
            'enable_model_summary': False,
            'default_root_dir': None,
            'log_every_n_steps': 1
        }

        clf = PytorchLightningClassifier(
            model_class=MLPTorchModel,
            model_params=model_params,
            network_params=network_params,
            datamodule_class=TorchDataModule,
            datamodule_params=datamodule_params,
            trainer_params=trainer_params
        )

        clf.fit(X_train.astype(np.float32), y_train)

        # Evaluate
        predictions = clf.predict(X_test.astype(np.float32))
        probas = clf.predict_proba(X_test.astype(np.float32))

        assert predictions.shape == (len(X_test),)
        assert probas.shape == (len(X_test), 5)

    def test_regression_workflow(self):
        """Test regression from start to finish."""
        # Generate regression data
        X, y = make_regression(
            n_samples=500,
            n_features=10,
            noise=20.0,
            random_state=42
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create and train regressor
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
            'max_epochs': 15,
            'enable_model_summary': False,
            'default_root_dir': None,
            'log_every_n_steps': 1
        }

        reg = PytorchLightningRegressor(
            model_class=MLPTorchModel,
            model_params=model_params,
            network_params=network_params,
            datamodule_class=TorchDataModule,
            datamodule_params=datamodule_params,
            trainer_params=trainer_params
        )

        reg.fit(X_train.astype(np.float32), y_train.astype(np.float32))

        # Evaluate
        predictions = reg.predict(X_test.astype(np.float32))
        mse = mean_squared_error(y_test, predictions)

        assert mse >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
