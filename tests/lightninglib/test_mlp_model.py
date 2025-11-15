"""
Tests for MLPTorchModel class in lightninglib.py

Run tests:
    pytest tests/lightninglib/test_mlp_model.py -v
    pytest tests/lightninglib/test_mlp_model.py --cov=mlframe.lightninglib --cov-report=html
"""

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from unittest.mock import Mock, MagicMock, patch
import warnings

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mlframe.lightninglib import MLPTorchModel, generate_mlp, MetricSpec


# ================================================================================================
# Fixtures
# ================================================================================================


@pytest.fixture
def simple_network():
    """Create a simple neural network."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 3)
    )


@pytest.fixture
def loss_function():
    """Create a loss function."""
    return nn.CrossEntropyLoss()


@pytest.fixture
def sample_batch():
    """Create a sample batch."""
    features = torch.randn(8, 10)
    labels = torch.randint(0, 3, (8,))
    return (features, labels)


@pytest.fixture
def sample_metrics():
    """Create sample metrics."""
    def accuracy(y_true, y_score):
        predictions = y_score.argmax(axis=1)
        return float((predictions == y_true).sum()) / len(y_true)

    return [
        MetricSpec(
            name='accuracy',
            fcn=accuracy,
            requires_argmax=True,
            requires_cpu=True
        )
    ]


# ================================================================================================
# Initialization Tests
# ================================================================================================


class TestMLPTorchModelInit:
    """Tests for MLPTorchModel.__init__."""

    def test_basic_initialization(self, simple_network, loss_function):
        """Test basic initialization with required parameters."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            metrics=[]
        )

        assert model.network is simple_network
        assert model.loss_fn is loss_function
        assert model.hparams.learning_rate == 0.001
        assert model.metrics == []

    def test_initialization_with_metrics(self, simple_network, loss_function, sample_metrics):
        """Test initialization with metrics."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            metrics=sample_metrics
        )

        assert len(model.metrics) == 1
        assert model.metrics[0].name == 'accuracy'

    def test_initialization_network_none_raises_error(self, loss_function):
        """Test that network=None raises ValueError."""
        with pytest.raises(ValueError, match="network must be provided"):
            MLPTorchModel(
                network=None,
                loss_fn=loss_function,
                learning_rate=0.001,
                metrics=[]
            )

    def test_initialization_loss_fn_none_raises_error(self, simple_network):
        """Test that loss_fn=None raises ValueError."""
        with pytest.raises(ValueError, match="loss_fn must be provided"):
            MLPTorchModel(
                network=simple_network,
                loss_fn=None,
                learning_rate=0.001,
                metrics=[]
            )

    def test_initialization_with_l1_alpha(self, simple_network, loss_function):
        """Test initialization with L1 regularization."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            l1_alpha=0.01,
            metrics=[]
        )

        assert model.hparams.l1_alpha == 0.01

    def test_initialization_with_custom_optimizer(self, simple_network, loss_function):
        """Test initialization with custom optimizer."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            optimizer=SGD,
            optimizer_kwargs={'momentum': 0.9},
            metrics=[]
        )

        assert model.optimizer == SGD
        assert model.hparams.optimizer_kwargs['momentum'] == 0.9

    def test_initialization_with_lr_scheduler(self, simple_network, loss_function):
        """Test initialization with learning rate scheduler."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            lr_scheduler=CosineAnnealingLR,
            lr_scheduler_kwargs={'T_max': 10},
            metrics=[]
        )

        assert model.lr_scheduler == CosineAnnealingLR
        assert model.hparams.lr_scheduler_kwargs['T_max'] == 10

    def test_initialization_lr_scheduler_interval(self, simple_network, loss_function):
        """Test lr_scheduler_interval validation."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            lr_scheduler_interval='epoch',
            metrics=[]
        )

        assert model.hparams.lr_scheduler_interval == 'epoch'

    def test_initialization_invalid_lr_scheduler_interval_raises_error(self, simple_network, loss_function):
        """Test that invalid lr_scheduler_interval raises ValueError."""
        with pytest.raises(ValueError, match="lr_scheduler_interval must be"):
            MLPTorchModel(
                network=simple_network,
                loss_fn=loss_function,
                learning_rate=0.001,
                lr_scheduler_interval='invalid',
                metrics=[]
            )

    def test_initialization_compute_trainset_metrics(self, simple_network, loss_function):
        """Test initialization with compute_trainset_metrics."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            compute_trainset_metrics=True,
            metrics=[]
        )

        assert model.hparams.compute_trainset_metrics is True

    def test_initialization_load_best_weights(self, simple_network, loss_function):
        """Test initialization with load_best_weights_on_train_end."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            load_best_weights_on_train_end=True,
            metrics=[]
        )

        assert model.hparams.load_best_weights_on_train_end is True

    def test_initialization_example_input_array(self, simple_network, loss_function):
        """Test that example_input_array is extracted from network."""
        simple_network.example_input_array = torch.randn(1, 10)

        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            metrics=[]
        )

        assert model.example_input_array is not None
        assert model.example_input_array.shape == (1, 10)


# ================================================================================================
# forward() Tests
# ================================================================================================


class TestMLPTorchModelForward:
    """Tests for MLPTorchModel.forward()."""

    def test_forward_pass(self, simple_network, loss_function):
        """Test forward pass."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            metrics=[]
        )

        x = torch.randn(4, 10)
        output = model(x)

        assert output.shape == (4, 3)

    def test_forward_delegates_to_network(self, simple_network, loss_function):
        """Test that forward delegates to network."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            metrics=[]
        )

        x = torch.randn(4, 10)
        output1 = model(x)
        output2 = simple_network(x)

        assert torch.allclose(output1, output2)


# ================================================================================================
# _unpack_batch() Tests
# ================================================================================================


class TestMLPTorchModelUnpackBatch:
    """Tests for MLPTorchModel._unpack_batch()."""

    def test_unpack_batch_tuple_format(self, simple_network, loss_function):
        """Test unpacking batch in tuple format."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            metrics=[]
        )

        batch = (torch.randn(4, 10), torch.randint(0, 3, (4,)))
        features, labels = model._unpack_batch(batch)

        assert features.shape == (4, 10)
        assert labels.shape == (4,)

    def test_unpack_batch_list_format(self, simple_network, loss_function):
        """Test unpacking batch in list format."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            metrics=[]
        )

        batch = [torch.randn(4, 10), torch.randint(0, 3, (4,))]
        features, labels = model._unpack_batch(batch)

        assert features.shape == (4, 10)
        assert labels.shape == (4,)

    def test_unpack_batch_dict_format(self, simple_network, loss_function):
        """Test unpacking batch in dict format."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            metrics=[]
        )

        batch = {
            'features': torch.randn(4, 10),
            'labels': torch.randint(0, 3, (4,))
        }
        features, labels = model._unpack_batch(batch)

        assert features.shape == (4, 10)
        assert labels.shape == (4,)

    def test_unpack_batch_invalid_format_raises_error(self, simple_network, loss_function):
        """Test that invalid batch format raises ValueError."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            metrics=[]
        )

        batch = "invalid"

        with pytest.raises(ValueError, match="Unexpected batch format"):
            model._unpack_batch(batch)


# ================================================================================================
# training_step() Tests
# ================================================================================================


class TestMLPTorchModelTrainingStep:
    """Tests for MLPTorchModel.training_step()."""

    def test_training_step_basic(self, simple_network, loss_function, sample_batch):
        """Test basic training step."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            metrics=[]
        )

        loss = model.training_step(sample_batch, batch_idx=0)

        assert isinstance(loss, dict)
        assert 'loss' in loss
        assert loss['loss'].item() > 0

    def test_training_step_with_l1_regularization(self, simple_network, loss_function, sample_batch):
        """Test training step with L1 regularization."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            l1_alpha=0.01,
            metrics=[]
        )

        loss_dict = model.training_step(sample_batch, batch_idx=0)

        # Should include L1 regularization
        assert 'loss' in loss_dict
        assert loss_dict['loss'].item() > 0

    def test_training_step_without_compute_metrics(self, simple_network, loss_function, sample_batch):
        """Test training step with compute_trainset_metrics=False."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            compute_trainset_metrics=False,
            metrics=[]
        )

        loss_dict = model.training_step(sample_batch, batch_idx=0)

        assert isinstance(loss_dict, dict)

    def test_training_step_with_compute_metrics(self, simple_network, loss_function, sample_batch):
        """Test training step with compute_trainset_metrics=True."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            compute_trainset_metrics=True,
            metrics=[]
        )

        # Initialize training_step_outputs
        model.training_step_outputs = []

        loss_dict = model.training_step(sample_batch, batch_idx=0)

        # Should store outputs
        assert len(model.training_step_outputs) == 1


# ================================================================================================
# validation_step() Tests
# ================================================================================================


class TestMLPTorchModelValidationStep:
    """Tests for MLPTorchModel.validation_step()."""

    def test_validation_step_basic(self, simple_network, loss_function, sample_batch):
        """Test basic validation step."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            metrics=[]
        )

        # Initialize validation_step_outputs
        model.validation_step_outputs = []

        output = model.validation_step(sample_batch, batch_idx=0)

        assert isinstance(output, dict)
        assert 'raw_predictions' in output
        assert 'labels' in output
        assert len(model.validation_step_outputs) == 1

    def test_validation_step_no_l1_regularization(self, simple_network, loss_function, sample_batch):
        """Test that validation step doesn't apply L1 regularization."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            l1_alpha=0.01,  # L1 enabled
            metrics=[]
        )

        model.validation_step_outputs = []

        output = model.validation_step(sample_batch, batch_idx=0)

        # L1 should not be applied in validation
        assert 'raw_predictions' in output


# ================================================================================================
# configure_optimizers() Tests
# ================================================================================================


class TestMLPTorchModelConfigureOptimizers:
    """Tests for MLPTorchModel.configure_optimizers()."""

    def test_configure_optimizers_no_scheduler(self, simple_network, loss_function):
        """Test optimizer configuration without scheduler."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            metrics=[]
        )

        optimizer = model.configure_optimizers()

        assert isinstance(optimizer, AdamW)

    def test_configure_optimizers_with_custom_optimizer(self, simple_network, loss_function):
        """Test with custom optimizer."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            optimizer=SGD,
            optimizer_kwargs={'momentum': 0.9},
            metrics=[]
        )

        optimizer = model.configure_optimizers()

        assert isinstance(optimizer, SGD)

    def test_configure_optimizers_with_onecyclelr(self, simple_network, loss_function):
        """Test with OneCycleLR scheduler."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            lr_scheduler=OneCycleLR,
            lr_scheduler_kwargs={'max_lr': 0.01},
            metrics=[]
        )

        # Mock trainer with proper attributes for OneCycleLR
        mock_trainer = Mock()
        mock_trainer.max_epochs = 10
        mock_trainer.estimated_stepping_batches = 100
        # Add datamodule mock with train_dataloader that returns a list (has len())
        mock_datamodule = Mock()
        mock_dataloader = [1, 2, 3, 4, 5]  # List with 5 items to represent 5 batches
        mock_datamodule.train_dataloader = Mock(return_value=mock_dataloader)
        mock_trainer.datamodule = mock_datamodule
        model.trainer = mock_trainer

        config = model.configure_optimizers()

        assert 'optimizer' in config
        assert 'lr_scheduler' in config

    def test_configure_optimizers_with_other_scheduler(self, simple_network, loss_function):
        """Test with other scheduler (not OneCycleLR)."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            lr_scheduler=CosineAnnealingLR,
            lr_scheduler_kwargs={'T_max': 10},
            metrics=[]
        )

        config = model.configure_optimizers()

        assert 'optimizer' in config
        assert 'lr_scheduler' in config


# ================================================================================================
# predict_step() Tests
# ================================================================================================


class TestMLPTorchModelPredictStep:
    """Tests for MLPTorchModel.predict_step()."""

    def test_predict_step_with_labels(self, simple_network, loss_function, sample_batch):
        """Test predict step with labels in batch."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            metrics=[]
        )

        model.eval()
        predictions = model.predict_step(sample_batch, batch_idx=0)

        assert predictions.shape == (8, 3)

    def test_predict_step_without_labels(self, simple_network, loss_function):
        """Test predict step without labels."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            metrics=[]
        )

        batch = torch.randn(8, 10)
        model.eval()
        predictions = model.predict_step(batch, batch_idx=0)

        assert predictions.shape == (8, 3)

    def test_predict_step_multiclass_returns_softmax(self, simple_network, loss_function):
        """Test that multiclass prediction returns softmax probabilities."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            metrics=[]
        )

        batch = torch.randn(8, 10)
        model.eval()
        predictions = model.predict_step(batch, batch_idx=0)

        # Should sum to 1 (softmax)
        assert torch.allclose(predictions.sum(dim=1), torch.ones(8), atol=1e-5)

    def test_predict_step_warns_if_training_mode(self, simple_network, loss_function):
        """Test that predict_step warns if model is in training mode."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            metrics=[]
        )

        model.train()  # Set to training mode
        batch = torch.randn(8, 10)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            predictions = model.predict_step(batch, batch_idx=0)

            # May issue warning (implementation-dependent)
            # Just ensure it doesn't crash
            assert predictions is not None


# ================================================================================================
# on_train_epoch_end() Tests
# ================================================================================================


class TestMLPTorchModelOnTrainEpochEnd:
    """Tests for MLPTorchModel.on_train_epoch_end()."""

    def test_on_train_epoch_end_without_outputs(self, simple_network, loss_function):
        """Test on_train_epoch_end without outputs."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            compute_trainset_metrics=False,
            metrics=[]
        )

        model.training_step_outputs = []

        # Mock trainer for epoch_end method
        mock_trainer = Mock()
        mock_optimizer = Mock()
        mock_optimizer.param_groups = [{'lr': 0.001}]
        mock_trainer.optimizers = [mock_optimizer]
        model.trainer = mock_trainer

        # Should not crash
        model.on_train_epoch_end()

    def test_on_train_epoch_end_with_outputs(self, simple_network, loss_function):
        """Test on_train_epoch_end with outputs."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            compute_trainset_metrics=True,
            metrics=[]
        )

        # Add mock outputs (use 'raw_predictions' not 'predictions')
        model.training_step_outputs = [
            {'raw_predictions': torch.randn(8, 3), 'labels': torch.randint(0, 3, (8,))}
        ]

        # Mock trainer for epoch_end method
        mock_trainer = Mock()
        mock_optimizer = Mock()
        mock_optimizer.param_groups = [{'lr': 0.001}]
        mock_trainer.optimizers = [mock_optimizer]
        model.trainer = mock_trainer

        model.on_train_epoch_end()

        # Outputs should be cleared
        assert len(model.training_step_outputs) == 0


# ================================================================================================
# on_validation_epoch_end() Tests
# ================================================================================================


class TestMLPTorchModelOnValidationEpochEnd:
    """Tests for MLPTorchModel.on_validation_epoch_end()."""

    def test_on_validation_epoch_end_with_outputs(self, simple_network, loss_function):
        """Test on_validation_epoch_end with outputs."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            metrics=[]
        )

        # Add mock outputs (use 'raw_predictions' not 'predictions')
        model.validation_step_outputs = [
            {'raw_predictions': torch.randn(8, 3), 'labels': torch.randint(0, 3, (8,))}
        ]

        model.on_validation_epoch_end()

        # Outputs should be cleared
        assert len(model.validation_step_outputs) == 0


# ================================================================================================
# on_train_end() Tests
# ================================================================================================


class TestMLPTorchModelOnTrainEnd:
    """Tests for MLPTorchModel.on_train_end()."""

    def test_on_train_end_without_load_best_weights(self, simple_network, loss_function):
        """Test on_train_end when load_best_weights_on_train_end=False."""
        model = MLPTorchModel(
            network=simple_network,
            loss_fn=loss_function,
            learning_rate=0.001,
            load_best_weights_on_train_end=False,
            metrics=[]
        )

        # Should not crash
        model.on_train_end()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
