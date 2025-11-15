"""
Tests for callback classes in lightninglib.py

Run tests:
    pytest tests/lightninglib/test_callbacks.py -v
    pytest tests/lightninglib/test_callbacks.py --cov=mlframe.lightninglib --cov-report=html
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
import warnings

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mlframe.lightninglib import (
    NetworkGraphLoggingCallback,
    AggregatingValidationCallback,
    BestEpochModelCheckpoint,
    PeriodicLearningRateFinder,
)


# ================================================================================================
# NetworkGraphLoggingCallback Tests
# ================================================================================================


class TestNetworkGraphLoggingCallback:
    """Tests for NetworkGraphLoggingCallback."""

    def test_initialization(self):
        """Test callback initialization."""
        callback = NetworkGraphLoggingCallback()
        assert callback is not None

    def test_on_train_end_with_logger(self):
        """Test on_train_end logs graph with logger."""
        callback = NetworkGraphLoggingCallback()

        # Mock trainer and pl_module
        mock_trainer = Mock()

        # The callback uses pl_module.logger, not trainer.logger
        mock_logger = Mock()
        mock_module = Mock()
        mock_module.logger = mock_logger
        mock_module.example_input_array = torch.randn(1, 10)

        # Call the method
        callback.on_train_end(mock_trainer, mock_module)

        # Verify pl_module.logger.log_graph was called (actual implementation calls model=pl_module, not with example_input)
        mock_logger.log_graph.assert_called_once_with(model=mock_module)

    def test_on_train_end_without_logger(self):
        """Test on_train_end handles missing logger gracefully."""
        callback = NetworkGraphLoggingCallback()

        mock_trainer = Mock()
        mock_trainer.logger = None

        mock_module = Mock()

        # Should not crash
        callback.on_train_end(mock_trainer, mock_module)


# ================================================================================================
# AggregatingValidationCallback Tests
# ================================================================================================


class TestAggregatingValidationCallback:
    """Tests for AggregatingValidationCallback."""

    def test_initialization(self):
        """Test callback initialization."""
        def dummy_metric(y_true, y_score):
            return 0.5

        callback = AggregatingValidationCallback(
            metric_name='accuracy',
            metric_fcn=dummy_metric,
            on_epoch=True,
            on_step=False
        )

        assert callback.metric_name == 'accuracy'
        assert callback.metric_fcn == dummy_metric
        assert callback.on_epoch is True
        assert callback.on_step is False
        assert callback.batched_predictions == []
        assert callback.batched_labels == []

    def test_init_accumulators(self):
        """Test init_accumulators resets lists."""
        def dummy_metric(y_true, y_score):
            return 0.5

        callback = AggregatingValidationCallback(
            metric_name='test',
            metric_fcn=dummy_metric
        )

        # Add some data
        callback.batched_predictions.append(torch.tensor([1, 2, 3]))
        callback.batched_labels.append(torch.tensor([0, 1, 0]))

        # Reset
        callback.init_accumulators()

        assert callback.batched_predictions == []
        assert callback.batched_labels == []

    def test_on_validation_batch_end(self):
        """Test on_validation_batch_end appends predictions and labels."""
        def dummy_metric(y_true, y_score):
            return 0.5

        callback = AggregatingValidationCallback(
            metric_name='test',
            metric_fcn=dummy_metric
        )

        # Mock outputs - callback expects TUPLE, not dict
        predictions = torch.tensor([0.1, 0.9, 0.3])
        labels = torch.tensor([0, 1, 0])
        outputs = (predictions, labels)

        callback.on_validation_batch_end(None, None, outputs, None, None)

        assert len(callback.batched_predictions) == 1
        assert len(callback.batched_labels) == 1
        assert torch.equal(callback.batched_predictions[0], predictions)
        assert torch.equal(callback.batched_labels[0], labels)

    def test_on_validation_batch_end_multiple_batches(self):
        """Test accumulating multiple batches."""
        def dummy_metric(y_true, y_score):
            return 0.5

        callback = AggregatingValidationCallback(
            metric_name='test',
            metric_fcn=dummy_metric
        )

        # Add multiple batches - callback expects TUPLE, not dict
        for i in range(3):
            predictions = torch.tensor([0.1, 0.9])
            labels = torch.tensor([0, 1])
            outputs = (predictions, labels)
            callback.on_validation_batch_end(None, None, outputs, None, None)

        assert len(callback.batched_predictions) == 3
        assert len(callback.batched_labels) == 3

    def test_on_validation_epoch_end(self):
        """Test on_validation_epoch_end computes and logs metric."""
        def dummy_metric(y_true, y_score):
            # Return accuracy
            return float((y_true == (y_score > 0.5)).sum()) / len(y_true)

        callback = AggregatingValidationCallback(
            metric_name='accuracy',
            metric_fcn=dummy_metric,
            on_epoch=True,
            on_step=False
        )

        # Add batches
        callback.batched_predictions.append(torch.tensor([0.1, 0.9, 0.3, 0.8]))
        callback.batched_labels.append(torch.tensor([0, 1, 0, 1]))

        # Mock pl_module
        mock_module = Mock()

        callback.on_validation_epoch_end(None, mock_module)

        # Verify log was called with correct prefix
        mock_module.log.assert_called_once()
        args, kwargs = mock_module.log.call_args
        assert kwargs['name'] == 'val_accuracy'
        assert isinstance(kwargs['value'], float)
        assert kwargs['on_epoch'] is True
        assert kwargs['on_step'] is False
        assert kwargs['prog_bar'] is True

    def test_on_validation_epoch_end_resets_accumulators(self):
        """Test that on_validation_epoch_end resets accumulators."""
        def dummy_metric(y_true, y_score):
            return 0.5

        callback = AggregatingValidationCallback(
            metric_name='test',
            metric_fcn=dummy_metric
        )

        # Add batches
        callback.batched_predictions.append(torch.tensor([0.1, 0.9]))
        callback.batched_labels.append(torch.tensor([0, 1]))

        mock_module = Mock()

        callback.on_validation_epoch_end(None, mock_module)

        # Accumulators should be reset
        assert callback.batched_predictions == []
        assert callback.batched_labels == []

    def test_on_validation_epoch_end_concatenates_batches(self):
        """Test that batches are concatenated correctly."""
        captured_labels = []
        captured_predictions = []

        def capture_metric(y_true, y_score):
            captured_labels.append(y_true)
            captured_predictions.append(y_score)
            return 0.5

        callback = AggregatingValidationCallback(
            metric_name='test',
            metric_fcn=capture_metric
        )

        # Add multiple batches
        callback.batched_predictions.append(torch.tensor([0.1, 0.2]))
        callback.batched_predictions.append(torch.tensor([0.3, 0.4]))
        callback.batched_labels.append(torch.tensor([0, 1]))
        callback.batched_labels.append(torch.tensor([1, 0]))

        mock_module = Mock()

        callback.on_validation_epoch_end(None, mock_module)

        # Check concatenation
        assert len(captured_labels) == 1
        assert len(captured_labels[0]) == 4  # 2 + 2


# ================================================================================================
# BestEpochModelCheckpoint Tests
# ================================================================================================


class TestBestEpochModelCheckpoint:
    """Tests for BestEpochModelCheckpoint."""

    def test_initialization_mode_min(self):
        """Test initialization with mode='min'."""
        callback = BestEpochModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath='checkpoints',
            filename='best'
        )

        assert callback.best_epoch is None
        assert callback.best_score == float('inf')
        assert callback.monitor_op(1, 2) is True  # 1 < 2
        assert callback.monitor_op(2, 1) is False

    def test_initialization_mode_max(self):
        """Test initialization with mode='max'."""
        callback = BestEpochModelCheckpoint(
            monitor='val_acc',
            mode='max',
            dirpath='checkpoints',
            filename='best'
        )

        assert callback.best_epoch is None
        assert callback.best_score == float('-inf')
        assert callback.monitor_op(2, 1) is True  # 2 > 1
        assert callback.monitor_op(1, 2) is False

    def test_initialization_invalid_mode_raises_error(self):
        """Test that invalid mode raises exception."""
        # Lightning's ModelCheckpoint validates mode in parent class
        from lightning.fabric.utilities.exceptions import MisconfigurationException
        with pytest.raises(MisconfigurationException, match="mode"):
            BestEpochModelCheckpoint(
                monitor='val_loss',
                mode='invalid',
                dirpath='checkpoints',
                filename='best'
            )

    @patch('mlframe.lightninglib.logger')
    def test_initialization_logs_message(self, mock_logger):
        """Test that initialization logs a message."""
        callback = BestEpochModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath='checkpoints',
            filename='best'
        )

        # Check that logger.info was called
        mock_logger.info.assert_called()

    def test_on_validation_end_first_epoch(self):
        """Test on_validation_end on first epoch (always improves)."""
        callback = BestEpochModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath='checkpoints',
            filename='best'
        )

        # Mock trainer and pl_module
        mock_trainer = Mock()
        mock_trainer.current_epoch = 0
        mock_trainer.callback_metrics = {'val_loss': torch.tensor(0.5)}

        mock_module = Mock()

        with patch('builtins.print'):
            callback.on_validation_end(mock_trainer, mock_module)

        assert callback.best_epoch == 0
        assert callback.best_score == 0.5

    def test_on_validation_end_improvement(self):
        """Test on_validation_end when metric improves."""
        callback = BestEpochModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath='checkpoints',
            filename='best'
        )

        callback.best_epoch = 0
        callback.best_score = 0.5

        # Mock trainer with improved metric
        mock_trainer = Mock()
        mock_trainer.current_epoch = 1
        mock_trainer.callback_metrics = {'val_loss': torch.tensor(0.3)}

        mock_module = Mock()

        with patch('builtins.print'):
            callback.on_validation_end(mock_trainer, mock_module)

        assert callback.best_epoch == 1
        assert pytest.approx(callback.best_score, abs=1e-5) == 0.3

    def test_on_validation_end_no_improvement(self):
        """Test on_validation_end when metric doesn't improve."""
        callback = BestEpochModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath='checkpoints',
            filename='best'
        )

        callback.best_epoch = 0
        callback.best_score = 0.3

        # Mock trainer with worse metric
        mock_trainer = Mock()
        mock_trainer.current_epoch = 1
        mock_trainer.callback_metrics = {'val_loss': torch.tensor(0.5)}

        mock_module = Mock()

        with patch('builtins.print'):
            callback.on_validation_end(mock_trainer, mock_module)

        # Should not update
        assert callback.best_epoch == 0
        assert callback.best_score == 0.3

    def test_on_validation_end_metric_not_found(self):
        """Test on_validation_end when monitor metric not in callback_metrics."""
        callback = BestEpochModelCheckpoint(
            monitor='val_loss',
            mode='min',
            dirpath='checkpoints',
            filename='best'
        )

        # Mock trainer without the monitored metric
        mock_trainer = Mock()
        mock_trainer.current_epoch = 0
        mock_trainer.callback_metrics = {'other_metric': torch.tensor(0.5)}

        mock_module = Mock()

        with patch('mlframe.lightninglib.logger.warning') as mock_warn:
            callback.on_validation_end(mock_trainer, mock_module)

        # Should issue warning via logger
        mock_warn.assert_called_once()
        assert callback.best_epoch is None  # Should not update

    def test_on_validation_end_tensor_conversion(self):
        """Test that tensor metrics are converted to float."""
        callback = BestEpochModelCheckpoint(
            monitor='val_acc',
            mode='max',
            dirpath='checkpoints',
            filename='best'
        )

        # Mock trainer with tensor metric
        mock_trainer = Mock()
        mock_trainer.current_epoch = 0
        mock_trainer.callback_metrics = {'val_acc': torch.tensor(0.85)}

        mock_module = Mock()

        with patch('builtins.print'):
            callback.on_validation_end(mock_trainer, mock_module)

        # best_score should be float, not tensor
        assert isinstance(callback.best_score, float)
        assert pytest.approx(callback.best_score, abs=1e-5) == 0.85

    def test_on_validation_end_mode_max(self):
        """Test on_validation_end with mode='max'."""
        callback = BestEpochModelCheckpoint(
            monitor='val_acc',
            mode='max',
            dirpath='checkpoints',
            filename='best'
        )

        callback.best_epoch = 0
        callback.best_score = 0.8

        # Mock trainer with improved metric (higher is better)
        mock_trainer = Mock()
        mock_trainer.current_epoch = 1
        mock_trainer.callback_metrics = {'val_acc': torch.tensor(0.9)}

        mock_module = Mock()

        with patch('builtins.print'):
            callback.on_validation_end(mock_trainer, mock_module)

        assert callback.best_epoch == 1
        assert pytest.approx(callback.best_score, abs=1e-5) == 0.9


# ================================================================================================
# PeriodicLearningRateFinder Tests
# ================================================================================================


class TestPeriodicLearningRateFinder:
    """Tests for PeriodicLearningRateFinder."""

    def test_initialization(self):
        """Test callback initialization."""
        callback = PeriodicLearningRateFinder(period=5)

        assert callback.period == 5

    def test_initialization_with_kwargs(self):
        """Test initialization with additional kwargs."""
        callback = PeriodicLearningRateFinder(
            period=3,
            min_lr=1e-8,
            max_lr=1.0
        )

        assert callback.period == 3

    def test_on_train_epoch_start_epoch_zero(self):
        """Test that LR finder runs at epoch 0."""
        callback = PeriodicLearningRateFinder(period=5)

        # Mock lr_find method
        callback.lr_find = Mock()

        # Mock trainer and module
        mock_trainer = Mock()
        mock_trainer.current_epoch = 0

        mock_module = Mock()
        mock_module.learning_rate = 0.001

        with patch('builtins.print'):
            callback.on_train_epoch_start(mock_trainer, mock_module)

        # lr_find should be called at epoch 0
        callback.lr_find.assert_called_once_with(mock_trainer, mock_module)

    def test_on_train_epoch_start_at_period(self):
        """Test that LR finder runs at period intervals."""
        callback = PeriodicLearningRateFinder(period=3)

        callback.lr_find = Mock()

        # Epoch 3 (3 % 3 == 0)
        mock_trainer = Mock()
        mock_trainer.current_epoch = 3

        mock_module = Mock()
        mock_module.learning_rate = 0.001

        with patch('builtins.print'):
            callback.on_train_epoch_start(mock_trainer, mock_module)

        callback.lr_find.assert_called_once()

    def test_on_train_epoch_start_not_at_period(self):
        """Test that LR finder doesn't run between periods."""
        callback = PeriodicLearningRateFinder(period=5)

        callback.lr_find = Mock()

        # Epoch 3 (3 % 5 != 0)
        mock_trainer = Mock()
        mock_trainer.current_epoch = 3

        mock_module = Mock()
        mock_module.learning_rate = 0.001

        callback.on_train_epoch_start(mock_trainer, mock_module)

        # lr_find should NOT be called
        callback.lr_find.assert_not_called()

    def test_on_train_epoch_start_multiple_periods(self):
        """Test LR finder at multiple period intervals."""
        callback = PeriodicLearningRateFinder(period=2)

        callback.lr_find = Mock()

        mock_module = Mock()
        mock_module.learning_rate = 0.001

        epochs_to_test = [0, 2, 4, 6]
        for epoch in epochs_to_test:
            mock_trainer = Mock()
            mock_trainer.current_epoch = epoch

            with patch('builtins.print'):
                callback.on_train_epoch_start(mock_trainer, mock_module)

        # Should be called 4 times (epochs 0, 2, 4, 6)
        assert callback.lr_find.call_count == 4

    def test_on_train_epoch_start_prints_messages(self):
        """Test that on_train_epoch_start prints messages."""
        callback = PeriodicLearningRateFinder(period=5)

        callback.lr_find = Mock()

        mock_trainer = Mock()
        mock_trainer.current_epoch = 0

        mock_module = Mock()
        mock_module.learning_rate = 0.001

        with patch('builtins.print') as mock_print:
            callback.on_train_epoch_start(mock_trainer, mock_module)

        # Should print 2 messages (before and after)
        assert mock_print.call_count == 2

    def test_period_one(self):
        """Test with period=1 (every epoch)."""
        callback = PeriodicLearningRateFinder(period=1)

        callback.lr_find = Mock()

        mock_module = Mock()
        mock_module.learning_rate = 0.001

        # Test epochs 0-5
        for epoch in range(6):
            mock_trainer = Mock()
            mock_trainer.current_epoch = epoch

            with patch('builtins.print'):
                callback.on_train_epoch_start(mock_trainer, mock_module)

        # Should be called 6 times (every epoch)
        assert callback.lr_find.call_count == 6

    def test_large_period(self):
        """Test with large period."""
        callback = PeriodicLearningRateFinder(period=100)

        callback.lr_find = Mock()

        mock_module = Mock()
        mock_module.learning_rate = 0.001

        # Test epochs 0, 50, 99, 100
        for epoch in [0, 50, 99, 100]:
            mock_trainer = Mock()
            mock_trainer.current_epoch = epoch

            with patch('builtins.print'):
                callback.on_train_epoch_start(mock_trainer, mock_module)

        # Should be called 2 times (epochs 0 and 100)
        assert callback.lr_find.call_count == 2


# ================================================================================================
# Integration Tests
# ================================================================================================


class TestCallbacksIntegration:
    """Integration tests for callbacks."""

    def test_multiple_callbacks_together(self):
        """Test using multiple callbacks together."""
        def dummy_metric(y_true, y_score):
            return 0.5

        callbacks = [
            NetworkGraphLoggingCallback(),
            AggregatingValidationCallback(
                metric_name='acc',
                metric_fcn=dummy_metric
            ),
            BestEpochModelCheckpoint(
                monitor='val_loss',
                mode='min',
                dirpath='checkpoints',
                filename='best'
            )
        ]

        # All should be instantiated without errors
        assert len(callbacks) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
