"""
Base infrastructure for PyTorch Lightning models in mlframe.

This module provides:
- sklearn-compatible estimator wrappers (PytorchLightningEstimator, Regressor, Classifier)
- Callbacks (BestEpochModelCheckpoint, AggregatingValidationCallback, etc.)
- Utilities (MetricSpec, to_tensor_any, to_numpy_safe)
"""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import os
import operator  # for picklable comparison functions (needed by ddp_spawn strategy)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import psutil
import lightning as L
from pydantic import BaseModel

from lightning import LightningDataModule
from lightning.pytorch.tuner import Tuner

from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ReduceLROnPlateau, LambdaLR

from lightning.pytorch.callbacks import Callback, LearningRateFinder
from lightning.pytorch.callbacks.early_stopping import EarlyStopping as EarlyStoppingCallback
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar, ModelPruning, LearningRateMonitor, LearningRateFinder
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor, StochasticWeightAveraging, GradientAccumulationScheduler
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only

from enum import Enum, auto
from functools import partial
import pandas as pd, numpy as np, polars as pl

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from pyutilz.pythonlib import store_params_in_object, get_parent_func_args
from mlframe.metrics import compute_probabilistic_multiclass_error

from sklearn.metrics import r2_score, accuracy_score, root_mean_squared_error
from copy import deepcopy


# ----------------------------------------------------------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------------------------------------------------------


def _rmse_metric(y_true, y_score):
    """Wrapper for root_mean_squared_error that accepts y_score parameter name."""
    return root_mean_squared_error(y_true=y_true, y_pred=y_score)


# ----------------------------------------------------------------------------------------------------------------------------
# MetricSpec
# ----------------------------------------------------------------------------------------------------------------------------


class MetricSpec(BaseModel):
    name: str
    fcn: Callable  # the metric function
    requires_argmax: bool = False  # True if metric wants predicted class labels
    requires_probs: bool = False  # True if metric wants probabilities (softmax)
    requires_cpu: bool = True  # True if metric should run on CPU (sklearn), False if GPU-compatible


def custom_collate_fn(batch):
    # Return the batch as-is (mimicking lambda x: x)
    return batch


def to_tensor_any(data, dtype=torch.float32, device=None, safe=True):
    """
    Converts pandas / polars / numpy / torch input to a torch.Tensor
    with minimal copies and correct dtype.

    If safe=True, ignores categorical/object columns gracefully.
    """

    # --- Pandas
    if isinstance(data, (pd.DataFrame, pd.Series)):
        data = data.to_numpy()
    # --- Polars
    elif isinstance(data, pl.DataFrame):
        data = data.to_torch()
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)

    return data.to(dtype=dtype, device=device)


def to_numpy_safe(tensor: torch.Tensor, cpu: bool = False) -> np.ndarray:
    """Convert a torch.Tensor to a NumPy array safely and efficiently.

    - Moves tensor to CPU if needed.
    - Converts unsupported dtypes (bfloat16, float16) to float32.
    - Keeps dtype otherwise unchanged (no accidental downcast).
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    # Ensure tensor is detached and on CPU
    t = tensor.detach()
    if cpu and t.device.type != "cpu":
        t = t.cpu()

    # Handle NumPy-incompatible dtypes
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.to(torch.float32)
    elif t.dtype == torch.complex32:
        t = t.to(torch.complex64)

    # Direct conversion is now safe
    return t.numpy()


# ----------------------------------------------------------------------------------------------------------------------------
# Sklearn compatibility
# ----------------------------------------------------------------------------------------------------------------------------


class PytorchLightningEstimator(BaseEstimator):
    """Wrapper that allows Pytorch Lightning model, datamodule and trainer to participate in sklearn pipelines.
    Supports early stopping (via eval_set in fit_params).
    """

    def __init__(
        self,
        model_class: object,
        model_params: dict,
        network_params: dict,
        datamodule_class: object,
        datamodule_params: dict,
        trainer_params: object,
        use_swa: bool = False,
        swa_params: dict = None,
        tune_params: bool = False,
        tune_batch_size: bool = False,
        float32_matmul_precision: str = None,
        early_stopping_rounds: int = 100,
    ):
        # Note: Don't modify swa_params here (e.g., `swa_params or {}`) because sklearn's
        # clone() requires that constructor parameters are not modified. Handle None later.
        store_params_in_object(obj=self, params=get_parent_func_args())

    def _fit_common(
        self,
        X,
        y,
        eval_set: tuple = (None, None),
        is_partial_fit: bool = False,
        classes: Optional[np.ndarray] = None,
        fit_params: dict = None,
        sample_weight=None,
    ):
        """Common logic for fit and partial_fit."""
        # Lazy import to avoid circular dependency
        from .flat import generate_mlp

        if fit_params is None:
            fit_params = {}

        # Enable TF32 for float32 matrix multiplication if on GPU
        if self.float32_matmul_precision and torch.cuda.is_available():
            assert self.float32_matmul_precision in "highest high medium".split()
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision(self.float32_matmul_precision)
                logger.info(f"Enabled float32_matmul_precision={self.float32_matmul_precision}")

        # Check validation availability once
        has_validation = eval_set[0] is not None

        # Extract validation sample weights if provided
        eval_sample_weight = fit_params.get("eval_sample_weight")

        # Create datamodule with sample weights
        dm = self.datamodule_class(
            train_features=X,
            train_labels=y,
            train_sample_weight=sample_weight,  # NEW
            val_features=eval_set[0],
            val_labels=eval_set[1],
            val_sample_weight=eval_sample_weight,  # NEW
            **self.datamodule_params,
        )

        # Set classifier-specific attributes
        if isinstance(self, ClassifierMixin):
            if is_partial_fit and classes is not None:
                self.classes_ = np.array(classes)
            elif not hasattr(self, "classes_"):
                self.classes_ = sorted(np.unique(y) if not isinstance(y, pd.Series) else y.unique())
            num_classes = len(self.classes_)
        else:
            num_classes = 1

        # Reset network on fit() to match sklearn convention (fit resets, partial_fit continues).
        # This ensures each fit() call creates a fresh network with correct input dimensions,
        # which is critical when feature counts change between training iterations.
        if not is_partial_fit:
            self.network = None
            self.model = None  # Also reset the LightningModule wrapper

        # Initialize model if needed (first call to fit or partial_fit)
        # Use getattr to handle freshly cloned models that don't have network attribute yet
        if getattr(self, 'network', None) is None:
            self.network = generate_mlp(num_features=X.shape[1], num_classes=num_classes, **self.network_params)

        # Configure metrics and monitoring
        if num_classes > 1:
            metric_name = "ICE"
            metrics = [MetricSpec(name=metric_name, fcn=compute_probabilistic_multiclass_error, requires_probs=True)]
        else:
            metric_name = "MSE"
            metrics = [MetricSpec(name=metric_name, fcn=_rmse_metric)]

        # Setup checkpointing with appropriate monitor
        # When no validation data, monitor train_loss instead of train metrics (which may not be logged)
        if has_validation:
            monitor_metric = f"val_{metric_name}"
        else:
            monitor_metric = "train_loss"

        checkpointing = BestEpochModelCheckpoint(
            monitor=monitor_metric,
            dirpath=self.trainer_params["default_root_dir"],
            filename=f"model-{{{monitor_metric}:.4f}}",
            enable_version_counter=True,
            save_last=False,
            save_top_k=1,
            mode="min",
        )

        # Configure trainer params - only modify if needed
        trainer_params = self.trainer_params.copy()
        if not has_validation:
            logger.info("No validation data - training without validation")
            trainer_params.update({"num_sanity_val_steps": 0, "limit_val_batches": 0})

        # Set default logger for LearningRateMonitor compatibility
        # Use CSVLogger instead of TensorBoardLogger to avoid TensorFlow dependency
        if "logger" not in trainer_params:
            trainer_params["logger"] = CSVLogger(save_dir="lightning_logs", name="")

        # Build callbacks list
        callbacks = [
            checkpointing,
            TQDMProgressBar(refresh_rate=10),
        ]

        # Only add LearningRateMonitor if logger is enabled
        if trainer_params.get("logger") is not False:
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        if self.use_swa:
            swa_params = self.swa_params or {}
            callbacks.append(StochasticWeightAveraging(**swa_params))

        if has_validation:
            logger.info(f"Using early_stopping_rounds={self.early_stopping_rounds:_}")
            callbacks.append(
                EarlyStoppingCallback(
                    monitor=f"val_{metric_name}",
                    min_delta=0.001,
                    patience=self.early_stopping_rounds,
                    mode="min",
                    verbose=True,
                )
            )

        trainer = L.Trainer(**trainer_params, callbacks=callbacks)

        # Initialize model
        with trainer.init_module():
            self.model = self.model_class(network=self.network, metrics=metrics, **self.model_params)

            features_dtype = self.datamodule_params.get("features_dtype", torch.float32)
            data_slice = X.iloc[0:2, :].values if isinstance(X, pd.DataFrame) else X[0:2, :]

            try:
                self.model.example_input_array = to_tensor_any(data_slice, dtype=features_dtype, safe=True)
            except Exception as e:
                logger.warning(f"Failed to prepare example_input_array: {e}")

        # Tune parameters if requested
        if self.tune_params and not (is_partial_fit and hasattr(self, "_tuned")):
            tuner = Tuner(trainer)

            if self.tune_batch_size:
                tuner.scale_batch_size(model=self.model, datamodule=dm, mode="binsearch", init_val=self.datamodule_params.get("batch_size", 32))

            lr_finder = tuner.lr_find(self.model, datamodule=dm, num_training=300)
            new_lr = lr_finder.suggestion()
            logger.info(f"Using suggested LR={new_lr}")
            self.model.hparams.learning_rate = new_lr

            if is_partial_fit:
                self._tuned = True

        # Train
        trainer.fit(model=self.model, datamodule=dm)

        # Extract best epoch from model (set by checkpoint callback, DDP-safe)
        # Prefer model.best_epoch over callback.best_epoch for distributed training compatibility
        if hasattr(self.model, "best_epoch") and self.model.best_epoch is not None:
            self.best_epoch = self.model.best_epoch
            logger.info(f"Best epoch recorded: {self.best_epoch}")
        else:
            # Fallback to callback for backward compatibility
            for callback in trainer.callbacks:
                if isinstance(callback, BestEpochModelCheckpoint):
                    self.best_epoch = callback.best_epoch
                    if self.best_epoch is not None:
                        logger.info(f"Best epoch recorded from callback: {self.best_epoch}")
                    break

        # Clean up to avoid pickle issues and free memory
        self.trainer = None

        return self

    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit the model to the data.

        Args:
            X: Training features
            y: Training labels
            sample_weight: Optional per-sample weights for training
            **fit_params: Additional parameters including:
                - eval_set: Tuple of (X_val, y_val) for validation
                - eval_sample_weight: Optional validation sample weights
        """
        eval_set = fit_params.get("eval_set", (None, None))
        # Support sample_weight both as parameter and in fit_params
        if sample_weight is None:
            sample_weight = fit_params.get("sample_weight")
        return self._fit_common(X, y, eval_set=eval_set, is_partial_fit=False, fit_params=fit_params, sample_weight=sample_weight)

    def partial_fit(self, X, y, classes: Optional[np.ndarray] = None, sample_weight=None, **fit_params):
        """Incremental training for online learning."""
        eval_set = fit_params.get("eval_set", (None, None))
        if sample_weight is None:
            sample_weight = fit_params.get("sample_weight")
        return self._fit_common(X, y, eval_set=eval_set, is_partial_fit=True, classes=classes, fit_params=fit_params, sample_weight=sample_weight)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Returns a dictionary of all parameters for scikit-learn compatibility.

        All __init__ parameters must be included for sklearn.base.clone() to work correctly.
        """
        params = {
            "model_class": self.model_class,
            "model_params": deepcopy(self.model_params) if deep else self.model_params,
            "network_params": deepcopy(self.network_params) if deep else self.network_params,
            "datamodule_class": self.datamodule_class,
            "datamodule_params": deepcopy(self.datamodule_params) if deep else self.datamodule_params,
            "trainer_params": self.trainer_params,
            "use_swa": self.use_swa,
            "swa_params": deepcopy(self.swa_params) if deep and self.swa_params else self.swa_params,
            "tune_params": self.tune_params,
            "tune_batch_size": self.tune_batch_size,
            "float32_matmul_precision": self.float32_matmul_precision,
            "early_stopping_rounds": self.early_stopping_rounds,
        }
        return params

    def set_params(self, **params: Any) -> "PytorchLightningEstimator":
        """Sets parameters for scikit-learn compatibility."""
        for key, value in params.items():
            if key in ("model_params", "datamodule_params"):
                setattr(self, key, deepcopy(value))  # Deep copy nested dicts
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Parameter {key} not found in {self.__class__.__name__}")
        return self

    def _predict_raw(self, X, device: Optional[str] = None, precision: Optional[str] = None, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Internal method for memory-efficient batched prediction using Lightning's trainer.predict().

        This method processes data in batches to avoid OOM errors, leveraging the existing
        DataModule prediction infrastructure.

        Args:
            X: Input data (numpy array, pandas DataFrame, polars DataFrame, or torch.Tensor)
            device: Optional device string ('cpu' or 'cuda'). If None, uses trainer's device.
            precision: Optional precision mode for inference ('16-mixed', 'bf16-mixed', 'bf16-true', or None).
                       If not provided, falls back to the trainer's precision.
            batch_size: Optional batch size for prediction. If None, uses datamodule's batch_size.
                        Larger batch sizes can speed up prediction but use more memory.

        Returns:
            numpy.ndarray: Model predictions (probabilities for classification, values for regression)
        """

        # Validate that model has been fitted
        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before predict().")

        # Setup prediction data in the datamodule
        if not hasattr(self, "prediction_datamodule") or self.prediction_datamodule is None:
            # Create a minimal datamodule for prediction if not available
            logger.warning("No datamodule found from training. Creating temporary datamodule for prediction.")
            datamodule = self.datamodule_class(**self.datamodule_params)

        # Determine batch size for prediction (allow override for larger batches)
        pred_batch_size = batch_size if batch_size is not None else self.datamodule_params.get("batch_size", 64)
        logger.info(f"Using batch_size={pred_batch_size} for prediction")

        # Setup prediction dataset
        datamodule.setup_predict(X, batch_size=pred_batch_size)

        # Create prediction trainer with appropriate settings
        if not hasattr(self, "trainer") or self.trainer is None:
            # logger.warning("No trainer found from training. Creating temporary trainer for prediction.")
            # Create minimal trainer for prediction
            trainer_params = {
                "accelerator": "auto",
                "devices": 1,
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
            }
            prediction_trainer = L.Trainer(**trainer_params)
        else:
            # Use existing trainer but create a new one to avoid state issues
            trainer_params = {
                "accelerator": (
                    self.trainer.accelerator.__class__.__name__.replace("Accelerator", "").lower() if hasattr(self.trainer, "accelerator") else "auto"
                ),
                "devices": 1,  # Use single device for prediction
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
            }

            # Override device if specified
            if device is not None:
                if device.startswith("cuda"):
                    trainer_params["accelerator"] = "cuda"
                elif device == "cpu":
                    trainer_params["accelerator"] = "cpu"

            # Override precision if specified
            if precision is not None:
                trainer_params["precision"] = precision
            elif hasattr(self.trainer, "precision"):
                trainer_params["precision"] = self.trainer.precision

            prediction_trainer = L.Trainer(**trainer_params)

        # Ensure model is in eval mode and warn if not
        if self.model.training:
            logger.warning("Model was in training mode during prediction. Switching to eval mode.")
            self.model.eval()

        # Additional check for compiled models
        if hasattr(self.model, "_orig_mod"):
            if self.model._orig_mod.training:
                logger.warning("Compiled model's original module was in training mode. Switching to eval mode.")
                self.model._orig_mod.eval()

        # Run prediction using Lightning's batched prediction
        try:
            predictions = prediction_trainer.predict(
                model=self.model,
                datamodule=datamodule,
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

        self.trainer = None

        # Concatenate batch predictions into single array
        if len(predictions) == 0:
            raise RuntimeError("No predictions were generated. Check your data and model.")

        # Handle different return types from predict_step
        if isinstance(predictions[0], torch.Tensor):
            # Direct tensor outputs
            predictions = torch.cat(predictions, dim=0)
            predictions = to_numpy_safe(predictions, cpu=True)
        elif isinstance(predictions[0], np.ndarray):
            # Already numpy arrays
            predictions = np.concatenate(predictions, axis=0)
        else:
            raise TypeError(f"Unexpected prediction type: {type(predictions[0])}")

        logger.info(f"Generated predictions with shape {predictions.shape}")

        return predictions

    def predict(self, X, device: Optional[str] = None, precision: Optional[str] = None, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Predict using the model with memory-efficient batched processing.

        Args:
            X: Input data (numpy array, pandas DataFrame, polars DataFrame, or torch.Tensor)
            device: Optional device string ('cpu' or 'cuda'). If None, uses trainer's device.
            precision: Optional precision mode for inference ('16-mixed', 'bf16-mixed', etc.)
            batch_size: Optional batch size for prediction. Larger values speed up prediction but use more memory.

        Returns:
            numpy.ndarray: Model predictions (class labels for classification, values for regression)
        """
        predictions = self._predict_raw(X, device=device, precision=precision, batch_size=batch_size)

        # For regression, return raw predictions (squeeze to 1D for single-target regression)
        if isinstance(self, RegressorMixin):
            # Squeeze last dimension if it's 1 (single-target regression)
            if predictions.ndim == 2 and predictions.shape[1] == 1:
                predictions = predictions.squeeze(axis=1)
            return predictions

        # For classification in the base class, return probabilities
        # (PytorchLightningClassifier will override to return labels)
        return predictions

    def score(self, X, y, sample_weight: Optional[np.ndarray] = None) -> float:
        """Returns the coefficient of determination R^2 for regression or accuracy for classification."""
        y_pred = self.predict(X)
        if isinstance(self, RegressorMixin):
            return r2_score(y, y_pred, sample_weight=sample_weight)
        elif isinstance(self, ClassifierMixin):
            # y_pred is already class labels from PytorchLightningClassifier.predict()
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        else:
            raise ValueError("Estimator must be a RegressorMixin or ClassifierMixin")


class PytorchLightningRegressor(RegressorMixin, PytorchLightningEstimator):  # RegressorMixin must come first
    _estimator_type = "regressor"


class PytorchLightningClassifier(
    ClassifierMixin,
    PytorchLightningEstimator,
):  # ClassifierMixin must come first
    _estimator_type = "classifier"

    def predict(self, X, device: Optional[str] = None, precision: Optional[str] = None, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: Input data
            device: Optional device string ('cpu' or 'cuda')
            precision: Optional precision mode for inference
            batch_size: Optional batch size for prediction

        Returns:
            numpy.ndarray: Predicted class labels
        """
        proba = self._predict_raw(X, device=device, precision=precision, batch_size=batch_size)
        return np.argmax(proba, axis=1)

    def predict_proba(self, X, device: Optional[str] = None, precision: Optional[str] = None, batch_size: Optional[int] = None) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Args:
            X: Input data
            device: Optional device string ('cpu' or 'cuda')
            precision: Optional precision mode for inference
            batch_size: Optional batch size for prediction

        Returns:
            numpy.ndarray: Predicted class probabilities
        """
        return self._predict_raw(X, device=device, precision=precision, batch_size=batch_size)


# ----------------------------------------------------------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------------------------------------------------------


class NetworkGraphLoggingCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        pl_module.logger.log_graph(model=pl_module)


class AggregatingValidationCallback(Callback):

    def __init__(self, metric_name: str, metric_fcn: object, on_epoch: bool = True, on_step: bool = False, prog_bar: bool = True):
        params = get_parent_func_args()
        store_params_in_object(obj=self, params=params)
        self.init_accumulators()

    def init_accumulators(self):
        self.batched_predictions = []
        self.batched_labels = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        predictions, labels = outputs
        self.batched_labels.append(labels)
        self.batched_predictions.append(predictions)

    def on_validation_epoch_end(self, trainer, pl_module):
        labels = torch.concat(self.batched_labels).detach().cpu().numpy()
        predictions = torch.concat(self.batched_predictions).detach().cpu().float().numpy()
        metric_value = self.metric_fcn(y_true=labels, y_score=predictions)
        pl_module.log(name="val_" + self.metric_name, value=metric_value, on_epoch=self.on_epoch, on_step=self.on_step, prog_bar=True)
        self.init_accumulators()


class BestEpochModelCheckpoint(ModelCheckpoint):
    """
    Custom ModelCheckpoint that tracks the epoch of the best model according to the monitored metric.
    """

    def __init__(self, monitor: str = "val_loss", mode: str = "min", **kwargs):
        super().__init__(monitor=monitor, mode=mode, **kwargs)
        self.best_epoch: Optional[int] = None
        self.best_score: Optional[float] = None

        # Determine comparison operator (using operator module for pickling support)
        if mode == "min":
            self.monitor_op = operator.lt
            self.best_score = float("inf")
        elif mode == "max":
            self.monitor_op = operator.gt
            self.best_score = float("-inf")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        logger.info(f"Initialized BestEpochModelCheckpoint with monitor={monitor}, mode={mode}")

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """
        Update best_epoch after each validation step if current metric improves.
        """
        super().on_validation_end(trainer, pl_module)

        # Get the current value of the monitored metric
        current_score = trainer.callback_metrics.get(self.monitor)

        if current_score is None:
            logger.warning(f"Monitor metric '{self.monitor}' not found in callback_metrics.")
            return

        # Convert to float in case it's a tensor
        if isinstance(current_score, torch.Tensor):
            current_score = current_score.item()

        # Check if it's the new best
        if self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.best_epoch = trainer.current_epoch
            # Also set on pl_module for DDP synchronization
            pl_module.best_epoch = self.best_epoch
            logger.info(f"New best model at epoch {self.best_epoch} with {self.monitor}={self.best_score:.4f}")


class PeriodicLearningRateFinder(LearningRateFinder):
    def __init__(self, period: int, *args, **kwargs):
        assert period > 0 and isinstance(period, int)
        super().__init__(*args, **kwargs)
        self.period = period

    def on_train_epoch_start(self, trainer, pl_module):
        if (trainer.current_epoch % self.period) == 0 or trainer.current_epoch == 0:
            print(f"Finding optimal learning rate. Current rate={getattr(pl_module,'learning_rate')}")
            self.lr_find(trainer, pl_module)
            print(f"Set learning rate to {getattr(pl_module,'learning_rate')}")
