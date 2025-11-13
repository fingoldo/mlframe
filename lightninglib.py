"""Everything that makes working with Pytorch Lightning easier."""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

# from pyutilz.pythonlib import ensure_installed;ensure_installed("torch torchvision torchaudio lightning pandas scikit-learn")

import os
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

from enum import Enum, auto
from functools import partial
import pandas as pd, numpy as np, polars as pl

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from pyutilz.pythonlib import store_params_in_object, get_parent_func_args
from mlframe.metrics import compute_probabilistic_multiclass_error

from sklearn.metrics import r2_score, accuracy_score, root_mean_squared_error
from copy import deepcopy

# ----------------------------------------------------------------------------------------------------------------------------
# ENUMS
# ----------------------------------------------------------------------------------------------------------------------------


class MLPNeuronsByLayerArchitecture(Enum):
    Constant = auto()
    Declining = auto()
    Expanding = auto()
    ExpandingThenDeclining = auto()
    Autoencoder = auto()


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
        swa_params = swa_params or {}
        store_params_in_object(obj=self, params=get_parent_func_args())

    def _fit_common(self, X, y, eval_set: tuple = (None, None), is_partial_fit: bool = False, classes: Optional[np.ndarray] = None, fit_params: dict = None):
        """Common logic for fit and partial_fit."""

        if fit_params is None:
            fit_params = {}

        # Enable TF32 for float32 matrix multiplication if on GPU
        if self.float32_matmul_precision and torch.cuda.is_available():
            assert self.float32_matmul_precision in "highest high medium".split()
            if self.float32_matmul_precision and hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision(self.float32_matmul_precision)
                logger.info(f"Enabled float32_matmul_precision={self.float32_matmul_precision} for float32 matrix multiplication to improve performance on GPU")

        # Create datamodule
        dm = self.datamodule_class(
            train_features=X,
            train_labels=y,
            val_features=eval_set[0],
            val_labels=eval_set[1],
            **self.datamodule_params,
        )

        # Store datamodule for later use in predict
        self.datamodule = dm

        # Set classifier-specific attributes
        if isinstance(self, ClassifierMixin):
            if is_partial_fit and classes is not None:
                self.classes_ = np.array(classes)
            elif not hasattr(self, "classes_"):
                self.classes_ = sorted(np.unique(y) if not isinstance(y, pd.Series) else y.unique())

            num_classes = len(self.classes_)
        else:
            num_classes = 1

        # Initialize model if not partial_fit or model doesn't exist
        if not is_partial_fit:
            self.network = generate_mlp(num_features=X.shape[1], num_classes=num_classes, **self.network_params)  # init here to allow feature selectors

            if num_classes > 1:
                early_stopping_metric_name = "ICE"
                metric_fcn = compute_probabilistic_multiclass_error
                metrics = [MetricSpec(name=early_stopping_metric_name, fcn=metric_fcn, requires_probs=True, requires_argmax=False)]
            else:
                early_stopping_metric_name = "MSE"
                metric_fcn = root_mean_squared_error
                metrics = [MetricSpec(name=early_stopping_metric_name, fcn=metric_fcn, requires_probs=False, requires_argmax=False)]

            checkpointing = BestEpochModelCheckpoint(
                monitor="val_" + early_stopping_metric_name,
                dirpath=self.trainer_params["default_root_dir"],
                filename="model-{" + "val_" + early_stopping_metric_name + ":.4f}",
                enable_version_counter=True,
                save_last=False,
                save_top_k=1,
                mode="min",
            )
            # metric_computing_callback = AggregatingValidationCallback(metric_name=early_stopping_metric_name, metric_fcn=metric_fcn)

            # tb_logger = TensorBoardLogger(save_dir=args.experiment_path, log_graph=True)  # save_dir="s3://my_bucket/logs/"
            progress_bar = TQDMProgressBar(refresh_rate=50)  # leave=True

            callbacks = [
                checkpointing,
                # metric_computing_callback,
                # NetworkGraphLoggingCallback(),
                LearningRateMonitor(logging_interval="epoch"),
                progress_bar,
                # PeriodicLearningRateFinder(period=10),
            ]
            if self.use_swa:
                callbacks.append(
                    (**self.swa_params))

            if eval_set is not None and (eval_set[0] is not None):

                early_stopping_rounds = self.early_stopping_rounds
                logger.info(f"Using early_stopping_rounds={early_stopping_rounds:_}")

                early_stopping = EarlyStoppingCallback(
                    monitor="val_" + early_stopping_metric_name, min_delta=0.001, patience=early_stopping_rounds, mode="min", verbose=True
                )  # stopping_threshold: Stops training immediately once the monitored quantity reaches this threshold.
                callbacks.append(early_stopping)

            trainer = L.Trainer(
                **self.trainer_params,
                # ----------------------------------------------------------------------------------------------------------------------
                # Callbacks:
                # ----------------------------------------------------------------------------------------------------------------------
                callbacks=callbacks,
                # DeviceStatsMonitor(),
                # ModelPruning("l1_unstructured", amount=0.5)
            )

            with trainer.init_module():
                self.model = self.model_class(network=self.network, metrics=metrics, **self.model_params)

                features_dtype = self.datamodule_params.get("features_dtype", torch.float32)
                data_slice = X.iloc[0:2, :].values if isinstance(X, pd.DataFrame) else X[0:2, :]

                try:
                    self.model.example_input_array = to_tensor_any(data_slice, dtype=features_dtype, safe=True)
                except Exception as e:
                    raise RuntimeError(f"Failed to prepare example_input_array: {e}")

            # Store trainer for later use in predict
            self.trainer = trainer

        # Tune parameters if requested and not already tuned (for partial_fit)
        if self.tune_params and not (is_partial_fit and hasattr(self, "_tuned")):

            tuner = Tuner(trainer)

            if self.tune_batch_size:
                tuner.scale_batch_size(model=self.model, datamodule=dm, mode="binsearch", init_val=self.datamodule_params.get("batch_size", 32))

            lr_finder = tuner.lr_find(self.model, datamodule=dm, num_training=300)
            new_lr = lr_finder.suggestion()
            logger.info(f"Using suggested LR={new_lr}")
            self.model.hparams.learning_rate = new_lr

            if is_partial_fit:
                self._tuned = True  # Mark as tuned for subsequent partial_fit calls

        # Train the model
        trainer.fit(model=self.model, datamodule=dm)

        # NOTE: best_epoch is stored in BestEpochModelCheckpoint callback and copied here
        # for convenience. This duplication is intentional to provide easy access via the
        # estimator without needing to search through callbacks. The callback remains the
        # source of truth during training, while this copy is for post-training access.
        for callback in trainer.callbacks:
            if isinstance(callback, BestEpochModelCheckpoint):
                self.best_epoch = callback.best_epoch
                logger.info(f"Best epoch recorded: {self.best_epoch}")
                break

        return self

    def fit(self, X, y, **fit_params):
        """Fit the model to the data."""
        eval_set = fit_params.get("eval_set", (None, None))
        return self._fit_common(X, y, eval_set=eval_set, is_partial_fit=False, fit_params=fit_params)

    def partial_fit(self, X, y, classes: Optional[np.ndarray] = None, **fit_params):
        """Incremental training for online learning."""
        eval_set = fit_params.get("eval_set", (None, None))
        return self._fit_common(X, y, eval_set=eval_set, is_partial_fit=True, classes=classes, fit_params=fit_params)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Returns a dictionary of all parameters for scikit-learn compatibility."""
        params = {
            "model_class": self.model_class,
            "model_params": deepcopy(self.model_params) if deep else self.model_params,
            # "network": self.network,
            "datamodule_class": self.datamodule_class,
            "datamodule_params": deepcopy(self.datamodule_params) if deep else self.datamodule_params,
            "tune_params": self.tune_params,
            "tune_batch_size": self.tune_batch_size,
            "float32_matmul_precision": self.float32_matmul_precision,
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
        if not hasattr(self, "datamodule") or self.datamodule is None:
            # Create a minimal datamodule for prediction if not available
            logger.warning("No datamodule found from training. Creating temporary datamodule for prediction.")
            self.datamodule = self.datamodule_class(
                train_features=X[:1],  # Dummy data
                train_labels=np.zeros(1),
                val_features=None,
                val_labels=None,
                **self.datamodule_params,
            )

        # Determine batch size for prediction (allow override for larger batches)
        pred_batch_size = batch_size if batch_size is not None else self.datamodule_params.get("batch_size", 64)
        logger.info(f"Using batch_size={pred_batch_size} for prediction")

        # Setup prediction dataset
        self.datamodule.setup_predict(X, batch_size=pred_batch_size)

        # Create prediction trainer with appropriate settings
        if not hasattr(self, "trainer") or self.trainer is None:
            logger.warning("No trainer found from training. Creating temporary trainer for prediction.")
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
                datamodule=self.datamodule,
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

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

        # For regression, return raw predictions
        if isinstance(self, RegressorMixin):
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
# Data
# ----------------------------------------------------------------------------------------------------------------------------


class TorchDataset(Dataset):
    def __init__(
        self,
        features,
        labels=None,  # Make optional
        features_dtype: torch.dtype = torch.float32,
        labels_dtype: torch.dtype = torch.float32,
        device: Optional[str] = None,
        batch_size: int = 0,
    ):
        self.features_dtype = features_dtype
        self.labels_dtype = labels_dtype
        self.device = device
        self.batch_size = batch_size

        # Store features as-is, unless GPU preloading requested
        if device == "cuda":
            self.features = to_tensor_any(features, features_dtype, device)
        else:
            self.features = features

        # Handle labels (optional for prediction)
        if labels is not None:
            if isinstance(labels, (pd.DataFrame, pd.Series)):
                labels = labels.to_numpy()
            elif isinstance(labels, pl.DataFrame):
                labels = labels.to_numpy()
            elif not isinstance(labels, (np.ndarray, torch.Tensor)):
                labels = np.asarray(labels)

            labels = np.asarray(labels).reshape(-1)
            self.labels = torch.tensor(labels, dtype=labels_dtype, device=device)
            dataset_length = len(self.labels)
        else:
            self.labels = None
            # Determine length from features
            if torch.is_tensor(self.features):
                dataset_length = len(self.features)
            elif isinstance(self.features, (np.ndarray, pd.DataFrame)):
                dataset_length = len(self.features)
            elif isinstance(self.features, pl.DataFrame):
                dataset_length = self.features.height
            else:
                raise TypeError(f"Cannot determine length from features type: {type(self.features)}")

        # Determine # of batches if in batch mode
        if batch_size > 0:
            self.num_batches = int(np.ceil(dataset_length / batch_size))

        self.dataset_length = dataset_length

    def __len__(self):
        return self.num_batches if self.batch_size > 0 else self.dataset_length

    def _extract(self, data, indices):
        """Extract and convert subset to tensor."""
        if isinstance(data, torch.Tensor):
            subset = data[indices]
        elif isinstance(data, np.ndarray):
            subset = data[indices]
        elif isinstance(data, pd.DataFrame):
            subset = data.iloc[indices, :].to_numpy()
        elif isinstance(data, pl.DataFrame):
            subset = data[indices].to_torch()
        else:
            raise TypeError(f"Unsupported data type for extraction: {type(data)}")

        if isinstance(subset, np.ndarray):
            subset = torch.from_numpy(subset)

        return subset.to(dtype=self.features_dtype, device=self.device)

    def __getitem__(self, idx):
        if self.batch_size > 0:
            # batched mode
            start = idx * self.batch_size
            end = min((idx + 1) * self.batch_size, self.dataset_length)
            indices = slice(start, end)
        else:
            # sample mode
            indices = idx

        x = self._extract(self.features, indices)

        # Return only features if no labels
        if self.labels is None:
            return x

        y = self.labels[indices]

        # Squeeze single-sample dimension only in sample mode
        if self.batch_size == 0 and x.ndim == 2 and x.shape[0] == 1:
            x = x.squeeze(0)

        return x, y


class TorchDataModule(LightningDataModule):
    """
    Improved Lightning DataModule with support for train/val/test/predict dataloaders.

    Features:
    - Supports reading from file for multi-GPU workloads
    - Supports placing entire dataset on GPU
    - Handles all stages: fit, test, predict
    - Reduces code duplication
    - Flexible batch size and dataloader configuration

    Args:
        train_features: Training features (DataFrame, array, or file path)
        train_labels: Training labels (DataFrame, array, or file path)
        val_features: Validation features (DataFrame, array, or file path)
        val_labels: Validation labels (DataFrame, array, or file path)
        test_features: Optional test features (DataFrame, array, or file path)
        test_labels: Optional test labels (DataFrame, array, or file path)
        read_fcn: Optional function to read data from file paths
        data_placement_device: Device to place data on ('cuda', 'cuda:0', etc.)
        features_dtype: Dtype for features (default: torch.float32)
        labels_dtype: Dtype for labels (default: torch.int64)
        dataloader_params: Dict of DataLoader parameters (batch_size, num_workers, etc.)
    """

    def __init__(
        self,
        train_features: Union[pd.DataFrame, np.ndarray, str],
        train_labels: Union[pd.DataFrame, np.ndarray, str],
        val_features: Union[pd.DataFrame, np.ndarray, str],
        val_labels: Union[pd.DataFrame, np.ndarray, str],
        test_features: Optional[Union[pd.DataFrame, np.ndarray, str]] = None,
        test_labels: Optional[Union[pd.DataFrame, np.ndarray, str]] = None,
        read_fcn: Optional[Callable] = None,
        data_placement_device: Optional[str] = None,
        features_dtype: torch.dtype = torch.float32,
        labels_dtype: torch.dtype = torch.int64,
        dataloader_params: Optional[dict] = None,
    ):
        """
        Initialize DataModule. Data pre-loading here allows automatic sharing
        between spawned processes via shared memory when using 'ddp_spawn'.
        """
        super().__init__()

        # Validate data_placement_device
        if data_placement_device is not None:
            if not data_placement_device.startswith("cuda"):
                raise ValueError(f"data_placement_device must be None or 'cuda'/'cuda:X', got: {data_placement_device}")

        # Store all parameters
        self.train_features = train_features
        self.train_labels = train_labels
        self.val_features = val_features
        self.val_labels = val_labels
        self.test_features = test_features
        self.test_labels = test_labels
        self.read_fcn = read_fcn
        self.data_placement_device = data_placement_device
        self.features_dtype = features_dtype
        self.labels_dtype = labels_dtype
        self.dataloader_params = dataloader_params or {}

        # For dynamic prediction data
        self.predict_features = None

        # Extract batch_size for easy access
        self.batch_size = self.dataloader_params.get("batch_size", 64)

    def prepare_data(self):
        """
        Called once on main process for data download/preprocessing.
        Do not assign state here (self.x = y) as it won't be available to other processes.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Called on every process/GPU. Load and setup datasets here.

        Args:
            stage: Current stage ('fit', 'test', 'predict', or None for all)
        """
        if stage in ("fit", None):
            self._load_data_from_files(["train_features", "train_labels", "val_features", "val_labels"])
            self._convert_features_dtype(["train_features", "val_features"])

        if stage in ("test", None) and self.test_features is not None:
            self._load_data_from_files(["test_features", "test_labels"])
            self._convert_features_dtype(["test_features"])

        if stage == "predict" and self.predict_features is not None:
            self._load_data_from_files(["predict_features"])
            self._convert_features_dtype(["predict_features"])

    def _load_data_from_files(self, var_names: list):
        """
        Load data from file paths if read_fcn is provided.

        Args:
            var_names: List of attribute names to check and load
        """
        if not self.read_fcn:
            return

        for var_name in var_names:
            var_content = getattr(self, var_name, None)
            if var_content is not None and isinstance(var_content, str):
                setattr(self, var_name, self.read_fcn(var_content))

    def _convert_features_dtype(self, feature_names: list):
        """
        Convert features to float32 for compatibility.

        Args:
            feature_names: List of feature attribute names to convert
        """
        for feature_name in feature_names:
            features = getattr(self, feature_name, None)
            if features is None:
                continue

            # Convert pandas/numpy to float32 if possible
            if hasattr(features, "astype"):
                try:
                    setattr(self, feature_name, features.astype("float32"))
                except (ValueError, TypeError):
                    # Not convertible, keep original dtype
                    pass

    def teardown(self, stage: Optional[str] = None):
        """Clean up resources (temp files, etc.)."""
        pass

    def on_gpu(self) -> bool:
        """Check if current model runs on GPU."""
        try:
            return type(self.trainer.accelerator).__name__ == "CUDAAccelerator"
        except (AttributeError, Exception):
            return False

    def _get_device(self) -> Optional[str]:
        """
        Determine device for data placement.

        Returns:
            Device string ('cuda', 'cuda:0', etc.) or None
        """
        if self.data_placement_device and self.on_gpu():
            return self.data_placement_device
        return None

    def _create_dataloader(
        self,
        features,
        labels=None,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        """
        Create a DataLoader with consistent configuration.

        Args:
            features: Feature data
            labels: Label data (optional, None for prediction)
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch

        Returns:
            Configured DataLoader

        Note:
            TorchDataset handles batching internally when batch_size > 0,
            so DataLoader is created with batch_size=None to iterate over
            pre-batched data. This enables vectorized extraction and GPU preloading.
        """
        device = self._get_device()
        on_gpu = self.on_gpu()

        # Prepare dataloader params (extract batch_size for TorchDataset)
        dl_params = self.dataloader_params.copy()
        batch_size = dl_params.pop("batch_size", self.batch_size)

        # Override shuffle and drop_last
        dl_params["shuffle"] = shuffle
        dl_params["drop_last"] = drop_last
        dl_params["pin_memory"] = on_gpu

        # IMPORTANT: Set batch_size=None since TorchDataset handles batching internally
        # when its own batch_size parameter > 0
        dl_params["batch_size"] = None

        # Create dataset with internal batching
        dataset = TorchDataset(
            features=features,
            labels=labels,
            features_dtype=self.features_dtype,
            labels_dtype=self.labels_dtype if labels is not None else None,
            batch_size=batch_size,  # TorchDataset will handle batching
            device=device,
        )

        return DataLoader(dataset, **dl_params)

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        return self._create_dataloader(
            features=self.train_features,
            labels=self.train_labels,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        return self._create_dataloader(
            features=self.val_features,
            labels=self.val_labels,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        if self.test_features is None:
            raise RuntimeError("test_features not provided during initialization")

        return self._create_dataloader(
            features=self.test_features,
            labels=self.test_labels,
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        """
        Return prediction DataLoader.

        Call setup_predict() first to set prediction data.
        """
        if self.predict_features is None:
            raise RuntimeError("predict_features not set. Call setup_predict(X) before using predict_dataloader()")

        return self._create_dataloader(
            features=self.predict_features,
            labels=None,  # No labels for prediction
            shuffle=False,
            drop_last=False,
        )

    def setup_predict(
        self,
        X: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        batch_size: Optional[int] = None,
    ):
        """
        Setup prediction data dynamically for sklearn-style predict(X) API.

        Args:
            X: Features to predict on
            batch_size: Optional batch size override

        Example:
            >>> datamodule.setup_predict(X, batch_size=2048)
            >>> predictions = trainer.predict(model, datamodule=datamodule)
        """
        self.predict_features = X

        # Update batch size if provided
        if batch_size is not None:
            self.batch_size = batch_size

        # Trigger setup for predict stage
        self.setup(stage="predict")

    def has_test_data(self) -> bool:
        """Check if test data is available."""
        return self.test_features is not None

    def get_feature_dim(self) -> int:
        """Get the feature dimension from training data."""
        features = self.train_features

        if isinstance(features, (pd.DataFrame, np.ndarray)):
            return features.shape[1]
        elif torch.is_tensor(features):
            return features.shape[1]
        else:
            raise TypeError(f"Cannot determine feature dimension from type: {type(features)}")

    def get_num_classes(self) -> Optional[int]:
        """
        Get number of classes from training labels (for classification tasks).

        Returns:
            Number of unique classes, or None if not applicable
        """
        labels = self.train_labels

        if labels is None:
            return None

        try:
            if isinstance(labels, pd.DataFrame):
                return len(labels.iloc[:, 0].unique())
            elif isinstance(labels, np.ndarray):
                return len(np.unique(labels))
            elif torch.is_tensor(labels):
                return len(torch.unique(labels))
        except Exception:
            return None

        return None


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

        # Determine comparison operator
        if mode == "min":
            self.monitor_op = lambda a, b: a < b
            self.best_score = float("inf")
        elif mode == "max":
            self.monitor_op = lambda a, b: a > b
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


# ----------------------------------------------------------------------------------------------------------------------------
# Network structure
# ----------------------------------------------------------------------------------------------------------------------------


def get_valid_num_groups(num_channels, preferred_num_groups):
    for g in range(preferred_num_groups, 0, -1):
        if num_channels % g == 0:
            return g
    return 1  # Fallback to 1 (LayerNorm-like) if no divisor found


def generate_mlp(
    num_features: int,
    num_classes: int,
    nlayers: int = 1,
    first_layer_num_neurons: int = None,
    min_layer_neurons: int = 1,
    neurons_by_layer_arch: MLPNeuronsByLayerArchitecture = MLPNeuronsByLayerArchitecture.Constant,
    consec_layers_neurons_ratio: float = 1.1,
    activation_function: Callable = torch.nn.ReLU,
    weights_init_fcn: Callable = None,
    dropout_prob: float = 0.15,
    inputs_dropout_prob: float = 0.002,
    use_layernorm: bool = True,
    use_batchnorm: bool = True,
    groupnorm_num_groups: int = 0,
    norm_kwargs: dict = None,
    verbose: int = 0,
):
    """Generates multilayer perceptron with specific architecture.
    If first_layer_num_neurons is not specified, uses num_features.
    Suitable in NAS and HPT/HPO procedures for generating ANN candidates.

    Args:
        verbose (int): If 1, logs the network architecture (e.g., 10-2-5-1).
    """

    # Auto inits

    if norm_kwargs is None:
        norm_kwargs = dict(eps=0.00001, momentum=0.1)

    if not first_layer_num_neurons:
        first_layer_num_neurons = num_features
    if num_classes:
        if min_layer_neurons < num_classes:
            min_layer_neurons = num_classes

    # Sanity checks
    assert dropout_prob >= 0.0
    assert consec_layers_neurons_ratio >= 1.0
    assert (num_classes >= 0 and isinstance(num_classes, int)) or num_classes is None
    assert nlayers >= 1 and isinstance(nlayers, int)
    assert min_layer_neurons >= 1 and isinstance(min_layer_neurons, int)
    assert first_layer_num_neurons >= min_layer_neurons and isinstance(first_layer_num_neurons, int)

    layers = []
    layer_sizes = [num_features]  # Track layer sizes for verbose logging
    if inputs_dropout_prob:
        layers.append(nn.Dropout(inputs_dropout_prob))
    if use_layernorm:
        layers.append(nn.LayerNorm(num_features, **norm_kwargs))

    if groupnorm_num_groups:
        num_groups_for_input = get_valid_num_groups(num_features, groupnorm_num_groups)
        if num_groups_for_input > 1:
            layers.append(nn.GroupNorm(num_groups=num_groups_for_input, num_channels=num_features, **norm_kwargs))  # Reuse kwargs for eps, etc.

    prev_layer_neurons = num_features
    cur_layer_neurons = first_layer_num_neurons
    cur_layer_virt_neurons = first_layer_num_neurons
    for layer in range(nlayers):

        # ----------------------------------------------------------------------------------------------------------------------------
        # Compute # of neurons in current layer
        # ----------------------------------------------------------------------------------------------------------------------------

        if layer > 0:
            if neurons_by_layer_arch == MLPNeuronsByLayerArchitecture.Declining:
                cur_layer_virt_neurons = prev_layer_virt_neurons / consec_layers_neurons_ratio
            elif neurons_by_layer_arch == MLPNeuronsByLayerArchitecture.Expanding:
                cur_layer_virt_neurons = prev_layer_virt_neurons * consec_layers_neurons_ratio
            elif neurons_by_layer_arch == MLPNeuronsByLayerArchitecture.ExpandingThenDeclining:
                if layer <= nlayers // 2:
                    cur_layer_virt_neurons = prev_layer_virt_neurons * consec_layers_neurons_ratio
                else:
                    cur_layer_virt_neurons = prev_layer_virt_neurons / consec_layers_neurons_ratio
            elif neurons_by_layer_arch == MLPNeuronsByLayerArchitecture.Autoencoder:
                if layer <= nlayers // 2:
                    cur_layer_virt_neurons = prev_layer_virt_neurons / consec_layers_neurons_ratio
                else:
                    cur_layer_virt_neurons = prev_layer_virt_neurons * consec_layers_neurons_ratio

            cur_layer_neurons = int(cur_layer_virt_neurons)
        if cur_layer_neurons < min_layer_neurons:
            if layer > 0 and not (neurons_by_layer_arch == MLPNeuronsByLayerArchitecture.Autoencoder):
                break
            else:  # no prev layers exist, need to create at least one
                cur_layer_neurons = min_layer_neurons

        # ----------------------------------------------------------------------------------------------------------------------------
        # Add linear layer with that many neurons
        # ----------------------------------------------------------------------------------------------------------------------------

        layers.append(nn.Linear(prev_layer_neurons, cur_layer_neurons))
        layer_sizes.append(cur_layer_neurons)

        # ----------------------------------------------------------------------------------------------------------------------------
        # Add optional bells & whistles - batchnorm, activation, dropout
        # ----------------------------------------------------------------------------------------------------------------------------

        if use_batchnorm:
            layers.append(nn.BatchNorm1d(cur_layer_neurons, **norm_kwargs))
        if activation_function:
            layers.append(activation_function)
        if dropout_prob:
            layers.append(nn.Dropout(dropout_prob))

        prev_layer_neurons = cur_layer_neurons
        prev_layer_virt_neurons = cur_layer_virt_neurons

    # ----------------------------------------------------------------------------------------------------------------------------
    # For classification, just add final linear layer with num_classes neurons, to get logits
    # ----------------------------------------------------------------------------------------------------------------------------

    if num_classes:
        layers.append(nn.Linear(prev_layer_neurons, num_classes))
        layer_sizes.append(num_classes)

    model = nn.Sequential(*layers)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Log network architecture if verbose is enabled
    # ----------------------------------------------------------------------------------------------------------------------------

    if verbose == 1:
        architecture = "-".join(str(size) for size in layer_sizes)
        logger.info(f"Network architecture: {architecture}")

    # ----------------------------------------------------------------------------------------------------------------------------
    # Init weights explicitly if weights_init_fcn is set
    # ----------------------------------------------------------------------------------------------------------------------------

    # Weights initialization
    if weights_init_fcn:

        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.BatchNorm1d)):
                if isinstance(weights_init_fcn, partial):
                    func_to_check = weights_init_fcn.func
                else:
                    func_to_check = weights_init_fcn

                # Initialize weights
                if hasattr(m, "weight") and m.weight is not None:
                    if func_to_check in (
                        torch.nn.init.xavier_normal_,
                        torch.nn.init.xavier_uniform_,
                        torch.nn.init.kaiming_normal_,
                        torch.nn.init.kaiming_uniform_,
                    ):
                        if m.weight.dim() >= 2:  # Only for Linear weights (2D)
                            weights_init_fcn(m.weight)
                        elif isinstance(m, nn.BatchNorm1d):  # BatchNorm weight (gamma, 1D)
                            torch.nn.init.normal_(m.weight, mean=1.0, std=0.02)  # Standard for BatchNorm
                    else:
                        weights_init_fcn(m.weight)

                # Initialize biases
                if hasattr(m, "bias") and m.bias is not None:
                    if func_to_check in (
                        torch.nn.init.xavier_normal_,
                        torch.nn.init.xavier_uniform_,
                        torch.nn.init.kaiming_normal_,
                        torch.nn.init.kaiming_uniform_,
                    ):
                        torch.nn.init.constant_(m.bias, 0.0)  # Standard for biases
                    else:
                        weights_init_fcn(m.bias)

        model.apply(init_weights)
        # Handle logging for partial functions
        init_name = weights_init_fcn.func.__name__ if isinstance(weights_init_fcn, partial) else weights_init_fcn.__name__
        logger.info(f"Applied {init_name} initialization to Linear weights; normal_/constant_ for BatchNorm weights/biases and Linear biases")

    model.example_input_array = torch.zeros(1, num_features)

    return model


class MLPTorchModel(L.LightningModule):
    def __init__(
        self,
        loss_fn: Callable,
        metrics: List[MetricSpec],
        network: torch.nn.Module,
        learning_rate: float = 1e-3,
        l1_alpha: float = 0.0,
        optimizer: Optional[type] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        lr_scheduler: Optional[type] = None,
        lr_scheduler_kwargs: Optional[Dict[str, Any]] = None,
        compile_network: Optional[str] = None,
        compute_trainset_metrics: bool = False,
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_monitor: Optional[str] = None,
        load_best_weights_on_train_end: bool = True,
    ):
        """
        PyTorch Lightning module for MLP training.

        Args:
            loss_fn: Loss function callable
            metrics: List of MetricSpec objects for evaluation
            network: The neural network module
            learning_rate: Learning rate for optimizer
            l1_alpha: L1 regularization coefficient (0.0 = no regularization)
            optimizer: Optimizer class (default: AdamW)
            optimizer_kwargs: Additional kwargs for optimizer
            lr_scheduler: Learning rate scheduler class
            lr_scheduler_kwargs: Additional kwargs for scheduler
            compile_network: torch.compile mode (e.g., 'max-autotune', 'reduce-overhead')
            compute_trainset_metrics: Whether to compute metrics on training set
            lr_scheduler_interval: 'epoch' or 'step'
            lr_scheduler_monitor: Metric to monitor for scheduler (e.g., 'val_loss')
            load_best_weights_on_train_end: Load best checkpoint weights after training
        """
        super().__init__()

        # Validation
        if network is None:
            raise ValueError("network must be provided")
        if loss_fn is None:
            raise ValueError("loss_fn must be provided")
        if metrics is None:
            metrics = []
        if lr_scheduler_interval not in ["epoch", "step"]:
            raise ValueError(f"lr_scheduler_interval must be 'epoch' or 'step', got {lr_scheduler_interval}")

        # Set defaults
        optimizer = optimizer or torch.optim.AdamW
        optimizer_kwargs = optimizer_kwargs or {}
        lr_scheduler_kwargs = lr_scheduler_kwargs or {}

        # Save hyperparameters (excluding non-serializable objects)
        self.save_hyperparameters(ignore=["loss_fn", "metrics", "network", "optimizer", "lr_scheduler"])

        # Store components
        self.network = network
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.best_epoch = None

        # Initialize lists to store outputs during epoch
        self.training_step_outputs = []
        self.validation_step_outputs = []

        # Apply torch.compile if requested
        self._apply_torch_compile()

        # Set example input for ONNX export if available
        if hasattr(network, "example_input_array"):
            self.example_input_array = network.example_input_array
        else:
            logger.debug("Network lacks 'example_input_array'; ONNX export may require manual input")

    def _apply_torch_compile(self) -> None:
        """Apply torch.compile to the network if enabled."""
        if not self.hparams.compile_network:
            return

        if torch.__version__ < "2.0":
            logger.warning("torch.compile requires PyTorch >= 2.0. Skipping compilation.")
            return

        try:
            self.network = torch.compile(self.network, mode=self.hparams.compile_network)
            logger.info(f"Applied torch.compile with mode='{self.hparams.compile_network}'")
        except Exception as e:
            logger.warning(f"Failed to apply torch.compile: {e}. Using uncompiled network.")

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(*args, **kwargs)

    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unpack batch into features and labels."""
        if isinstance(batch, dict):
            return batch["features"], batch["labels"]
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            return batch[0], batch[1]
        else:
            raise ValueError(f"Unexpected batch format: {type(batch)}")

    def training_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Training step."""
        features, labels = self._unpack_batch(batch)
        raw_predictions = self(features)

        # Compute loss
        loss = self.loss_fn(raw_predictions, labels)

        # Add L1 regularization if enabled
        if self.hparams.l1_alpha > 0:
            l1_norm = sum(p.abs().sum() for p in self.network.parameters())
            loss = loss + self.hparams.l1_alpha * l1_norm
            self.log("train_l1_norm", l1_norm, on_step=False, on_epoch=True)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Store predictions for metric computation if needed
        result = {"loss": loss}
        if self.hparams.compute_trainset_metrics:
            # MEMORY OPTIMIZATION: Store outputs on CPU immediately to free GPU memory
            # This prevents GPU memory accumulation throughout the epoch
            output = {"raw_predictions": raw_predictions.detach().cpu(), "labels": labels.detach().cpu()}
            self.training_step_outputs.append(output)

        return result

    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        logger.info(f"Epoch {self.current_epoch}, Step {self.global_step}: LR = {current_lr:.2e}")

        if not self.hparams.compute_trainset_metrics:
            return

        if not self.training_step_outputs:
            logger.warning("No training outputs collected for metric computation")
            return

        # Extract predictions and labels from collected outputs (already on CPU)
        preds_and_labels = [(out["raw_predictions"], out["labels"]) for out in self.training_step_outputs]

        # Compute metrics
        self.compute_metrics(preds_and_labels, prefix="train")

        # Clear outputs to free memory
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        features, labels = self._unpack_batch(batch)

        # No gradient computation needed
        raw_predictions = self(features)

        # Compute loss without L1 regularization for fair comparison
        loss = self.loss_fn(raw_predictions, labels)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # MEMORY OPTIMIZATION: Store outputs on CPU immediately to free GPU memory
        output = {"raw_predictions": raw_predictions.detach().cpu(), "labels": labels.detach().cpu()}
        self.validation_step_outputs.append(output)

        return output

    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        if not self.validation_step_outputs:
            logger.warning("No validation outputs collected for metric computation")
            return

        # Extract predictions and labels from collected outputs (already on CPU)
        preds_and_labels = [(out["raw_predictions"], out["labels"]) for out in self.validation_step_outputs]

        # Compute metrics
        self.compute_metrics(preds_and_labels, prefix="val")

        # Clear outputs to free memory
        self.validation_step_outputs.clear()

    def compute_metrics(self, predictions_and_labels: List[Tuple[torch.Tensor, torch.Tensor]], prefix: str = "val") -> None:
        """
        Compute and log all metrics given raw predictions and labels.
        Optimized: compute argmax, softmax, CPU/numpy only if needed, once each.

        Args:
            predictions_and_labels: List of (raw_predictions, labels) tuples from each batch (on CPU)
            prefix: Logging prefix ('train' or 'val')
        """
        # Concatenate all predictions and labels
        raw_predictions, labels = zip(*predictions_and_labels)
        raw_predictions = torch.cat(raw_predictions)
        labels = torch.cat(labels)

        # Determine which transformations are actually needed
        need_argmax = any(m.requires_argmax for m in self.metrics)
        need_softmax = any(m.requires_probs for m in self.metrics)
        need_cpu = any(m.requires_cpu for m in self.metrics)

        # Precompute transforms
        preds_dict = {}
        if need_argmax:
            preds_dict["argmax"] = raw_predictions.argmax(dim=1)
        if need_softmax:
            preds_dict["softmax"] = F.softmax(raw_predictions, dim=1)

        labels_cpu = None
        if need_cpu:
            # Already on CPU from storage optimization
            labels_cpu = to_numpy_safe(labels, cpu=False)

        # Compute metrics
        for metric in self.metrics:
            # Select correct prediction type
            if metric.requires_argmax:
                preds = preds_dict["argmax"]
            elif metric.requires_probs:
                preds = preds_dict["softmax"]
            else:
                preds = raw_predictions

            # Convert to CPU/numpy if needed
            if metric.requires_cpu:
                key = f"cpu_{id(preds)}"
                if key not in preds_dict:
                    # Already on CPU from storage optimization
                    preds_dict[key] = to_numpy_safe(preds, cpu=False)
                preds_np = preds_dict[key]
                labels_np = labels_cpu
            else:
                preds_np = preds
                labels_np = labels

            # Compute and log
            try:
                value = metric.fcn(y_true=labels_np, y_score=preds_np)
                self.log(
                    f"{prefix}_{metric.name}",
                    value,
                    prog_bar=True,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
            except Exception as e:
                logger.error(f"Failed to compute metric {prefix}_{metric.name}: {e}")

    def configure_optimizers(self):
        """Configure optimizer and optional learning rate scheduler."""
        # Create optimizer
        optimizer_kwargs = {"lr": self.hparams.learning_rate, **self.hparams.optimizer_kwargs}
        optimizer = self.optimizer(self.parameters(), **optimizer_kwargs)

        # Return optimizer only if no scheduler
        if self.lr_scheduler is None:
            return optimizer

        # Special handling for OneCycleLR
        if self.lr_scheduler.__name__ == "OneCycleLR":

            logger.info("Configuring OneCycleLR scheduler")

            # Calculate total steps
            steps_per_epoch = (
                len(self.trainer.datamodule.train_dataloader())
                if hasattr(self.trainer, "datamodule") and self.trainer.datamodule
                else len(self.trainer.train_dataloader)
            )

            total_steps = self.trainer.max_epochs * steps_per_epoch

            logger.info(f"OneCycleLR config:")
            logger.info(f"  - Steps per epoch: {steps_per_epoch}")
            logger.info(f"  - Max epochs: {self.trainer.max_epochs}")
            logger.info(f"  - Total steps: {total_steps}")
            logger.info(f"  - Interval: {self.hparams.lr_scheduler_interval}")

            # Update kwargs with calculated values
            scheduler_kwargs = {
                **self.hparams.lr_scheduler_kwargs,
                "total_steps": total_steps,
            }

            # Remove epochs/steps_per_epoch if present (use total_steps instead)
            scheduler_kwargs.pop("epochs", None)
            scheduler_kwargs.pop("steps_per_epoch", None)

            scheduler = self.lr_scheduler(optimizer, **scheduler_kwargs)
        else:
            # Normal scheduler creation
            scheduler = self.lr_scheduler(optimizer, **self.hparams.lr_scheduler_kwargs)

        # Configure scheduler settings
        scheduler_config = {
            "scheduler": scheduler,
            "interval": self.hparams.lr_scheduler_interval,
        }

        # Add monitor if specified (required for ReduceLROnPlateau)
        if self.hparams.lr_scheduler_monitor:
            scheduler_config["monitor"] = self.hparams.lr_scheduler_monitor

        logger.info(f"LR scheduler config: {scheduler_config}")
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    def on_train_end(self) -> None:
        """Load best model weights after training completes."""
        if not self.hparams.load_best_weights_on_train_end:
            return

        # Only rank 0 handles checkpoint loading in distributed settings
        if not self.trainer.is_global_zero:
            return

        # Find ModelCheckpoint callback
        checkpoint_callback = None
        for callback in self.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                checkpoint_callback = callback
                break

        if checkpoint_callback is None:
            logger.warning("No ModelCheckpoint callback found. Cannot load best weights.")
            return

        best_model_path = checkpoint_callback.best_model_path

        if not best_model_path or not os.path.exists(best_model_path):
            logger.warning(f"No valid checkpoint at {best_model_path}. Using current weights.")
            return

        # Log checkpoint info
        best_score = checkpoint_callback.best_model_score
        score_str = f"{best_score:.4f}" if best_score is not None else "N/A"
        logger.info(f"Loading best model from {best_model_path} (score: {score_str})")

        try:
            checkpoint = torch.load(best_model_path, map_location=self.device, weights_only=False)

            if "state_dict" not in checkpoint:
                logger.error("Checkpoint missing 'state_dict'. Cannot load weights.")
                return

            # Load state dict
            missing, unexpected = self.load_state_dict(checkpoint["state_dict"], strict=False)

            if missing:
                logger.warning(f"Missing keys in state_dict: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys in state_dict: {unexpected}")

            # Store best epoch info
            if "epoch" in checkpoint:
                self.best_epoch = checkpoint["epoch"]
                logger.info(f"Loaded weights from epoch {self.best_epoch}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)

    def predict_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Handle prediction for both (x, y) and x-only batches.

        Returns raw model output (logits for classification, values for regression).
        Softmax/argmax conversion is handled in the estimator's predict methods.
        """
        # Ensure model is in eval mode
        if self.training:
            logger.warning(f"Model was in training mode during predict_step at batch {batch_idx}. Switching to eval mode.")
            self.eval()

        # Handle both training/testing format (x, y) and prediction format (x only)
        if isinstance(batch, (tuple, list)):
            x = batch[0]
        else:
            x = batch

        # Forward pass - return raw logits/values
        with torch.no_grad():
            logits = self(x)

        # For classification, apply softmax to get probabilities
        # (MLPTorchModel doesn't have is_classification attribute, so check network output)
        if logits.dim() == 2 and logits.shape[1] > 1:
            # Multi-class classification - return probabilities
            return torch.softmax(logits, dim=1)
        else:
            # Regression or binary classification - return raw values
            return logits
