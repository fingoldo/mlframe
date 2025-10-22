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
from torch.utils.data import DataLoader, Dataset

import psutil
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning import LightningDataModule
from lightning.pytorch.tuner import Tuner

from lightning.pytorch.callbacks import Callback, LearningRateFinder
from lightning.pytorch.callbacks.early_stopping import EarlyStopping as EarlyStoppingCallback
from lightning.pytorch.callbacks import RichProgressBar, TQDMProgressBar, ModelPruning, LearningRateMonitor, LearningRateFinder
from lightning.pytorch.callbacks import ModelCheckpoint, DeviceStatsMonitor, StochasticWeightAveraging, GradientAccumulationScheduler

from enum import Enum, auto
from functools import partial
import pandas as pd, numpy as np

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


def custom_collate_fn(batch):
    # Return the batch as-is (mimicking lambda x: x)
    return batch


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
        tune_params: bool = False,
        tune_batch_size: bool = False,
        float32_matmul_precision: str = None,
    ):
        store_params_in_object(obj=self, params=get_parent_func_args())

    def _fit_common(self, X, y, eval_set: tuple = (None, None), is_partial_fit: bool = False, classes: Optional[np.ndarray] = None, fit_params: dict = None):
        """Common logic for fit and partial_fit."""

        if fit_params is None:
            fit_params = {}

        # Enable TF32 for float32 matrix multiplication if on GPU
        if self.float32_matmul_precision and torch.cuda.is_available():
            assert self.float32_matmul_precision in "highest high medium"
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

        # Set classifier-specific attributes
        if isinstance(self, ClassifierMixin):
            self.return_proba = True
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
            else:
                early_stopping_metric_name = "MSE"
                metric_fcn = root_mean_squared_error

            checkpointing = BestEpochModelCheckpoint(
                monitor="val_" + early_stopping_metric_name,
                dirpath=self.trainer_params["default_root_dir"],
                filename="model-{" + "val_" + early_stopping_metric_name + ":.4f}",
                enable_version_counter=True,
                save_last=False,
                save_top_k=1,
                mode="min",
            )
            metric_computing_callback = AggregatingValidationCallback(metric_name=early_stopping_metric_name, metric_fcn=metric_fcn)

            # tb_logger = TensorBoardLogger(save_dir=args.experiment_path, log_graph=True)  # save_dir="s3://my_bucket/logs/"
            progress_bar = TQDMProgressBar(refresh_rate=50)  # leave=True

            callbacks = [
                checkpointing,
                metric_computing_callback,
                NetworkGraphLoggingCallback(),
                LearningRateMonitor(logging_interval="epoch"),
                progress_bar,
                # PeriodicLearningRateFinder(period=10),
            ]
            if self.use_swa:
                callbacks.append(StochasticWeightAveraging(swa_epoch_start=5, swa_lrs=1e-3))

            if eval_set is not None and (eval_set[0] is not None):

                early_stopping_rounds = fit_params.get("early_stopping_rounds", 100)
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
                self.model = self.model_class(network=self.network, **self.model_params)
                
                features_dtype=self.datamodule_params.get("features_dtype", torch.float32)
                data_slice=X.iloc[0:2, :].values if isinstance(X, pd.DataFrame) else X[0:2, :]
                
                try_dtype=None
                if features_dtype==torch.float32:
                    try_dtype=np.float32
                elif features_dtype==torch.float64:
                    try_dtype=np.float64
                
                if try_dtype:
                    try:
                        data_slice=data_slice.astype(try_dtype)
                    except Exception as e:
                        pass
                        # if any categoricals, network must know itself how to handle them

                self.model.example_input_array = torch.tensor(
                    data_slice, dtype=features_dtype
                )

        # Tune parameters if requested and not already tuned (for partial_fit)
        if self.tune_params and not (is_partial_fit and hasattr(self, "_tuned")):
            tuner = Tuner(trainer)
            if self.tune_batch_size:
                tuner.scale_batch_size(model=self.model, datamodule=dm, mode="binsearch", init_val=self.datamodule_params.get("batch_size", 32))
            lr_finder = tuner.lr_find(self.model, datamodule=dm)
            new_lr = lr_finder.suggestion()
            logger.info(f"Using suggested LR={new_lr}")
            self.model.hparams.learning_rate = new_lr
            if is_partial_fit:
                self._tuned = True  # Mark as tuned for subsequent partial_fit calls

        # Train the model
        trainer.fit(model=self.model, datamodule=dm)

        # Store best epoch from checkpoint callback
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

    def predict(self, X, device: Optional[str] = None):
        """Predict using the model, handling device mismatches gracefully.

        Args:
            X: Input data (numpy array, pandas DataFrame, or Series).
            device: Optional device to place the model and input tensor (e.g., 'cpu', 'cuda'). If None, tries self.model.device, falls back to 'cpu' on failure.

        Returns:
            numpy.ndarray: Model predictions.
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()

        # Determine target device
        target_device = device if device is not None else getattr(self.model, "device", "cpu")

        try:
            # Try original behavior: place tensor on model's device
            X_tensor = torch.tensor(X, dtype=self.datamodule_params.get("features_dtype", torch.float32), device=target_device)
        except RuntimeError as e:
            # Handle device mismatch (e.g., model on GPU but no GPU available)
            logger.warning(f"Failed to place tensor on {target_device}: {e}. Falling back to CPU.")
            self.model.to("cpu")  # Move model to CPU
            X_tensor = torch.tensor(X, dtype=self.datamodule_params.get("features_dtype", torch.float32), device="cpu")

        self.model.eval()
        with torch.no_grad():
            res = self.model(X_tensor)

        res = res.detach().cpu()
        if res.dtype == torch.bfloat16:
            res = res.to(torch.float32)

        return res.numpy()

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Returns a dictionary of all parameters for scikit-learn compatibility."""
        params = {
            "model_class": self.model_class,
            "model_params": deepcopy(self.model_params) if deep else self.model_params,
            "network": self.network,
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

    def score(self, X, y, sample_weight: Optional[np.ndarray] = None) -> float:
        """Returns the coefficient of determination R^2 for regression or accuracy for classification."""
        y_pred = self.predict(X)
        if isinstance(self, RegressorMixin):
            return r2_score(y, y_pred, sample_weight=sample_weight)
        elif isinstance(self, ClassifierMixin):
            y_pred = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        else:
            raise ValueError("Estimator must be a RegressorMixin or ClassifierMixin")


class PytorchLightningRegressor(RegressorMixin, PytorchLightningEstimator):
    _estimator_type = "regressor"
    pass


class PytorchLightningClassifier(ClassifierMixin, PytorchLightningEstimator):
    _estimator_type = "classifier"

    def predict(self, X, device: Optional[str] = None):
        """Predict class labels for samples in X."""
        proba = super(PytorchLightningClassifier, self).predict(X, device=device)  # Get probabilities from parent
        return np.argmax(proba, axis=1)  # Convert to class labels

    def predict_proba(self, X, device: Optional[str] = None):
        """Predict class probabilities for samples in X."""
        return super(PytorchLightningClassifier, self).predict(X, device=device)  # Relay to parent's predict


# ----------------------------------------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------------------------------------


class TorchDataset(Dataset):
    """Wrapper around pandas dataframe or numpy array.
    Supports placing entire dataset on GPU.
    """

    def __init__(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        labels: Union[pd.DataFrame, np.ndarray],
        features_dtype: object = torch.float32,
        labels_dtype: object = torch.float32,
        device: str = None,
    ):
        "Converts pandas/numpy data into tensors of specified type, if needed"

        if isinstance(features, (pd.DataFrame, pd.Series)):
            features = features.to_numpy()
        if isinstance(labels, (pd.DataFrame, pd.Series)):
            labels = labels.to_numpy()

        self.features = torch.tensor(features, dtype=features_dtype, device=device)
        labels = np.asarray(labels).reshape(-1)
        self.labels = torch.tensor(labels, dtype=labels_dtype, device=device)
        self.device = device

    def __len__(self):
        # Returns the total amount of samples in your Dataset
        return len(self.labels)

    def __getitem__(self, idx):
        # Returns, given an index, the i-th sample and label
        x, y = self.features[idx], self.labels[idx]
        if x.ndim == 1:  # Convert [3072] to [3072, 1] if needed
            x = x.unsqueeze(-1)
        return x, y

    def __getitems__(self, indices: List):
        if len(indices) == len(self.labels):
            x, y = self.features, self.labels
        else:
            x, y = self.features[indices], self.labels[indices]

        if x.ndim == 1:  # Convert [3072] to [3072, 1] if needed
            x = x.unsqueeze(-1)
        return x, y


class TorchDataModule(LightningDataModule):
    """Template of a reasonable Lightning datamodule.
    Supports reading from file for multi-gpu workloads.
    Supports placing entire dataset on GPU.
    """

    def __init__(
        self,
        train_features: Union[pd.DataFrame, np.ndarray, str],
        train_labels: Union[pd.DataFrame, np.ndarray, str],
        val_features: Union[pd.DataFrame, np.ndarray, str],
        val_labels: Union[pd.DataFrame, np.ndarray, str],
        read_fcn: object = None,
        data_placement_device: str = None,
        features_dtype: object = torch.float32,
        labels_dtype: object = torch.int64,
        dataloader_params: dict = {},
    ):
        # A simple way to prevent redundant dataset replicas is to rely on torch.multiprocessing to share the data automatically between spawned processes via shared memory.
        # For this, all data pre-loading should be done on the main process inside DataModule.__init__(). As a result, all tensor-data will get automatically shared when using
        # the 'ddp_spawn' strategy.

        assert data_placement_device in (None, "cuda")

        super().__init__()

        params = get_parent_func_args()
        store_params_in_object(obj=self, params=params)

        # If main data is passed via init parameters, it's serialized for every worker. For bigger datasets,
        # it will be faster to dump to files outside of the Datamodule (especially memory-mapped files), and then read
        # separately from file in each of the worker processes.

    def prepare_data(self):
        # code to be executed only once (e.g. downloading dataset from db or internet)
        # prepare_data is called from the main process. It is not recommended to assign state here (e.g. self.x = y) since it is called on a single process and if you assign
        # states here then they won’t be available for other processes.
        pass

    def setup(self, stage: str = None):
        # This can be executed multiple times (say, once per each GPU)
        # Setup is called from every process across all the nodes. Setting state here is recommended.

        if stage == "fit" or stage is None:
            # if dataset is provided as string, along with function to read actual data - do it.
            if self.read_fcn:
                for var in "train_features train_labels val_features val_labels".split():
                    var_content = getattr(self, var)
                    if isinstance(var_content, str):
                        setattr(self, var, self.read_fcn(var_content))

    def teardown(self, stage: str = None):
        # Place to remove temp files etc.
        pass

    def on_gpu(self) -> bool:
        "Checks if current DM's model runs on GPU"
        try:
            on_gpu = type(self.trainer.accelerator).__name__ == "CUDAAccelerator"
        except Exception as e:
            on_gpu = False

        return on_gpu

    def train_dataloader(self):

        on_gpu = self.on_gpu()
        device = self.data_placement_device if (self.data_placement_device and on_gpu) else None

        features=self.train_features
        try:
            features=features.astype('float32')
        except Exception as e:
            pass

        return DataLoader(
            TorchDataset(
                features=features, labels=self.train_labels, features_dtype=self.features_dtype, labels_dtype=self.labels_dtype, device=device
            ),
            pin_memory=on_gpu,
            **self.dataloader_params,
        )

    def val_dataloader(self):

        on_gpu = self.on_gpu()
        device = self.data_placement_device if (self.data_placement_device and on_gpu) else None

        dataloader_params = self.dataloader_params.copy()
        dataloader_params["shuffle"] = False

        features=self.val_features
        try:
            features=features.astype('float32')
        except Exception as e:
            pass

        return DataLoader(
            TorchDataset(features=features, labels=self.val_labels, features_dtype=self.features_dtype, labels_dtype=self.labels_dtype, device=device),
            pin_memory=on_gpu,
            **dataloader_params,
        )


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


# Custom ModelCheckpoint to track best epoch
class BestEpochModelCheckpoint(ModelCheckpoint):
    def __init__(self, monitor: str = "val_loss", mode: str = "min", **kwargs):
        super().__init__(monitor=monitor, mode=mode, **kwargs)
        self.best_epoch = None  # Track the epoch of the best model
        self.best = float("inf") if mode == "min" else float("-inf")  # Initialize best score
        self.monitor_op = torch.lt if mode == "min" else torch.gt
        logger.info(f"Initialized BestEpochModelCheckpoint with monitor={monitor}, mode={mode}")

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_validation_end(trainer, pl_module)
        # Update best_epoch when a new best model is saved
        if self.best_k_models and self.best_model_path:
            current_score = self.best_k_models[self.best_model_path]
            if self.monitor_op(current_score, self.best):
                self.best = current_score  # Update best score
                self.best_epoch = trainer.current_epoch
                logger.info(f"New best model at epoch {self.best_epoch} with {self.monitor}={current_score:.4f}")


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
    use_batchnorm: bool = False,
    batchnorm_kwargs=dict(eps=0.00001, momentum=0.1),
    verbose: int = 0,
):
    """Generates multilayer perceptron with specific architecture.
    If first_layer_num_neurons is not specified, uses num_features.
    Suitable in NAS and HPT/HPO procedures for generating ANN candidates.

    Args:
        verbose (int): If 1, logs the network architecture (e.g., 10-2-5-1).
    """

    # Auto inits
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
            layers.append(nn.BatchNorm1d(cur_layer_neurons, **batchnorm_kwargs))
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
        return_proba: bool,
        network: torch.nn.Module = None,
        learning_rate: float = 1e-3,
        l1_alpha: float = 0.0,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs: dict = {},
        lr_scheduler: object = None,
        lr_scheduler_kwargs: dict = {},
        compile_network: str = None,  # New flag to toggle torch.compile
    ):
        """compile_network='max-autotune-no-cudagraphs' at least works on rtx 5060."""
        super().__init__()
        self.save_hyperparameters()  # ignore=["network"]
        store_params_in_object(obj=self, params=get_parent_func_args())

        self.is_compiled = False  # Track if network is compiled

        # Apply torch.compile if enabled
        if compile_network and torch.__version__ >= "2.0":
            try:

                self.network = torch.compile(
                    self.network, mode=compile_network
                )  # Serializing a compiled model with pickle fails with Can't pickle local object 'convert_frame.<locals>._convert_frame' and cannot pickle 'ConfigModuleInstance' object when using dil
                self.is_compiled = True  # Mark as compiled
                logger.info("Applied torch.compile with for optimized forward/backward passes")
            except Exception as e:
                logger.warning(f"Failed to apply torch.compile: {e}. Falling back to uncompiled network.")
                if torch.cuda.is_available():
                    device = torch.device("cuda")
                    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
                    logger.info(f"GPU SM count: {sm_count}")
        elif compile_network:
            logger.warning("torch.compile requires PyTorch >= 2.0. Skipping compilation.")

        try:
            self.example_input_array = network.example_input_array  # specifying allows to skip example_input_array when doing ONNX export
        except Exception:
            pass

    def forward(self, x):
        logits = self.network(x)

        if not self.return_proba:
            return logits
        else:
            return torch.softmax(logits, dim=1)

    def compute_loss(self, batch):
        features, labels = batch
        logits = self.network(features)

        loss = self.loss_fn(logits, labels)

        if self.l1_alpha:  # l2 regularization is already implemented in optimizers via weight_decay parameter
            l1_norm = sum(p.abs().sum() for p in self.network.parameters())
            loss = loss + self.l1_alpha * l1_norm

        return loss

    def training_step(self, batch, batch_idx):

        features, labels = batch

        # skip empty batches
        if features.size(0) == 0:
            # return None tells Lightning to skip this batch
            return None

        loss = self.compute_loss(batch)

        # ensure loss is a scalar and requires grad
        if loss.requires_grad is False:
            raise RuntimeError("Loss does not require gradients!")

        self.log(
            "train_loss",
            loss,
            on_epoch=False,
            on_step=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        predictions = self.forward(features)
        # we can still compute val_loss if needed
        return predictions, labels

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        features, _ = batch
        with torch.inference_mode():
            logits = self.network(features)
            if self.return_proba:
                return torch.softmax(logits, dim=1)
            return logits

    def configure_optimizers(self):

        optimizer = self.optimizer(params=self.network.parameters(), **self.optimizer_kwargs)
        res = {"optimizer": optimizer}
        if self.lr_scheduler:
            res["lr_scheduler"] = self.lr_scheduler(optimizer=optimizer, **self.lr_scheduler_kwargs)

        return res

    def on_train_end(self):
        # Lightning ES callback do not auto-load best weights, so we locate ModelCheckpoint & do that ourselves.

        for callback in self.trainer.callbacks:
            if isinstance(callback, BestEpochModelCheckpoint):
                logger.info(f"Loading weights from {callback.best_model_path} (best epoch: {callback.best_epoch})")
                best_model = self.__class__.load_from_checkpoint(callback.best_model_path)
                self.load_state_dict(best_model.state_dict())
                self.best_iteration = callback.best_epoch  # Store for access
                break
