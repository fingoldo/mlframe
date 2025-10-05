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

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning import LightningDataModule
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import Callback, LearningRateFinder

from enum import Enum, auto
from functools import partial
import pandas as pd, numpy as np

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from pyutilz.pythonlib import store_params_in_object, get_parent_func_args

# ----------------------------------------------------------------------------------------------------------------------------
# ENUMS
# ----------------------------------------------------------------------------------------------------------------------------


class MLPNeuronsByLayerArchitecture(Enum):
    Constant = auto()
    Declining = auto()
    Expanding = auto()
    ExpandingThenDeclining = auto()
    Autoencoder = auto()


# ----------------------------------------------------------------------------------------------------------------------------
# Sklearn compatibility
# ----------------------------------------------------------------------------------------------------------------------------


class PytorchLightningEstimator(BaseEstimator):
    """Wrapper that allows Pytorch Lightning model, datamodule and trainer to participate in sklearn pipelines.
    Supports early stopping (via eval_set in fit_params).
    """

    def __init__(
        self,
        model: object,
        network: object,
        datamodule: object,
        trainer: object,
        args: object,
        loss_fn: Callable,
        features_dtype: object = torch.float32,
        labels_dtype: object = torch.int64,
        tune_params: bool = True,
    ):
        store_params_in_object(obj=self, params=get_parent_func_args())

    def fit(self, X, y, **fit_params):

        if isinstance(fit_params, dict):
            eval_set = fit_params.get("eval_set")
        else:
            eval_set = None, None

        dm = self.datamodule(
            train_features=X,
            train_labels=y,
            val_features=eval_set[0],
            val_labels=eval_set[1],
            features_dtype=self.features_dtype,
            labels_dtype=self.labels_dtype,
            batch_size=self.args.batch_size,
        )

        if isinstance(self, ClassifierMixin):
            return_proba = True
            if isinstance(y, pd.Series):
                self.classes_ = sorted(y.unique())
            else:
                self.classes_ = sorted(np.unique(y))
        else:
            return_proba = False

        # For faster initialization, you can create model parameters with desired dtype directly on the device:
        with self.trainer.init_module():
            self.model = self.model(model=self.network, loss_fn=self.loss_fn, learning_rate=self.args.lr, args=self.args, return_proba=return_proba)

            self.model.example_input_array = torch.tensor(X.iloc[0:2, :].values, dtype=torch.float32)

        if self.tune_params:
            tuner = Tuner(self.trainer)

            # Auto-scale batch size with binary search
            if False:
                tuner.scale_batch_size(
                    model=self.model, datamodule=dm, mode="binsearch", init_val=self.args.batch_size
                )  # dm has to have in __init__ & persist the batch_size parameter. scale_batch_size sets it.

            # Run learning rate finder
            lr_finder = tuner.lr_find(self.model, datamodule=dm)

            # Results can be found in lr_finder.results

            # Plot with
            fig = lr_finder.plot(suggest=True)
            fig.show()

            new_lr = lr_finder.suggestion()
            logger.info(f"Using suggested LR={new_lr}")
            self.model.hparams.learning_rate = new_lr
            self.model.model.learning_rate = new_lr

        self.trainer.fit(model=self.model, datamodule=dm)

        return self

    def predict(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()
        X = torch.tensor(X, dtype=self.features_dtype, device=self.model.device)
        self.model.eval()
        res = self.model(X)
        return res.detach().cpu().numpy()


class PytorchLightningRegressor(RegressorMixin, PytorchLightningEstimator):
    _estimator_type = "regressor"
    pass


class PytorchLightningClassifier(ClassifierMixin, PytorchLightningEstimator):
    _estimator_type = "classifier"

    def predict_proba(self, X):
        return self.predict(X)


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
        return self.features[idx], self.labels[idx]

    def __getitems__(self, indices: List):
        if len(indices) == len(self.labels):
            # print(indices[:10], indices[-10:])
            return self.features, self.labels
        else:
            return self.features[indices], self.labels[indices]


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
        batch_size: int = 32,
        params=dict(
            sampler=None,
            batch_sampler=None,
            num_workers=min(8, os.cpu_count()),
            collate_fn=lambda x: x,  # required for __getitems__ to work in TorchDataset down the road
            drop_last=False,
            timeout=0,
            worker_init_fn=None,
            prefetch_factor=None,
            persistent_workers=False,
        ),
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
        return DataLoader(
            TorchDataset(
                features=self.train_features, labels=self.train_labels, features_dtype=self.features_dtype, labels_dtype=self.labels_dtype, device=device
            ),
            shuffle=(self.batch_size < len(self.train_labels)),
            batch_size=self.batch_size,
            pin_memory=on_gpu,
            **self.params,
        )

    def val_dataloader(self):

        on_gpu = self.on_gpu()
        device = self.data_placement_device if (self.data_placement_device and on_gpu) else None
        return DataLoader(
            TorchDataset(features=self.val_features, labels=self.val_labels, features_dtype=self.features_dtype, labels_dtype=self.labels_dtype, device=device),
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=on_gpu,
            **self.params,
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
    dropout_prob: float = 0.0,
    inputs_dropout_prob: float = 0.0,
    use_batchnorm: bool = False,
    batchnorm_kwargs=dict(eps=0.00001, momentum=0.1),
):
    """Generates multilayer perceptron with specific architecture.
    If first_layer_num_neurons is not specified, uses num_features.
    Suitable in NAS and HPT/HPO procedures for generating ANN candidates.
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

    model = nn.Sequential(*layers)

    # ----------------------------------------------------------------------------------------------------------------------------
    # Init weights explicitly if weights_init_fcn is set
    # ----------------------------------------------------------------------------------------------------------------------------

    if weights_init_fcn:  # e.g., torch.nn.init.xavier_uniform

        if isinstance(weights_init_fcn, partial):
            func_to_check = weights_init_fcn.func
        else:
            func_to_check = weights_init_fcn

        def init_weights(m):
            for block_name in ("weight", "bias"):
                if hasattr(m, block_name):
                    block = getattr(m, block_name)
                    if func_to_check in (
                        torch.nn.init.xavier_normal_,
                        torch.nn.init.xavier_uniform_,
                        torch.nn.init.kaiming_normal_,
                        torch.nn.init.kaiming_uniform_,
                    ):
                        if block.dim() >= 2:
                            weights_init_fcn(block)
                    else:
                        weights_init_fcn(block)

        model.apply(init_weights)

    model.example_input_array = torch.zeros(1, num_features)

    return model


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


class MLPTorchModel(L.LightningModule):
    def __init__(
        self,
        args: object,
        loss_fn: Callable,
        return_proba: bool,
        model: torch.nn.Module = None,
        learning_rate: float = 0.1,
        l1_alpha: float = 0.0,
        optimizer=torch.optim.Adam,
        optimizer_kwargs: dict = {},
        lr_scheduler: object = None,
        lr_scheduler_kwargs: dict = {},
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        store_params_in_object(obj=self, params=get_parent_func_args())

        try:
            self.example_input_array = model.example_input_array  # specifying allows to skip example_input_array when doing ONNX export
        except Exception:
            pass

    def forward(self, x):
        logits = self.model(x)

        if not self.return_proba:
            return logits
        else:
            return torch.softmax(logits, dim=1)

    def compute_loss(self, batch):
        features, labels = batch
        logits = self.model(features)

        loss = self.loss_fn(logits, labels)

        if self.l1_alpha:  # l2 regularization is already implemented in optimizers via weight_decay parameter
            l1_norm = sum(p.abs().sum() for p in self.model.parameters())
            loss = loss + self.l1_alpha * l1_norm

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("train_loss", loss, on_epoch=False, on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        features, labels = batch
        predictions = self.forward(features)
        # we can still compute val_loss if needed
        return predictions, labels

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        features, _ = batch
        with torch.inference_mode():
            logits = self.model(features)
            if self.return_proba:
                return torch.softmax(logits, dim=1)
            return logits

    def configure_optimizers(self):

        optimizer = self.optimizer(params=self.model.parameters(), **self.optimizer_kwargs)
        res = {"optimizer": optimizer}
        if self.lr_scheduler:
            res["lr_scheduler"] = self.lr_scheduler(optimizer=optimizer, **self.lr_scheduler_kwargs)

        return res

    def on_train_end(self):
        # Lightning ES callback do not auto-load best weights, so we locate ModelCheckpoint & do that ourselves.
        for callback in self.trainer.callbacks:
            if isinstance(callback, ModelCheckpoint):
                logger.info(f"Loading weights from {callback.best_model_path}")
                best_model = self.__class__.load_from_checkpoint(callback.best_model_path)
                self.load_state_dict(best_model.state_dict())
                break


def get_predictions(model, dataloader):
    model.eval()
    probs = []
    with torch.inference_mode():
        for features, labels in dataloader:
            logits = model(features)
            probs.append(torch.softmax(logits))
    return torch.concat(probs)
