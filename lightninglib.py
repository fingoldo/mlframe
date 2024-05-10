"""Everything that makes working with Pytorch Lightning easier."""

# ----------------------------------------------------------------------------------------------------------------------------
# Normal Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import torch
import pandas as pd, numpy as np
from lightning import LightningDataModule
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import Callback
from torch.utils.data import DataLoader, Dataset

from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from pyutilz.pythonlib import store_params_in_object, get_parent_func_args

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

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
        datamodule: object,
        trainer: object,
        args: object,
        features_dtype: object = torch.float32,
        labels_dtype: object = torch.int64,
        tune_params: bool = True,
    ):
        params = get_parent_func_args()
        store_params_in_object(obj=self, params=params)

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
            if isinstance(y, pd.Series):
                self.classes_ = sorted(y.unique())
            else:
                self.classes_ = sorted(np.unique(y))

        # For faster initialization, you can create model parameters with the desired dtype directly on the device:
        with self.trainer.init_module():
            if isinstance(self, ClassifierMixin):
                self.model = self.model(num_features=X.shape[1], num_classes=len(self.classes_), learning_rate=self.args.lr, args=self.args)
            else:
                self.model = self.model(num_features=X.shape[1], learning_rate=self.args.lr, args=self.args)

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

            # Results can be found in
            print(lr_finder.results)

            # Plot with
            fig = lr_finder.plot(suggest=True)
            fig.show()

            new_lr = lr_finder.suggestion()
            logger.info(f"Using suggested LR={new_lr}")
            self.model.model.learning_rate = new_lr

        self.trainer.fit(model=self.model, datamodule=dm)

        return self

    def predict(self, X):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()
        X = torch.tensor(X, dtype=self.features_dtype, device=self.model.device)
        self.model = self.model.eval()
        res = self.model(X)
        return res.detach().cpu().numpy()


class PytorchLightningRegressor(PytorchLightningEstimator, RegressorMixin):
    pass


class PytorchLightningClassifier(PytorchLightningEstimator, ClassifierMixin):
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
            num_workers=0,
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
        self.batched_predictions = []
        self.batched_labels = []

    def on_validation_batch_end(self, trainer, pl_module, outputs):
        predictions, labels = outputs
        self.batched_predictions.append(predictions)
        self.batched_labels.append(labels)

    def on_validation_epoch_end(self, trainer, pl_module):
        predictions = torch.concat(self.batched_predictions)
        labels = torch.concat(self.batched_labels)

        metric_value = self.metric_fcn(predictions, labels)
        pl_module.log("val_" + self.metric_name, metric_value, on_epoch=self.on_epoch, on_step=self.on_step, prog_bar=True)
