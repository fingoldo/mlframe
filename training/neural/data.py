"""
Data handling for PyTorch Lightning models in mlframe.

This module provides:
- TorchDataset: PyTorch Dataset for tabular data with GPU preloading support
- TorchDataModule: Lightning DataModule with train/val/test/predict stages
"""

# ----------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------------

from typing import *

import torch
from torch.utils.data import DataLoader, Dataset

from lightning import LightningDataModule

import pandas as pd
import numpy as np
import polars as pl


# Local imports
from .base import to_tensor_any


# ----------------------------------------------------------------------------------------------------------------------------
# TorchDataset
# ----------------------------------------------------------------------------------------------------------------------------


class TorchDataset(Dataset):
    def __init__(
        self,
        features,
        labels=None,  # Make optional
        sample_weight=None,  # Optional sample weights
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

        # Handle sample weights (optional)
        if sample_weight is not None:
            if isinstance(sample_weight, (pd.DataFrame, pd.Series)):
                sample_weight = sample_weight.to_numpy()
            elif isinstance(sample_weight, pl.DataFrame):
                sample_weight = sample_weight.to_numpy()
            elif not isinstance(sample_weight, (np.ndarray, torch.Tensor)):
                sample_weight = np.asarray(sample_weight)

            sample_weight = np.asarray(sample_weight).reshape(-1)
            self.sample_weight = torch.tensor(sample_weight, dtype=torch.float32, device=device)
        else:
            self.sample_weight = None

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

        # Return with sample weights if available
        if self.sample_weight is not None:
            w = self.sample_weight[indices]
            return x, y, w

        return x, y


# ----------------------------------------------------------------------------------------------------------------------------
# TorchDataModule
# ----------------------------------------------------------------------------------------------------------------------------


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
        train_features: Optional[Union[pd.DataFrame, np.ndarray, str]] = None,
        train_labels: Optional[Union[pd.DataFrame, np.ndarray, str]] = None,
        train_sample_weight: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        val_features: Optional[Union[pd.DataFrame, np.ndarray, str]] = None,
        val_labels: Optional[Union[pd.DataFrame, np.ndarray, str]] = None,
        val_sample_weight: Optional[Union[pd.DataFrame, np.ndarray]] = None,
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
        self.train_sample_weight = train_sample_weight
        self.val_features = val_features
        self.val_labels = val_labels
        self.val_sample_weight = val_sample_weight
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
        sample_weight=None,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        """
        Create a DataLoader with consistent configuration.

        Args:
            features: Feature data
            labels: Label data (optional, None for prediction)
            sample_weight: Sample weights (optional)
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
            sample_weight=sample_weight,
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
            sample_weight=self.train_sample_weight,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        return self._create_dataloader(
            features=self.val_features,
            labels=self.val_labels,
            sample_weight=self.val_sample_weight,
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
