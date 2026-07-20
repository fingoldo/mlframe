"""
Data handling for PyTorch Lightning models in mlframe.

This module provides:
- TorchDataset: PyTorch Dataset for tabular data with GPU preloading support
- TorchDataModule: Lightning DataModule with train/val/test/predict stages
"""

from __future__ import annotations


import logging
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import polars as pl
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .base import to_tensor_any

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------------
# TorchDataset
# ----------------------------------------------------------------------------------------------------------------------------


class TorchDataset(Dataset):
    """PyTorch ``Dataset`` wrapping feature/label/sample-weight arrays; eager-converts small inputs to tensors up front and defers per-batch conversion for large inputs to bound peak RAM (see the module's byte-size gate)."""

    def __init__(
        self,
        features,
        labels=None,
        sample_weight=None,
        features_dtype: torch.dtype = torch.float32,
        labels_dtype: torch.dtype = torch.float32,
        device: Optional[str] = None,
        batch_size: int = 0,
        share_memory: bool = True,
    ):
        """``share_memory``: when True (default) AND the dataset eager-converts the feature/label tensors to CPU memory, call ``tensor.share_memory_()``
        so PyTorch DataLoader child workers (``num_workers > 0``) attach to the same backing buffer instead of each pickling+receiving a fresh copy.

        Rationale (Yuxin Wu, 2022, "Demystify RAM Usage in Multi-Process Data Loaders"): the naive multi-worker DataLoader pattern leaks memory across
        workers because Linux copy-on-write fails for Python objects - accessing ANY Python attribute increments its refcount, which is a write, which
        triggers a per-process copy of the page. The standard mitigation is to put the dataset's hot buffers into ``torch.Tensor`` so PyTorch's custom
        ``ForkingPickler`` moves them to ``/dev/shm`` (Linux) or a SharedMemoryManager handle (Windows) instead of serialising bytes. ``share_memory_()``
        does that promotion eagerly so the first worker access is zero-copy from the start.

        Disable (``False``) only for tests or for special tensor types (sparse, MPS) where ``share_memory_()`` raises. Negligible cost at
        ``num_workers=0``; with ``num_workers >= 2`` and a 1 GB feature buffer the net RAM saving is ``num_workers * 1 GB``.

        Multi-GPU note: this dataset is single-process / single-GPU. For multi-GPU DDP, designate rank-0 to load data once, then pass the shared-memory
        tensor handle to ranks 1..N-1 via ``ForkingPickler``.

        Note on pinning: PyTorch DataLoader's own ``pin_memory=True`` kwarg performs per-batch pinning (already wired via
        ``_create_dataloader: dl_params["pin_memory"] = on_gpu``). Pinning the WHOLE feature tensor once would lock the buffer's RAM permanently - only
        worth it for small frames AND a GPU target.
        """
        # Wave 56 (2026-05-20): forward to torch Dataset base for forward-compat
        # (currently a no-op; reserves the hook for future torch state).
        super().__init__()
        self.features_dtype = features_dtype
        self.labels_dtype = labels_dtype
        self.device = device
        self.batch_size = batch_size
        self._share_memory = share_memory

        # Eager-convert features to torch.Tensor in __init__ to avoid per-batch isinstance + .to(dtype, device) chain (~38us/call) in __getitem__.
        # Memory-safety: eager conversion copies the underlying buffer. For HUGE frames (100GB+) this would OOM; threshold on byte-size and fall back
        # to the legacy per-batch path above the cap. 2GB is safely below any typical RAM budget AND well above the worst-case 10M-row x 100-col fit.
        _EAGER_TENSOR_BYTES_CAP = 2 * 1024**3  # 2 GB
        try:
            _bytes_estimate = (
                features.nbytes
                if hasattr(features, "nbytes")
                else getattr(features, "estimated_size", lambda: 0)() if hasattr(features, "estimated_size") else 0
            )
        except Exception:
            _bytes_estimate = 0
        self._eager_features = (
            isinstance(features, torch.Tensor)
            or _bytes_estimate == 0  # unknown size, prefer eager (small frame)
            or _bytes_estimate <= _EAGER_TENSOR_BYTES_CAP
            or device == "cuda"  # CUDA path always eager (preload semantics)
        )
        if self._eager_features:
            self.features = to_tensor_any(features, features_dtype, device)
            # Promote to shared memory so DataLoader workers attach to the same buffer (zero-copy across procs). Guarded: some tensor types / devices
            # reject share_memory_().
            if self._share_memory and isinstance(self.features, torch.Tensor) and self.features.device.type == "cpu":
                try:
                    self.features.share_memory_()
                except (RuntimeError, NotImplementedError):
                    pass
        else:
            # Above-cap frame: keep original carrier; ``_extract`` will convert per batch (slow path, but memory-safe).
            self.features = features

        # Handle labels (optional for prediction)
        if labels is not None:
            if isinstance(labels, (pd.DataFrame, pd.Series)):
                labels = labels.to_numpy()
            elif isinstance(labels, pl.DataFrame):
                labels = labels.to_numpy()
            elif not isinstance(labels, (np.ndarray, torch.Tensor)):
                labels = np.asarray(labels)

            # Preserve 2-D labels for multilabel ``(N, K)``. ``reshape(-1)`` would flatten to ``(N*K,)`` and silently break BCEWithLogitsLoss
            # (``target.shape != input.shape``). Pure 1-D labels (regression / single-label classification) keep their original shape.
            # ``pl.DataFrame.to_numpy()`` on a single-column frame returns ``(N, 1)`` - under default ``labels_dtype=int64`` cross_entropy then sees a
            # 2-D Long target and errors with ``Expected floating point type for target with class probabilities, got Long``. Squeeze the trailing
            # length-1 dim so single-label classification delivers ``(N,)`` regardless of whether the upstream carrier was a Series or a 1-col
            # DataFrame. Genuine multilabel ``(N, K>=2)`` is unaffected.
            _arr = np.asarray(labels)
            if _arr.ndim == 1:
                _arr = _arr.reshape(-1)  # explicit 1-D contiguous (no-op for already-1D)
            elif _arr.ndim == 2 and _arr.shape[1] == 1:
                _arr = _arr.reshape(-1)  # collapse (N, 1) -> (N,) for single-label paths
            # 2-D ndarray with K>=2 passes through; ndim>=3 untouched (caller's responsibility)
            self.labels: Optional[torch.Tensor] = torch.tensor(_arr, dtype=labels_dtype, device=device)
            # Same shared-memory promotion as features (labels are accessed per-batch alongside features).
            if self._share_memory and isinstance(self.labels, torch.Tensor) and self.labels.device.type == "cpu":
                try:
                    self.labels.share_memory_()
                except (RuntimeError, NotImplementedError):
                    pass
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
            self.sample_weight: Optional[torch.Tensor] = torch.tensor(sample_weight, dtype=torch.float32, device=device)
            if self._share_memory and isinstance(self.sample_weight, torch.Tensor) and self.sample_weight.device.type == "cpu":
                try:
                    self.sample_weight.share_memory_()
                except (RuntimeError, NotImplementedError):
                    pass
        else:
            self.sample_weight = None

        if batch_size > 0:
            self.num_batches = int(np.ceil(dataset_length / batch_size))

        self.dataset_length = dataset_length

    def __len__(self):
        return self.num_batches if self.batch_size > 0 else self.dataset_length

    def _extract(self, data, indices):
        """Extract and convert subset to tensor.

        Features are normally torch.Tensor (converted once in __init__), so this fastpaths to a single indexing op. The non-Tensor branches stay for
        the rare case where ``self.features`` is replaced post-init by a caller (defensive; not exercised by the mlframe suite).
        """
        if isinstance(data, torch.Tensor):
            # Hot path: features already a tensor of the right dtype/device from __init__'s to_tensor_any call.
            return data[indices]
        if isinstance(data, np.ndarray):
            subset = torch.from_numpy(data[indices])
        elif isinstance(data, pd.DataFrame):
            subset = torch.from_numpy(data.iloc[indices, :].to_numpy())
        elif isinstance(data, pl.DataFrame):
            subset = data[indices].to_torch()
        else:
            raise TypeError(f"Unsupported data type for extraction: {type(data)}")
        return subset.to(dtype=self.features_dtype, device=self.device)

    def __getitem__(self, idx):
        if self.batch_size > 0:
            start = idx * self.batch_size
            end = min((idx + 1) * self.batch_size, self.dataset_length)
            indices = slice(start, end)
        else:
            indices = idx

        x = self._extract(self.features, indices)

        if self.labels is None:
            return x

        y = self.labels[indices]

        # Squeeze single-sample dimension only in sample mode
        if self.batch_size == 0 and x.ndim == 2 and x.shape[0] == 1:
            x = x.squeeze(0)

        if self.sample_weight is not None:
            w = self.sample_weight[indices]
            return x, y, w

        return x, y

    def __getitems__(self, indices):
        """Defensive batched-fetch path for callers that pair this dataset with
        a custom ``batch_sampler`` (PyTorch >= 1.13 calls ``__getitems__`` when
        present instead of running ``[dataset[i] for i in indices]``).

        Standard mlframe flow doesn't reach this method: ``TorchDataModule``
        configures the DataLoader with ``batch_size=None`` and lets this
        dataset's own ``batch_size>0`` internal batching produce one stacked
        batch per integer index — DataLoader iterates ints, not lists. Added
        for protocol completeness so an external user pairing
        ``TorchDataset(batch_size=0)`` with a list-yielding sampler (akin to
        the LTR ``GroupBatchSampler``) gets one batched tensor index instead
        of N per-row index calls.

        Internal-batch mode delegates per-item — each ``__getitem__`` call
        already returns a stacked batch and the DataLoader semantics would
        not double-batch. Sample mode batch-indexes the underlying tensors
        once and emits a list of per-row tuples so ``default_collate`` stacks
        them in the standard way (no custom collate required at the
        DataLoader side).
        """
        if self.batch_size > 0:
            return [self[i] for i in indices]
        idx_list = indices if isinstance(indices, list) else list(indices)
        x_batch = self._extract(self.features, idx_list)
        if self.labels is None:
            return [x_batch[i] for i in range(x_batch.shape[0])]
        y_batch = self.labels[idx_list]
        if self.sample_weight is None:
            return [(x_batch[i], y_batch[i]) for i in range(x_batch.shape[0])]
        w_batch = self.sample_weight[idx_list]
        return [(x_batch[i], y_batch[i], w_batch[i]) for i in range(x_batch.shape[0])]


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
        Initialize DataModule. Data pre-loading here allows automatic sharing between spawned processes via shared memory when using 'ddp_spawn'.
        """
        super().__init__()

        if data_placement_device is not None:
            if not data_placement_device.startswith("cuda"):
                raise ValueError(f"data_placement_device must be None or 'cuda'/'cuda:X', got: {data_placement_device}")

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
        self.predict_features: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None

        self.batch_size = self.dataloader_params.get("batch_size", 64)

    @staticmethod
    def _infer_n_features(features) -> Optional[int]:
        """Best-effort feature width probe; cheap for numpy, pandas, polars."""
        try:
            if hasattr(features, "shape") and len(features.shape) >= 2:
                return int(features.shape[1])
            if hasattr(features, "columns"):
                return len(features.columns)
        except Exception:
            return None
        return None

    def _resolve_batch_size(self, batch_size, features, split_name: str) -> int:
        """Resolve ``batch_size`` to a concrete int: passes an explicit int through, or auto-sizes via ``resolve_mlp_train_batch_size`` (based on feature width + available memory) when ``"auto"`` is requested."""
        if isinstance(batch_size, str):
            if batch_size.lower() != "auto":
                raise ValueError(f"Unsupported MLP DataLoader batch_size={batch_size!r}; " "expected an int or 'auto'.")
            try:
                from mlframe.training.mlp_runtime_defaults import (
                    resolve_mlp_train_batch_size,
                )
                n_features = self._infer_n_features(features)
                resolved = resolve_mlp_train_batch_size(n_features=n_features)
            except Exception:
                n_features = self._infer_n_features(features)
                resolved = 1024
            logger.info(
                "MLP %s DataLoader auto-selected batch_size=%s (n_features=%s)",
                split_name,
                resolved,
                n_features if n_features is not None else "unknown",
            )
            return max(1, int(resolved))
        return max(0, int(batch_size))

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
            if self.trainer is None:
                return False
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
        split_name: str = "data",
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
            TorchDataset handles batching internally when batch_size > 0, so DataLoader is created with batch_size=None to iterate over pre-batched
            data. This enables vectorized extraction and GPU preloading.
        """
        device = self._get_device()
        on_gpu = self.on_gpu()

        # Extract batch_size for TorchDataset (it handles batching internally)
        dl_params = self.dataloader_params.copy()
        batch_size = dl_params.pop("batch_size", self.batch_size)
        batch_size = self._resolve_batch_size(batch_size, features, split_name)

        dl_params["shuffle"] = shuffle
        dl_params["drop_last"] = drop_last
        # setdefault (not unconditional assignment): some driver/CUDA-toolkit combos crash during
        # CachingHostAllocator/CUDAEvent teardown when pinned memory is in play (observed as a
        # fatal std::terminate from a tensor destructor, not a catchable Python exception) -- a
        # caller-supplied dataloader_params["pin_memory"]=False must be the only way out since GPU
        # detection alone can't distinguish a healthy pinned-memory path from a broken one.
        # MLFRAME_MLP_PIN_MEMORY (if set) wins over the on_gpu auto-detect but not over an explicit
        # per-call dataloader_params value, letting a whole process/test run opt out at once.
        from mlframe.training.mlp_runtime_defaults import pin_memory_env_override

        _env_pin = pin_memory_env_override()
        dl_params.setdefault("pin_memory", on_gpu if _env_pin is None else _env_pin)

        # F-71b (2026-05-31): OS-aware num_workers default for MLP, mirrors
        # F-71's RecurrentConfig.num_workers logic. Windows spawn semantics
        # cost ~150-300 ms per worker per epoch, so stay at 0. Linux + macOS
        # fork is cheap; default to 2. With TorchDataset's share_memory_()
        # promotion (data.py:101) workers attach to the same backing buffer
        # at zero copy cost, so num_workers=2 is essentially free on Linux
        # even when the eager-tensor path is in use. Slow-path frames (>2GB
        # cap, per-batch conversion) benefit from workers more directly.
        # Only set as a fallback -- the user-supplied ``dataloader_params``
        # value still wins.
        import os as _os
        dl_params.setdefault(
            "num_workers", 0 if _os.name == "nt" else 2,
        )

        # ``persistent_workers=True`` skips the worker-restart cost between epochs (each restart re-imports torch / lightning / numpy + re-attaches to
        # the shared-memory tensor handle, ~100-300 ms / restart on Windows). At typical 30-100 epoch fits with num_workers>=2 the saving is 3-30 s
        # per fit. Only set when num_workers > 0 - PyTorch warns if you ask for persistent workers with 0 workers.
        if dl_params.get("num_workers", 0) > 0:
            dl_params.setdefault("persistent_workers", True)

        # IMPORTANT: Set batch_size=None since TorchDataset handles batching internally when its own batch_size parameter > 0
        dl_params["batch_size"] = None

        dataset = TorchDataset(
            features=features,
            labels=labels,
            sample_weight=sample_weight,
            features_dtype=self.features_dtype,
            labels_dtype=self.labels_dtype,
            batch_size=batch_size,
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
            split_name="train",
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        return self._create_dataloader(
            features=self.val_features,
            labels=self.val_labels,
            sample_weight=self.val_sample_weight,
            shuffle=False,
            drop_last=False,
            split_name="val",
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
            split_name="test",
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
            labels=None,
            shuffle=False,
            drop_last=False,
            split_name="predict",
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

        if batch_size is not None:
            self.batch_size = batch_size
            # _create_dataloader resolves the batch from
            # ``dataloader_params['batch_size']`` whenever that key is present
            # (the suite seeds it with 'auto'), which would shadow this predict
            # override and route the predict split through the TRAIN batch-size
            # resolver. Mirror the override into dataloader_params so the predict
            # dataloader honors the resolved predict batch size. (New dict, so the
            # shared/original dataloader_params is left untouched.)
            self.dataloader_params = {**self.dataloader_params, "batch_size": batch_size}

        self.setup(stage="predict")

    def has_test_data(self) -> bool:
        """Check if test data is available."""
        return self.test_features is not None

    def get_feature_dim(self) -> int:
        """Get the feature dimension from training data."""
        features = self.train_features

        if isinstance(features, (pd.DataFrame, np.ndarray)):
            return int(features.shape[1])
        elif isinstance(features, torch.Tensor):
            return int(features.shape[1])
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
        except Exception as e:
            logger.debug("num_classes probe failed on labels of type %s (%s: %s)", type(labels).__name__, type(e).__name__, e)
            return None

        return None
