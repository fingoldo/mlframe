"""Recurrent model dataset and data module."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from lightning.pytorch import LightningDataModule

from .base import _ensure_numpy
from ._recurrent_config import InputMode  # noqa: F401  # re-exported for callers


class RecurrentDataset(Dataset):
    """
    Dataset for variable-length sequences with optional auxiliary features.

    Handles variable-length sequences and optional tabular features.
    Supports both classification (integer labels) and regression (float labels).
    """

    __slots__ = ("sequences", "aux_features", "labels", "sample_weights", "_has_sequences", "_is_regression")

    def __init__(
        self,
        sequences: List[np.ndarray] | None,
        aux_features: np.ndarray | None,
        labels: np.ndarray,
        sample_weights: np.ndarray | None = None,
        is_regression: bool = False,
    ) -> None:
        """
        Initialize dataset.

        Args:
            sequences: list of (seq_len, n_features) arrays, or None
            aux_features: (n_samples, n_features) array, or None
            labels: (n_samples,) array of labels
            sample_weights: (n_samples,) array of weights, or None
            is_regression: whether this is a regression task (affects label dtype)
        """
        self.sequences = sequences
        self._has_sequences = sequences is not None
        self._is_regression = is_regression

        # Label dtype rules:
        #   regression               -> float32 (MSELoss)
        #   2-D labels (multilabel)  -> float32 (BCEWithLogitsLoss expects float matching logits)
        #   1-D labels (binary/multiclass) -> int64 (CrossEntropyLoss expects class indices)
        labels_np = np.asarray(labels)
        if is_regression:
            self.labels = torch.as_tensor(labels_np, dtype=torch.float32)
        elif labels_np.ndim == 2:
            self.labels = torch.as_tensor(labels_np, dtype=torch.float32)
        else:
            self.labels = torch.as_tensor(labels_np, dtype=torch.long)

        # Pre-convert fixed-size arrays to tensors
        aux_np = _ensure_numpy(aux_features)
        self.aux_features = torch.as_tensor(aux_np, dtype=torch.float32) if aux_np is not None else None

        weights_np = _ensure_numpy(sample_weights)
        self.sample_weights = torch.as_tensor(weights_np, dtype=torch.float32) if weights_np is not None else None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {"labels": self.labels[idx]}

        if self._has_sequences:
            item["sequence"] = torch.as_tensor(self.sequences[idx], dtype=torch.float32)

        if self.aux_features is not None:
            item["aux_features"] = self.aux_features[idx]

        if self.sample_weights is not None:
            item["sample_weights"] = self.sample_weights[idx]

        return item


def recurrent_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function handling variable-length sequences. Pads to max length in batch.

    Args:
        batch: list of sample dicts from dataset

    Returns:
        Collated batch dict with:
        - sequences: (batch, max_len, n_features) padded tensor
        - lengths: (batch,) original lengths
        - aux_features: (batch, n_features) if present
        - labels: (batch,)
        - sample_weights: (batch,) if present
    """
    result: Dict[str, torch.Tensor] = {}

    # Labels (always present)
    result["labels"] = torch.stack([item["labels"] for item in batch])

    # Sequences (variable length)
    if "sequence" in batch[0]:
        sequences = [item["sequence"] for item in batch]
        lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
        padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        result["sequences"] = padded
        result["lengths"] = lengths

    # Auxiliary features
    if "aux_features" in batch[0]:
        result["aux_features"] = torch.stack([item["aux_features"] for item in batch])

    # Sample weights
    if "sample_weights" in batch[0]:
        result["sample_weights"] = torch.stack([item["sample_weights"] for item in batch])

    return result


# ----------------------------------------------------------------------------------------------------------------------------
# RecurrentDataModule
# ----------------------------------------------------------------------------------------------------------------------------


class RecurrentDataModule(LightningDataModule):
    """
    Lightning DataModule for recurrent models with sequence data.

    Handles train/val/test/predict stages with proper sequence handling.
    """

    def __init__(
        self,
        train_sequences: Optional[List[np.ndarray]] = None,
        train_features: Optional[np.ndarray] = None,
        train_labels: Optional[np.ndarray] = None,
        train_sample_weight: Optional[np.ndarray] = None,
        val_sequences: Optional[List[np.ndarray]] = None,
        val_features: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        val_sample_weight: Optional[np.ndarray] = None,
        test_sequences: Optional[List[np.ndarray]] = None,
        test_features: Optional[np.ndarray] = None,
        test_labels: Optional[np.ndarray] = None,
        batch_size: int = 256,
        num_workers: int = 0,
        is_regression: bool = False,
        use_stratified_sampler: bool = True,
    ):
        """
        Initialize DataModule.

        Args:
            train_sequences: training sequences (list of variable-length arrays)
            train_features: training tabular features
            train_labels: training labels
            train_sample_weight: training sample weights
            val_sequences: validation sequences
            val_features: validation tabular features
            val_labels: validation labels
            val_sample_weight: validation sample weights
            test_sequences: test sequences
            test_features: test tabular features
            test_labels: test labels
            batch_size: batch size for DataLoaders
            num_workers: number of workers for DataLoaders
            is_regression: whether this is a regression task
            use_stratified_sampler: use weighted sampling for imbalanced data
        """
        super().__init__()
        self.train_sequences = train_sequences
        self.train_features = train_features
        self.train_labels = train_labels
        self.train_sample_weight = train_sample_weight
        self.val_sequences = val_sequences
        self.val_features = val_features
        self.val_labels = val_labels
        self.val_sample_weight = val_sample_weight
        self.test_sequences = test_sequences
        self.test_features = test_features
        self.test_labels = test_labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_regression = is_regression
        self.use_stratified_sampler = use_stratified_sampler

        # For dynamic prediction
        self.predict_sequences = None
        self.predict_features = None

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage (data is already provided in __init__)."""
        pass

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        dataset = RecurrentDataset(
            sequences=self.train_sequences,
            aux_features=self.train_features,
            labels=self.train_labels,
            sample_weights=self.train_sample_weight,
            is_regression=self.is_regression,
        )

        sampler = None
        shuffle = True

        # Stratified weighted sampling for imbalanced classification data;
        # sampler is mutually exclusive with shuffle=True in PyTorch DataLoader, hence shuffle=False when sampler is set.
        if self.use_stratified_sampler and not self.is_regression:
            labels = dataset.labels.numpy()
            class_counts = np.bincount(labels)
            if len(class_counts) > 1 and all(c > 0 for c in class_counts):
                class_weights = 1.0 / class_counts
                sample_weights = class_weights[labels]
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(dataset),
                    replacement=True,
                )
                shuffle = False

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=recurrent_collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation DataLoader."""
        if self.val_labels is None:
            raise RuntimeError("Validation labels not provided")

        dataset = RecurrentDataset(
            sequences=self.val_sequences,
            aux_features=self.val_features,
            labels=self.val_labels,
            sample_weights=self.val_sample_weight,
            is_regression=self.is_regression,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=recurrent_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        """Return test DataLoader."""
        if self.test_labels is None:
            raise RuntimeError("Test labels not provided")

        dataset = RecurrentDataset(
            sequences=self.test_sequences,
            aux_features=self.test_features,
            labels=self.test_labels,
            is_regression=self.is_regression,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=recurrent_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def predict_dataloader(self) -> DataLoader:
        """Return prediction DataLoader."""
        if self.predict_sequences is None and self.predict_features is None:
            raise RuntimeError("Call setup_predict() first")

        n_samples = len(self.predict_sequences) if self.predict_sequences else len(self.predict_features)
        dummy_labels = np.zeros(n_samples, dtype=np.float32 if self.is_regression else np.int64)

        dataset = RecurrentDataset(
            sequences=self.predict_sequences,
            aux_features=self.predict_features,
            labels=dummy_labels,
            is_regression=self.is_regression,
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=recurrent_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def setup_predict(
        self,
        sequences: Optional[List[np.ndarray]] = None,
        features: Optional[np.ndarray] = None,
    ):
        """Stage data for prediction."""
        self.predict_sequences = sequences
        self.predict_features = features
