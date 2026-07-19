"""Recurrent model dataset and data module."""

from __future__ import annotations

from typing import Any, Optional, cast

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

    __slots__ = ("_has_sequences", "_is_regression", "aux_features", "labels", "sample_weights", "sequences")

    def __init__(
        self,
        sequences: list[np.ndarray] | None,
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

        # F-51 (2026-05-31): promote the fixed-size tensor attrs to shared
        # memory so DataLoader child workers (num_workers > 0) attach to a
        # single backing buffer instead of pickling a fresh copy each.
        # Mirrors MLP TorchDataset's design (data.py:45 docstring, Yuxin Wu
        # "Demystify RAM Usage" pattern). Per-worker RAM saving is roughly
        # (num_workers - 1) * sum(tensor.nbytes for tensor in promoted).
        # Caveat: self.sequences stays a list of variable-length np arrays
        # because share_memory_() doesn't apply to Python lists; for huge
        # sequence corpora a follow-up should consider concat-tensor +
        # offsets or worker_init_fn shared-memory plumbing. Tensors only
        # promote when on CPU (CUDA tensors are already shared via UVM).
        for _attr in ("labels", "aux_features", "sample_weights"):
            _t = getattr(self, _attr)
            if _t is None or _t.device.type != "cpu":
                continue
            try:
                _t.share_memory_()
            except (RuntimeError, NotImplementedError):
                pass

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {"labels": self.labels[idx]}

        if self._has_sequences and self.sequences is not None:
            # F-H fix (2026-05-31, audit follow-up): torch.as_tensor on a
            # contiguous float32 numpy array does a ZERO-COPY view that
            # shares storage with self.sequences[idx]. If ANY downstream
            # subclass introduces an in-place op on the per-sample tensor
            # (e.g. ``x.relu_()`` in a custom predict_step, or a learned
            # batchnorm scale that touches inputs via .data.add_), the
            # mutation silently corrupts the dataset for the NEXT epoch.
            # Bug LATENT until someone adds an in-place op. Use
            # torch.tensor (always copies) instead so per-sample tensors
            # are independent of the source numpy arrays. Per-sample cost
            # is a single contiguous memcpy; negligible vs the RNN forward.
            item["sequence"] = torch.tensor(self.sequences[idx], dtype=torch.float32)

        if self.aux_features is not None:
            item["aux_features"] = self.aux_features[idx]

        if self.sample_weights is not None:
            item["sample_weights"] = self.sample_weights[idx]

        return item


def recurrent_collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
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
    result: dict[str, torch.Tensor] = {}

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
        train_sequences: list[np.ndarray] | None = None,
        train_features: np.ndarray | None = None,
        train_labels: np.ndarray | None = None,
        train_sample_weight: np.ndarray | None = None,
        val_sequences: list[np.ndarray] | None = None,
        val_features: np.ndarray | None = None,
        val_labels: np.ndarray | None = None,
        val_sample_weight: np.ndarray | None = None,
        test_sequences: list[np.ndarray] | None = None,
        test_features: np.ndarray | None = None,
        test_labels: np.ndarray | None = None,
        batch_size: int = 256,
        num_workers: int = 0,
        is_regression: bool = False,
        use_stratified_sampler: bool = True,
        accelerator: str = "auto",
        prefetch_factor: int = 4,
        pin_memory: Optional[bool] = None,
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
        # F-52: prefetch_factor only takes effect when num_workers > 0 (PyTorch
        # DataLoader requirement). Stored as-is; per-loader emission guards
        # below pass it only when num_workers > 0.
        self.prefetch_factor = prefetch_factor
        # F-47 (2026-05-31): pin_memory should track the trainer accelerator,
        # not the host's CUDA availability. Pinning when the trainer runs on
        # CPU (e.g. user explicitly set accelerator="cpu" on a CUDA box for
        # debugging or smoke tests) wastes RAM and triggers a useless host-
        # side page-lock attempt for every batch. An explicit pin_memory=
        # always wins -- some driver/CUDA-toolkit combos crash during
        # pinned-memory teardown, which GPU auto-detection can't see.
        if pin_memory is not None:
            self._pin_memory = pin_memory
        else:
            self._pin_memory = torch.cuda.is_available() and accelerator in ("auto", "gpu", "cuda")

        # For dynamic prediction
        self.predict_sequences: list[np.ndarray] | None = None
        self.predict_features: np.ndarray | None = None

    def _worker_kwargs(self) -> dict:
        """F-52: extra DataLoader kwargs that only apply when num_workers > 0.

        prefetch_factor isn't accepted by PyTorch DataLoader when num_workers
        is 0 (raises ValueError); same for persistent_workers (warning only,
        but emit nothing for cleanliness). Gathering them once keeps the four
        loader methods symmetric without per-site if/else.
        """
        if self.num_workers <= 0:
            return {}
        return {
            "persistent_workers": True,
            "prefetch_factor": self.prefetch_factor,
        }

    def setup(self, stage: str | None = None):
        """Lightning DataModule hook.

        Intentionally a no-op: this DataModule receives already-prepared
        arrays via ``__init__`` rather than loading from disk per stage,
        so no per-stage setup work is needed. The method exists only to
        satisfy the LightningDataModule interface (Lightning calls
        ``setup(stage)`` before each fit/validate/test/predict).
        """
        return None

    def train_dataloader(self) -> DataLoader:
        """Return training DataLoader."""
        if self.train_labels is None:
            raise ValueError("train_dataloader() requires train_labels to be set.")
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
            # np.bincount requires non-negative contiguous integer labels and
            # returns length max+1 with zeros for any missing label in [0, max].
            # Use np.unique to handle negative labels and non-contiguous label
            # sets ({0, 5}, {-1, 1}) correctly.
            unique_labels, class_counts = np.unique(labels, return_counts=True)
            if len(unique_labels) > 1 and (class_counts > 0).all():
                label_to_weight = {int(lbl): 1.0 / int(cnt) for lbl, cnt in zip(unique_labels, class_counts)}
                sample_weights = np.array([label_to_weight[int(lbl)] for lbl in labels], dtype=np.float64)
                sampler = WeightedRandomSampler(
                    weights=cast(Any, sample_weights),
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
            pin_memory=self._pin_memory,
            **self._worker_kwargs(),
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

        # F-46 (2026-05-31): mirror train_dataloader persistent_workers.
        # EarlyStopping triggers val every epoch; without persistence we
        # re-spawn workers each call (~150-300ms on Windows).
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=recurrent_collate_fn,
            pin_memory=self._pin_memory,
            persistent_workers=self.num_workers > 0,
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

        # F-46: persist test workers across re-runs (mirror train+val).
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=recurrent_collate_fn,
            pin_memory=self._pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self) -> DataLoader:
        """Return prediction DataLoader."""
        if self.predict_sequences is None and self.predict_features is None:
            raise RuntimeError("Call setup_predict() first")

        # Explicit `is not None` check: an empty list is falsy but legitimate
        # (e.g. zero-row predict batch), and falling through to predict_features
        # in that case returns the wrong length or AttributeError when features is None.
        if self.predict_sequences is not None:
            n_samples = len(self.predict_sequences)
        elif self.predict_features is not None:
            n_samples = len(self.predict_features)
        else:
            raise ValueError("predict_dataloader() requires predict_sequences or predict_features to be set.")
        dummy_labels = np.zeros(n_samples, dtype=np.float32 if self.is_regression else np.int64)

        dataset = RecurrentDataset(
            sequences=self.predict_sequences,
            aux_features=self.predict_features,
            labels=dummy_labels,
            is_regression=self.is_regression,
        )

        # F-46: persist predict workers; predict_dataloader is called once
        # per predict() invocation -- and many estimators predict 5-20x per
        # fit suite (val + test + OOF + ensemble probes).
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=recurrent_collate_fn,
            pin_memory=self._pin_memory,
            persistent_workers=self.num_workers > 0,
        )

    def setup_predict(
        self,
        sequences: list[np.ndarray] | None = None,
        features: np.ndarray | None = None,
    ):
        """Stage data for prediction."""
        self.predict_sequences = sequences
        self.predict_features = features
