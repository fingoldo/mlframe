"""MLP-based learning-to-rank (LTR) model with RankNet / ListNet pairwise/listwise loss.

The model is sklearn-shaped (``fit(X, y, group_ids=...)`` + ``predict(X)``)
so it slots into ``mlframe.training.ranking::fit_ranker`` alongside
``CatBoostRanker``, ``XGBRanker``, and ``LGBMRanker``.

Two loss functions are exposed:

- **RankNet** (Burges 2005): pairwise ``BCEWithLogitsLoss`` on score
  differences. For every pair (i, j) within the same query where
  ``rel_i > rel_j``, we minimise ``-log sigmoid(score_i - score_j)``
  (equivalently ``BCE(s_i - s_j, target=1.0)``). Numerically stable
  (``log(sigmoid)`` would overflow for large |s_diff|).

- **ListNet** (Cao 2007): listwise softmax cross-entropy between the
  predicted top-1 distribution and the relevance-induced one. O(n) per
  query; weaker than LambdaRank on highly graded data but simpler.

Group-aware batching is via ``GroupBatchSampler`` which yields ONE query
per batch (skipping queries with <2 docs or single-class relevance --
no positive pairs possible). Each ``training_step`` thus receives the
full doc list of one query, computes all pairs (RankNet) or the listwise
loss (ListNet), and backprops once.

This matches the contract used by ``mlframe.training.ranking``:
- ``fit(X, y, group_ids, X_val=..., y_val=..., group_ids_val=..., ...)``
- ``predict(X)`` -> 1-D per-row scores

Verified against installed PyTorch + Lightning. The sklearn wrapper is
the unit shipped to the suite; the LightningModule inside is an
implementation detail.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from sklearn.base import BaseEstimator, RegressorMixin

from mlframe.training.neural.flat import generate_mlp

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------------------


def ranknet_pairwise_loss(scores: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
    """RankNet pairwise loss for one query.

    Parameters
    ----------
    scores : (N,) tensor of model scores for the query's N docs.
    relevance : (N,) tensor of integer (or float) ground-truth relevance.

    Returns
    -------
    Scalar loss tensor. Returns 0.0 when no informative pair exists
    (all docs same relevance).
    """
    if scores.dim() != 1:
        scores = scores.view(-1)
    n = scores.shape[0]
    if n < 2:
        return scores.new_zeros(())

    # Build all (i, j) pairs where rel_i > rel_j.
    rel = relevance.view(-1).to(scores.dtype)
    # Pairwise diff matrices (N, N).
    rel_diff = rel.unsqueeze(1) - rel.unsqueeze(0)  # rel_i - rel_j
    score_diff = scores.unsqueeze(1) - scores.unsqueeze(0)
    # Mask: pairs where rel_i strictly > rel_j (one-sided to avoid double-counting).
    pair_mask = (rel_diff > 0)
    if not pair_mask.any():
        return scores.new_zeros(())

    # BCEWithLogitsLoss on s_i - s_j with target=1 (i should rank above j).
    # Numerically stable via the built-in F.binary_cross_entropy_with_logits.
    target = torch.ones_like(score_diff)
    # Compute element-wise loss then mean over informative pairs only.
    bce = F.binary_cross_entropy_with_logits(score_diff, target, reduction="none")
    return bce[pair_mask].mean()


def listnet_top1_loss(scores: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:
    """ListNet top-1 listwise loss for one query.

    KL divergence between the relevance-induced softmax distribution
    (true) and the predicted-score softmax distribution. O(n) per query.
    """
    if scores.dim() != 1:
        scores = scores.view(-1)
    rel = relevance.view(-1).to(scores.dtype)
    n = scores.shape[0]
    if n < 2:
        return scores.new_zeros(())

    # Skip degenerate queries (all-equal relevance -> uniform target,
    # KL not informative).
    if (rel == rel[0]).all():
        return scores.new_zeros(())

    # softmax(target=relevance) vs softmax(predicted_scores)
    # ListNet original paper: use cross-entropy of true_top1 with
    # predicted_top1.
    true_p = F.softmax(rel, dim=0)
    pred_log_p = F.log_softmax(scores, dim=0)
    return -(true_p * pred_log_p).sum()


# ----------------------------------------------------------------------------------
# Group-aware batching
# ----------------------------------------------------------------------------------


class GroupBatchSampler(Sampler):
    """Yield ONE query's row indices per batch.

    Skips queries that:
      - have fewer than 2 docs (no pair possible);
      - have a single distinct relevance value (no positive pair).
    The relevance threshold means even valid 2-doc queries with both
    rel=0 (or both rel=1) are skipped -- they contribute 0 to the loss
    AND cause numerical issues in some Lightning callbacks.

    The sampler is deterministic when ``shuffle=False``. With
    ``shuffle=True`` it yields queries in a per-epoch random order
    (within-query doc order is preserved).
    """

    def __init__(
        self,
        group_ids: np.ndarray,
        relevance: np.ndarray,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.group_ids = np.asarray(group_ids)
        self.relevance = np.asarray(relevance)
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        # Pre-build per-query slices and skip-mask.
        sort_idx = np.argsort(self.group_ids, kind="stable")
        sorted_groups = self.group_ids[sort_idx]
        boundaries = np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1
        starts = np.concatenate(([0], boundaries, [len(sorted_groups)]))
        self._query_slices: List[np.ndarray] = []
        for i in range(len(starts) - 1):
            s, e = starts[i], starts[i + 1]
            indices = sort_idx[s:e]
            if len(indices) < 2:
                continue
            rel_slice = self.relevance[indices]
            if rel_slice.ndim > 1:
                # Multilabel-shaped relevance not supported here; coerce
                # to 1-D by summing labels (any non-zero = relevant).
                rel_slice = rel_slice.sum(axis=1)
            if len(np.unique(rel_slice)) < 2:
                continue
            self._query_slices.append(indices)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self):
        order = np.arange(len(self._query_slices))
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self._epoch)
            rng.shuffle(order)
        for i in order:
            yield self._query_slices[i].tolist()

    def __len__(self) -> int:
        return len(self._query_slices)


class _RankerDataset(Dataset):
    """Dataset wrapping (X, y, group_ids) for the GroupBatchSampler.

    ``__getitem__(i)`` returns the i-th row's features + label.
    The sampler controls grouping; the dataset is row-level.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------------------------------------------------------------
# Lightning module
# ----------------------------------------------------------------------------------


def _import_lightning():
    try:
        import lightning.pytorch as L
        return L
    except ImportError:
        import pytorch_lightning as L  # legacy package name
        return L


def _make_lightning_module(network: nn.Module, loss_name: str, learning_rate: float):
    L = _import_lightning()

    class MLPRankerLightningModule(L.LightningModule):
        def __init__(self, network, loss_name: str, learning_rate: float):
            super().__init__()
            self.network = network
            self.loss_name = loss_name
            self.learning_rate = learning_rate
            self.save_hyperparameters(ignore=["network"])

        def forward(self, x):
            return self.network(x).view(-1)

        def training_step(self, batch, batch_idx):
            x, y = batch
            scores = self.forward(x)
            if self.loss_name == "ranknet":
                loss = ranknet_pairwise_loss(scores, y)
            elif self.loss_name == "listnet":
                loss = listnet_top1_loss(scores, y)
            else:
                raise ValueError(f"unknown loss_name {self.loss_name!r}")
            self.log("train_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            with torch.no_grad():
                scores = self.forward(x)
                if self.loss_name == "ranknet":
                    loss = ranknet_pairwise_loss(scores, y)
                else:
                    loss = listnet_top1_loss(scores, y)
            self.log("val_loss", loss, prog_bar=False, on_epoch=True, on_step=False)
            return loss

        def predict_step(self, batch, batch_idx):
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            with torch.no_grad():
                return self.forward(x)

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

    return MLPRankerLightningModule(network, loss_name, learning_rate)


# ----------------------------------------------------------------------------------
# sklearn-shaped wrapper
# ----------------------------------------------------------------------------------


class MLPRanker(BaseEstimator, RegressorMixin):
    """sklearn-shaped MLP ranker. Returns per-row 1-D scores via ``predict``.

    Hyperparameters (sensible defaults):
        loss_fn      : "ranknet" (default) or "listnet"
        n_estimators : iterations / epochs (default 100)
        learning_rate: AdamW lr (default 1e-3)
        hidden_layers: tuple of hidden-layer sizes (default (64, 64))
        dropout      : (default 0.1)
        early_stopping_patience: stop when val_loss doesn't improve (default 10).
                                 None disables early stopping.

    Extra kwargs documented in ``generate_mlp``.
    """

    def __init__(
        self,
        loss_fn: str = "ranknet",
        n_estimators: int = 100,
        learning_rate: float = 1e-3,
        hidden_layers: Tuple[int, ...] = (64, 64),
        dropout: float = 0.1,
        early_stopping_patience: Optional[int] = 10,
        seed: int = 42,
        verbose: int = 0,
    ):
        self.loss_fn = loss_fn
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed
        self.verbose = verbose

    def _x_to_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            # Drop non-numeric columns silently (qid / target columns
            # should already be removed by the caller; this is defence-in-depth).
            X = X.select_dtypes(include=[np.number])
            return X.to_numpy(dtype=np.float32, copy=False)
        return np.asarray(X, dtype=np.float32)

    def fit(
        self,
        X,
        y,
        group_ids,
        X_val=None,
        y_val=None,
        group_ids_val=None,
        cat_features=None,
    ):
        """Fit the ranker. ``cat_features`` is accepted for signature
        symmetry with other rankers; categorical columns must already be
        numeric-encoded by the caller (MLPRanker doesn't support raw
        categoricals)."""
        L = _import_lightning()

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X_arr = self._x_to_array(X)
        y_arr = np.asarray(y, dtype=np.float32).ravel()
        n_features = X_arr.shape[1]

        # Network: simple MLP, single output (score).
        # We reuse ``generate_mlp`` with num_classes=1 (single output).
        network = generate_mlp(
            num_features=n_features,
            num_classes=1,
            nlayers=max(1, len(self.hidden_layers)),
            first_layer_num_neurons=self.hidden_layers[0] if self.hidden_layers else max(8, n_features),
            min_layer_neurons=1,
            dropout_prob=self.dropout,
            inputs_dropout_prob=0.0,
            use_layernorm=True,
            use_batchnorm=False,  # group-batching breaks BN stats; LN is safe
            verbose=0,
        )

        train_ds = _RankerDataset(X_arr, y_arr)
        train_sampler = GroupBatchSampler(
            group_ids=np.asarray(group_ids), relevance=y_arr,
            shuffle=True, seed=self.seed,
        )
        if len(train_sampler) == 0:
            raise ValueError(
                "MLPRanker.fit: zero usable training queries (every query "
                "has <2 docs OR single-class relevance). Increase rows per "
                "query OR ensure y has graded values."
            )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            num_workers=0,
        )

        val_loader = None
        if X_val is not None and y_val is not None and group_ids_val is not None:
            X_val_arr = self._x_to_array(X_val)
            y_val_arr = np.asarray(y_val, dtype=np.float32).ravel()
            val_ds = _RankerDataset(X_val_arr, y_val_arr)
            val_sampler = GroupBatchSampler(
                group_ids=np.asarray(group_ids_val), relevance=y_val_arr,
                shuffle=False, seed=self.seed,
            )
            if len(val_sampler) > 0:
                val_loader = DataLoader(
                    val_ds, batch_sampler=val_sampler, num_workers=0,
                )

        self.module_ = _make_lightning_module(
            network, self.loss_fn, self.learning_rate,
        )

        callbacks = []
        if self.early_stopping_patience is not None and val_loader is not None:
            try:
                from lightning.pytorch.callbacks import EarlyStopping
            except ImportError:
                from pytorch_lightning.callbacks import EarlyStopping
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss", patience=self.early_stopping_patience,
                    mode="min", verbose=False,
                )
            )

        trainer = L.Trainer(
            max_epochs=self.n_estimators,
            enable_model_summary=False,
            enable_progress_bar=bool(self.verbose),
            logger=False,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=1,
        )
        trainer.fit(
            model=self.module_,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )
        self.trainer_ = trainer
        self.n_features_in_ = n_features
        return self

    def predict(self, X) -> np.ndarray:
        """Return per-row scores (1-D, higher=more relevant)."""
        X_arr = self._x_to_array(X)
        self.module_.eval()
        device = next(self.module_.parameters()).device
        with torch.no_grad():
            x_t = torch.as_tensor(X_arr, dtype=torch.float32, device=device)
            scores = self.module_.forward(x_t).cpu().numpy()
        return scores.ravel()


__all__ = [
    "MLPRanker",
    "GroupBatchSampler",
    "ranknet_pairwise_loss",
    "listnet_top1_loss",
]
