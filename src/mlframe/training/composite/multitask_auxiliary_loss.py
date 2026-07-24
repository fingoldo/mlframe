"""``MultiTaskAuxiliaryLossRegressor``: shared-trunk NN, primary regression head + weighted auxiliary heads.

Source: LANL Earthquake Prediction 1st place -- an NN trained with time-to-failure MAE loss plus an
additional binary logloss ("ttf < 0.5") and an additional MAE loss on "time-since-failure", weighted lower,
specifically to fix weird spikes near event boundaries. Genuine multi-task learning: one shared trunk, three
heads, one joint backpropagated loss (weighted sum) -- distinct from every OTHER stacking/blending estimator
built this session (those all fit SEPARATE models and combine their outputs post-hoc; here the auxiliary
signal shapes the SAME shared representation the primary head reads from, during training itself).

Deliberately built as a small self-contained raw-PyTorch MLP rather than wired into mlframe's full
PyTorch-Lightning NN infra (``training/neural/``, confirmed via search-for-reuse to support only HOMOGENEOUS
multi-output regression/multi-label heads, one loss type, not a weighted sum of heterogeneous named losses)
-- same lower-integration-risk rationale already used for the swap-noise DAE (``denoising_autoencoder.py``):
avoids touching Lightning's large, heavily-tested training loop for a bottleneck-scale use case.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)


class MultiTaskAuxiliaryLossRegressor(BaseEstimator, RegressorMixin):
    """Shared-trunk MLP with a primary regression head plus optional auxiliary classification/regression
    heads, all heads' losses summed (auxiliary heads weighted lower) and backpropagated jointly.

    Parameters
    ----------
    hidden_sizes
        Shared-trunk hidden layer widths.
    aux_task_weight
        Weight applied to EACH auxiliary head's loss in the joint sum (the primary head's loss always has
        weight 1.0, matching the source's "weighted lower" auxiliary losses).
    n_epochs, lr, batch_size
        Training configuration (full-batch Adam by default, ``batch_size=None``).
    random_state
        Seed for weight init and any minibatch shuffling.

    Attributes
    ----------
    train_losses_
        Per-epoch joint training loss, for diagnostics.
    """

    def __init__(
        self,
        hidden_sizes: tuple = (32, 16),
        aux_task_weight: float = 0.3,
        n_epochs: int = 300,
        lr: float = 0.01,
        batch_size: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.aux_task_weight = aux_task_weight
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.random_state = random_state

    def _build_trunk_and_heads(self, n_features: int, has_aux_binary: bool, has_aux_regression: bool):
        """Build the shared MLP trunk plus the primary head and any requested auxiliary heads."""
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        layers = []
        prev = n_features
        for h in self.hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        trunk = nn.Sequential(*layers)
        primary_head = nn.Linear(prev, 1)
        aux_binary_head = nn.Linear(prev, 1) if has_aux_binary else None
        aux_regression_head = nn.Linear(prev, 1) if has_aux_regression else None
        return trunk, primary_head, aux_binary_head, aux_regression_head

    def fit(
        self,
        X: np.ndarray,
        y_primary: np.ndarray,
        y_aux_binary: Optional[np.ndarray] = None,
        y_aux_regression: Optional[np.ndarray] = None,
    ) -> "MultiTaskAuxiliaryLossRegressor":
        """Train the shared trunk jointly on the primary loss plus any weighted auxiliary-task losses."""
        import torch
        import torch.nn as nn

        X_arr = np.asarray(X, dtype=np.float32)
        y_primary_arr = np.asarray(y_primary, dtype=np.float32).reshape(-1, 1)
        has_aux_binary = y_aux_binary is not None
        has_aux_regression = y_aux_regression is not None

        self.trunk_, self.primary_head_, self.aux_binary_head_, self.aux_regression_head_ = self._build_trunk_and_heads(
            X_arr.shape[1], has_aux_binary, has_aux_regression
        )

        params = list(self.trunk_.parameters()) + list(self.primary_head_.parameters())
        if self.aux_binary_head_ is not None:
            params += list(self.aux_binary_head_.parameters())
        if self.aux_regression_head_ is not None:
            params += list(self.aux_regression_head_.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)

        X_t = torch.from_numpy(X_arr)
        y_primary_t = torch.from_numpy(y_primary_arr)
        y_aux_binary_t = torch.from_numpy(np.asarray(y_aux_binary, dtype=np.float32).reshape(-1, 1)) if has_aux_binary else None
        y_aux_regression_t = torch.from_numpy(np.asarray(y_aux_regression, dtype=np.float32).reshape(-1, 1)) if has_aux_regression else None

        mse = nn.MSELoss()
        bce = nn.BCEWithLogitsLoss()

        def _joint_loss(x_batch, y_primary_batch, y_aux_binary_batch, y_aux_regression_batch):
            """One forward pass through the shared trunk + all active heads; returns the weighted joint loss."""
            hidden = self.trunk_(x_batch)
            loss = mse(self.primary_head_(hidden), y_primary_batch)
            if self.aux_binary_head_ is not None:
                loss = loss + self.aux_task_weight * bce(self.aux_binary_head_(hidden), y_aux_binary_batch)
            if self.aux_regression_head_ is not None:
                loss = loss + self.aux_task_weight * mse(self.aux_regression_head_(hidden), y_aux_regression_batch)
            return loss

        self.train_losses_ = []
        n = X_t.shape[0]
        if self.batch_size is None or self.batch_size >= n:
            # Full-batch path (default): unchanged from before mini-batch support was added.
            for _ in range(self.n_epochs):
                optimizer.zero_grad()
                loss = _joint_loss(X_t, y_primary_t, y_aux_binary_t, y_aux_regression_t)
                loss.backward()
                optimizer.step()
                self.train_losses_.append(float(loss.item()))
        else:
            # Mini-batch SGD: reshuffle row order once per epoch (seeded off random_state so the
            # per-epoch batch composition is reproducible), one optimizer step per batch, epoch loss
            # = the mean of that epoch's per-batch losses (a diagnostics-only summary, not the loss
            # actually backpropagated on any single step).
            rng = np.random.default_rng(self.random_state)
            bs = int(self.batch_size)
            for _ in range(self.n_epochs):
                perm = rng.permutation(n)
                epoch_losses = []
                for start in range(0, n, bs):
                    idx_t = torch.from_numpy(perm[start : start + bs])
                    optimizer.zero_grad()
                    loss = _joint_loss(
                        X_t[idx_t], y_primary_t[idx_t],
                        y_aux_binary_t[idx_t] if y_aux_binary_t is not None else None,
                        y_aux_regression_t[idx_t] if y_aux_regression_t is not None else None,
                    )
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(float(loss.item()))
                self.train_losses_.append(float(np.mean(epoch_losses)))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the primary task's target via the trunk and primary head only."""
        import torch

        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            hidden = self.trunk_(X_t)
            pred = self.primary_head_(hidden)
        return np.asarray(pred.numpy().ravel(), dtype=np.float64)


__all__ = ["MultiTaskAuxiliaryLossRegressor"]
