"""``TrunkResidualMLPRegressor``: a shared trunk block skip-connected into EVERY deeper block.

Source: 3rd_jane-street-market-prediction.md -- a 49-layer MLP built from a trunk block (0) whose output is
skip-connected into every subsequent residual block 1..23. Distinct from mlframe's existing
`_ResidualLinearBlock` (`training/neural/flat.py`), which only skip-connects a block's own IMMEDIATE
predecessor (standard adjacent-layer ResNet skip): here ONE shared low-level representation is re-injected at
EVERY later block, not just handed down block-to-block -- as the network gets deep, adjacent-only skips still
let the original trunk signal dilute/degrade across many hops; re-injecting it directly at every block keeps
it available at full strength no matter how deep the tower gets.

Standalone `nn.Module` + a self-contained training loop, matching the `FixedSparseLinear`/`Tabular1DCNNRegressor`
precedent (raw-PyTorch pieces for niche architectures, not routed through mlframe's Lightning-based estimator
infra).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn


class _TrunkInjectedBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor, trunk: torch.Tensor) -> torch.Tensor:
        # trunk is re-injected (added) at THIS block's input, on top of the running residual carry from x --
        # the block sees both "what the tower has computed so far" and "the original trunk representation",
        # not just the former (which is all an adjacent-only skip would give it this many hops downstream).
        h: torch.Tensor = self.act(self.norm(self.linear(x + trunk)))
        return x + h


class _TrunkResidualMLPModule(nn.Module):
    def __init__(self, n_features: int, trunk_dim: int = 32, n_blocks: int = 6) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(n_features, trunk_dim), nn.LayerNorm(trunk_dim), nn.ReLU())
        self.blocks = nn.ModuleList([_TrunkInjectedBlock(trunk_dim) for _ in range(n_blocks)])
        self.head = nn.Linear(trunk_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trunk_out = self.trunk(x)
        h = trunk_out
        for block in self.blocks:
            h = block(h, trunk_out)
        out: torch.Tensor = self.head(h).squeeze(-1)
        return out


class TrunkResidualMLPRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible regressor: a shared trunk representation re-injected into every deeper block.

    Parameters
    ----------
    trunk_dim
        Width of the shared trunk representation (and every subsequent block).
    n_blocks
        Number of trunk-injected residual blocks stacked after the trunk.
    n_epochs
        Training epochs (full-batch Adam).
    learning_rate
        Adam learning rate.
    random_state
        Seed for model init and training.
    """

    def __init__(self, trunk_dim: int = 32, n_blocks: int = 6, n_epochs: int = 300, learning_rate: float = 0.01, random_state: int = 0) -> None:
        self.trunk_dim = trunk_dim
        self.n_blocks = n_blocks
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TrunkResidualMLPRegressor":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

        torch.manual_seed(self.random_state)
        self.model_ = _TrunkResidualMLPModule(n_features=X_arr.shape[1], trunk_dim=self.trunk_dim, n_blocks=self.n_blocks)

        X_t = torch.from_numpy(X_arr)
        y_t = torch.from_numpy(y_arr)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        self.model_.train()
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            loss = loss_fn(self.model_(X_t), y_t)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(torch.from_numpy(X_arr))
        return np.asarray(preds.numpy())


__all__ = ["TrunkResidualMLPRegressor"]
