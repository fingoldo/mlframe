"""``FieldGroupedMLPRegressor``: per-field sub-MLP encoders merged before a shared head.

Source: 1st_kkbox-music-recommendation-challenge.md -- "Field-aware: inputs divided into user/song/context
groups, high-level features extracted before concatenation." Distinct from mlframe's existing flat NN pipeline
(`training/neural/flat.py`), which treats every input column as one homogeneous block fed into a single MLP:
here each named FIELD (a group of related columns, e.g. all user-side columns vs all song-side columns) gets
its own small sub-MLP encoder first, and only the small ENCODED representations are concatenated before the
shared head -- structurally prevents the model from ever forming a spurious cross-field interaction between
two unrelated raw columns (it can only interact each field's own SUMMARY with other fields' summaries), which
reduces effective parameter count and overfitting risk versus flat concatenation of every raw column.

Standalone raw-PyTorch piece (matching the `FixedSparseLinear`/`Tabular1DCNNRegressor`/`TrunkResidualMLPRegressor`
precedent), not routed through mlframe's Lightning-based NN estimator infra.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from torch import nn


class _FieldGroupedMLPModule(nn.Module):
    def __init__(self, field_groups: Dict[str, Sequence[int]], field_hidden: int = 8, head_hidden: int = 16) -> None:
        super().__init__()
        self.field_indices = {name: list(idx) for name, idx in field_groups.items()}
        self.field_encoders = nn.ModuleDict(
            {name: nn.Sequential(nn.Linear(len(idx), field_hidden), nn.ReLU(), nn.Linear(field_hidden, field_hidden), nn.ReLU()) for name, idx in field_groups.items()}
        )
        total_encoded = field_hidden * len(field_groups)
        self.head = nn.Sequential(nn.Linear(total_encoded, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = []
        for name, idx in self.field_indices.items():
            field_x = x[:, idx]
            encoded.append(self.field_encoders[name](field_x))
        merged = torch.cat(encoded, dim=1)
        out: torch.Tensor = self.head(merged).squeeze(-1)
        return out


class FieldGroupedMLPRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible regressor: per-field sub-MLP encoders, merged before a shared head.

    Parameters
    ----------
    field_groups
        ``{field_name: column_indices}`` -- partitions the input feature columns into named fields, each
        routed through its own sub-MLP before merging. Every column index must appear in exactly one field.
    field_hidden
        Hidden width of each field's own sub-MLP encoder.
    head_hidden
        Hidden width of the shared head applied to the concatenated field encodings.
    n_epochs
        Training epochs (full-batch Adam).
    learning_rate
        Adam learning rate.
    random_state
        Seed for model init and training.
    """

    def __init__(self, field_groups: Dict[str, Sequence[int]], field_hidden: int = 8, head_hidden: int = 16, n_epochs: int = 300, learning_rate: float = 0.01, random_state: int = 0) -> None:
        self.field_groups = field_groups
        self.field_hidden = field_hidden
        self.head_hidden = head_hidden
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FieldGroupedMLPRegressor":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

        torch.manual_seed(self.random_state)
        self.model_ = _FieldGroupedMLPModule(self.field_groups, field_hidden=self.field_hidden, head_hidden=self.head_hidden)

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


__all__ = ["FieldGroupedMLPRegressor"]
