"""``Tabular1DCNNRegressor``: 1D-CNN over a correlation-ordered feature sequence.

Source: 2nd_mechanisms-of-action-moa-prediction.md -- reshapes/orders continuous tabular features (gene-
expression + cell-viability) along an axis and applies a 1D convolution to capture LOCAL higher-order
interactions a plain MLP treats as unstructured (an MLP's first layer sees every feature pair with an
independent weight; a 1D-CNN's kernel only sees a local window, forcing it to learn compact, reusable local
patterns -- efficient when nearby features in the ordering are genuinely related, wasteful/noisy when the
ordering is arbitrary). Feature ordering matters for this architecture to have any local structure to exploit
at all: :func:`correlation_order_features` greedily chains features so consecutive positions in the ordering
are highly correlated (a 1D analogue of correlation-based hierarchical-clustering reordering), giving the
convolution meaningful adjacency to work with instead of an arbitrary column order.

Standalone `nn.Module` + a self-contained training loop, NOT wired into mlframe's Lightning-based NN estimator
infra (`training/neural/base`, `flat.py`) -- matching the precedent set by `FixedSparseLinear` (raw-PyTorch
pieces for niche architectures where the full Lightning fit/predict loop isn't the right scope).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from torch import nn


def correlation_order_features(X: np.ndarray) -> List[int]:
    """Greedily chain feature indices so adjacent positions in the returned order are highly correlated.

    Starts from the feature most correlated (in absolute Pearson correlation, averaged over all others) with
    the rest, then repeatedly appends whichever remaining feature is most correlated with the LAST placed
    feature -- a cheap nearest-neighbor-chain heuristic giving a 1D-CNN's local kernel window genuine
    adjacency structure to exploit, without the cost of full hierarchical clustering.

    Parameters
    ----------
    X
        ``(n, d)`` feature matrix.

    Returns
    -------
    list of int
        A permutation of ``range(d)``.
    """
    d = X.shape[1]
    if d <= 2:
        return list(range(d))

    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    abs_corr = np.abs(corr)
    np.fill_diagonal(abs_corr, 0.0)

    remaining = set(range(d))
    current = int(np.argmax(abs_corr.sum(axis=1)))
    order = [current]
    remaining.discard(current)

    while remaining:
        remaining_list = list(remaining)
        scores = abs_corr[current, remaining_list]
        next_feature = remaining_list[int(np.argmax(scores))]
        order.append(next_feature)
        remaining.discard(next_feature)
        current = next_feature

    return order


class _Tabular1DCNNBackbone(nn.Module):
    """Shared conv stack: same architecture regression and classification heads sit on top of.

    Factored out so both ``Tabular1DCNNRegressor`` and ``Tabular1DCNNClassifier`` reuse one
    implementation of "1D-CNN over the correlation-ordered sequence" instead of duplicating it --
    only the head (and loss) differs between the two tasks.
    """

    def __init__(self, n_channels: int = 16, kernel_size: int = 5) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(n_channels, n_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, 1, n_features)
        out: torch.Tensor = self.conv(x)
        return out


class _Tabular1DCNNModule(nn.Module):
    def __init__(self, n_features: int, n_channels: int = 16, kernel_size: int = 5) -> None:
        super().__init__()
        self.backbone = _Tabular1DCNNBackbone(n_channels=n_channels, kernel_size=kernel_size)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.head(self.backbone(x)).squeeze(-1)
        return out


class _Tabular1DCNNClassifierModule(nn.Module):
    def __init__(self, n_features: int, n_classes: int, n_channels: int = 16, kernel_size: int = 5) -> None:
        super().__init__()
        self.backbone = _Tabular1DCNNBackbone(n_channels=n_channels, kernel_size=kernel_size)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(n_channels, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.head(self.backbone(x))  # (batch, n_classes) logits
        return out


class Tabular1DCNNRegressor(BaseEstimator, RegressorMixin):
    """sklearn-compatible regressor: reorders features by correlation, then fits a small 1D-CNN over them.

    Parameters
    ----------
    n_channels
        Conv1d channel width.
    kernel_size
        Conv1d kernel size (local window over the correlation-ordered feature sequence).
    n_epochs
        Training epochs (full-batch Adam).
    learning_rate
        Adam learning rate.
    random_state
        Seed for model init and training.
    """

    def __init__(self, n_channels: int = 16, kernel_size: int = 5, n_epochs: int = 300, learning_rate: float = 0.01, random_state: int = 0) -> None:
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Tabular1DCNNRegressor":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1)

        self.feature_order_ = correlation_order_features(X_arr)
        X_ordered = X_arr[:, self.feature_order_]

        torch.manual_seed(self.random_state)
        self.model_ = _Tabular1DCNNModule(n_features=X_ordered.shape[1], n_channels=self.n_channels, kernel_size=self.kernel_size)

        X_t = torch.from_numpy(X_ordered)
        y_t = torch.from_numpy(y_arr)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        self.model_.train()
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            preds = self.model_(X_t)
            loss = loss_fn(preds, y_t)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)[:, self.feature_order_]
        self.model_.eval()
        with torch.no_grad():
            preds = self.model_(torch.from_numpy(X_arr))
        return np.asarray(preds.numpy())


class Tabular1DCNNClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-compatible classifier counterpart of ``Tabular1DCNNRegressor``.

    Same correlation-ordered 1D-CNN backbone (:func:`correlation_order_features` +
    :class:`_Tabular1DCNNBackbone`), swapping the regression head/MSE loss for a linear
    ``n_classes``-wide head trained with cross-entropy -- binary and multiclass both go through the
    same code path (binary is just ``n_classes == 2``).

    Parameters
    ----------
    n_channels
        Conv1d channel width.
    kernel_size
        Conv1d kernel size (local window over the correlation-ordered feature sequence).
    n_epochs
        Training epochs (full-batch Adam).
    learning_rate
        Adam learning rate.
    random_state
        Seed for model init and training.
    """

    def __init__(self, n_channels: int = 16, kernel_size: int = 5, n_epochs: int = 300, learning_rate: float = 0.01, random_state: int = 0) -> None:
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Tabular1DCNNClassifier":
        X_arr = np.asarray(X, dtype=np.float32)

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(np.asarray(y).reshape(-1))
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError(f"Tabular1DCNNClassifier requires at least 2 classes, got {n_classes}")

        self.feature_order_ = correlation_order_features(X_arr)
        X_ordered = X_arr[:, self.feature_order_]

        torch.manual_seed(self.random_state)
        self.model_ = _Tabular1DCNNClassifierModule(
            n_features=X_ordered.shape[1], n_classes=n_classes, n_channels=self.n_channels, kernel_size=self.kernel_size
        )

        X_t = torch.from_numpy(X_ordered)
        y_t = torch.from_numpy(y_enc.astype(np.int64))
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        self.model_.train()
        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            logits = self.model_(X_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            optimizer.step()

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)[:, self.feature_order_]
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(torch.from_numpy(X_arr))
            proba = torch.softmax(logits, dim=-1)
        return np.asarray(proba.numpy())

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=-1)
        labels: np.ndarray = self.label_encoder_.inverse_transform(idx)
        return labels


__all__ = ["Tabular1DCNNRegressor", "Tabular1DCNNClassifier", "correlation_order_features"]
