"""Shared eval-mode ``predict`` for the ``training/neural`` torch MLP regressor wrappers
(``TrunkResidualMLPRegressor``, ``FieldGroupedMLPRegressor``): both wrap a fitted
``self.model_`` and were independently duplicating the identical eval/no_grad/forward
body, consolidated here so a fix can't silently drift out of sync across copies. Bind as
``predict = mlp_predict`` in the class body.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


def mlp_predict(self: Any, X: np.ndarray) -> np.ndarray:
    """Predict with the fitted torch MLP module (``self.model_``) in eval mode."""
    X_arr = np.asarray(X, dtype=np.float32)
    self.model_.eval()
    with torch.no_grad():
        preds = self.model_(torch.from_numpy(X_arr))
    return np.asarray(preds.numpy())
