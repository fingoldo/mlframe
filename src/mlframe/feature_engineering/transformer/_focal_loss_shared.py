"""Shared LightGBM focal-loss custom objective (Lin et al. 2017) for the ``multi_aux_ensemble.py`` /
``focal_lgb.py`` transformer family: independently duplicated across those modules, consolidated here
so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


def make_focal_objective(gamma: float = 2.0) -> Callable[[np.ndarray, Any], "tuple[np.ndarray, np.ndarray]"]:
    """Build a LightGBM-conforming ``(preds, train_data) -> (grad, hess)`` closure for binary focal loss."""

    def objective(preds, train_data):
        """LightGBM custom-objective signature: raw logits + Dataset -> (grad, hess) for the focal loss above."""
        labels = train_data.get_label()
        preds_clipped = np.clip(preds, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-preds_clipped))
        pt = labels * p + (1.0 - labels) * (1.0 - p)
        focal_term = (1.0 - pt) ** gamma
        grad = (
            focal_term
            * (labels * (gamma * pt * np.log(np.maximum(pt, 1e-9)) - (1.0 - pt)) + (1.0 - labels) * ((1.0 - pt) - gamma * pt * np.log(np.maximum(pt, 1e-9))))
            * (p - labels)
            / np.maximum(pt, 1e-9)
        )
        hess = focal_term * p * (1.0 - p) * (1.0 + gamma * (1.0 - pt))
        grad = np.clip(grad, -10.0, 10.0)
        hess = np.maximum(hess, 1e-6)
        return grad, hess

    return objective
