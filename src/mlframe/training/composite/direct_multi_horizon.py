"""``DirectMultiHorizonEnsemble``: one model per forecast-horizon block, trained directly (no recursion).

Source: M5 Forecasting Accuracy 4th place -- "Not to use complex and recursive features ... Recursive
features lead to error accumulation" -- instead partition the forecast horizon into blocks (e.g. weeks:
F01-F07, F08-F14, ...) and train one model PER BLOCK, each predicting its own horizon steps directly from
features available at the forecast origin (t=0) -- never from a PRIOR block's own predictions. This avoids
the classic recursive-forecasting failure mode: a one-step-ahead model applied repeatedly, feeding each
step's noisy prediction back in as the next step's lag feature, compounds error multiplicatively over the
horizon (a small per-step bias/variance snowballs by horizon 14 into a large one).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone

logger = logging.getLogger(__name__)


class DirectMultiHorizonEnsemble(BaseEstimator, RegressorMixin):
    """One independently-fit model per horizon block, each predicting directly from origin-time features.

    Parameters
    ----------
    estimator_factory
        Zero-arg callable returning a fresh unfitted estimator. Called once per horizon block; each block's
        model is trained ONLY on that block's own horizon columns of ``Y``, never on another block's
        predictions -- the defining property of the "direct" (non-recursive) multi-horizon strategy.
    horizon_blocks
        Sequence of horizon-index groups, e.g. ``[[0,1,2,3,4,5,6], [7,8,9,10,11,12,13], ...]`` for
        week-sized blocks over a 14-day horizon. Every index must appear in exactly one block.

    Attributes
    ----------
    block_models_
        One fitted clone per block, in ``horizon_blocks`` order.
    """

    def __init__(self, estimator_factory: Callable[[], Any], horizon_blocks: Sequence[Sequence[int]]) -> None:
        self.estimator_factory = estimator_factory
        self.horizon_blocks = horizon_blocks

    def _validate_blocks(self, n_horizons: int) -> None:
        seen: set[int] = set()
        for block in self.horizon_blocks:
            for h in block:
                if h in seen:
                    raise ValueError(f"horizon index {h} appears in more than one block.")
                seen.add(h)
        if seen != set(range(n_horizons)):
            raise ValueError(f"horizon_blocks must partition range(n_horizons={n_horizons}) exactly; got indices {sorted(seen)}.")

    def fit(self, X: Any, Y: np.ndarray) -> "DirectMultiHorizonEnsemble":
        """``Y``: ``(n, n_horizons)`` -- one column per forecast step, all measured from the SAME origin t=0
        (never a previous block's prediction)."""
        Y_arr = np.asarray(Y, dtype=np.float64)
        if Y_arr.ndim != 2:
            raise ValueError(f"Y must be 2-D (n, n_horizons), got shape {Y_arr.shape}")
        self._validate_blocks(Y_arr.shape[1])

        self.block_models_: list[Any] = []
        for block in self.horizon_blocks:
            model = clone(self.estimator_factory())
            block_idx = list(block)
            Y_block = Y_arr[:, block_idx]
            model.fit(X, Y_block[:, 0] if Y_block.shape[1] == 1 else Y_block)
            self.block_models_.append(model)
        self._n_horizons_ = Y_arr.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        out = np.zeros((n, self._n_horizons_), dtype=np.float64)
        for block, model in zip(self.horizon_blocks, self.block_models_):
            block_idx = list(block)
            pred = np.asarray(model.predict(X), dtype=np.float64)
            if pred.ndim == 1:
                out[:, block_idx[0]] = pred
            else:
                out[:, block_idx] = pred
        return out


__all__ = ["DirectMultiHorizonEnsemble"]
