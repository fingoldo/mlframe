"""Direct (non-recursive) horizon-bucket forecasting: independent per-bucket models, no lag-of-own-prediction.

Recursive multi-step forecasting (predict step 1, feed it back as a lag feature to predict step 2, ...)
accumulates error: a bad early-step prediction poisons every later step's input. An M5-forecasting 4th place
team deliberately avoided this, training INDEPENDENT direct-forecast models per (entity, horizon-bucket) using
only features known at forecast time (no self-referential lag), sidestepping the accumulation problem
entirely at the cost of training more models. This wrapper formalizes that recipe as the default-recommended
pattern over recursive forecasting.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class DirectHorizonBucketForecaster:
    """Train one independent direct-forecast model per (entity-group, horizon-bucket).

    Parameters
    ----------
    horizon_buckets
        List of ``(start, end)`` inclusive day-offset ranges partitioning the forecast horizon (e.g.
        ``[(1, 7), (8, 14), (15, 21), (22, 28)]``).
    model_factory
        Zero-arg factory returning a fresh sklearn-compatible regressor (``.fit(X, y)`` / ``.predict(X)``).
    group_col
        Optional column name in ``X`` to additionally split models by (e.g. store id); ``None`` trains one
        model per horizon bucket only, pooling all entities.

    Notes
    -----
    Every model is trained ONLY on features available at forecast time (whatever the caller passes in ``X``)
    -- this class does not construct or forbid any specific feature; the "no recursive lag-of-own-prediction"
    discipline is the CALLER's responsibility (don't pass a lag of this forecaster's own prior output as a
    feature), consistent with the source recipe.
    """

    def __init__(
        self,
        horizon_buckets: Sequence[Tuple[int, int]],
        model_factory: Callable[[], Any],
        group_col: Optional[str] = None,
    ):
        if not horizon_buckets:
            raise ValueError("DirectHorizonBucketForecaster: horizon_buckets must be non-empty")
        self.horizon_buckets = list(horizon_buckets)
        self.model_factory = model_factory
        self.group_col = group_col
        self.models_: Dict[Tuple[Any, Tuple[int, int]], Any] = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray, horizon_day: np.ndarray) -> "DirectHorizonBucketForecaster":
        """Fit one model per (group, bucket) using only rows whose ``horizon_day`` falls in that bucket."""
        y = np.asarray(y)
        horizon_day = np.asarray(horizon_day)
        groups = X[self.group_col].to_numpy() if self.group_col is not None else np.zeros(len(X), dtype=np.int8)
        unique_groups: List[Any] = [None] if self.group_col is None else list(pd.unique(groups))

        feature_cols = [c for c in X.columns if c != self.group_col]
        self.models_ = {}
        for group in unique_groups:
            group_mask = np.ones(len(X), dtype=bool) if group is None else (groups == group)
            for bucket in self.horizon_buckets:
                bucket_mask = group_mask & (horizon_day >= bucket[0]) & (horizon_day <= bucket[1])
                if not bucket_mask.any():
                    continue
                model = self.model_factory()
                model.fit(X.loc[bucket_mask, feature_cols], y[bucket_mask])
                self.models_[(group, bucket)] = model
        return self

    def predict(self, X: pd.DataFrame, horizon_day: np.ndarray) -> np.ndarray:
        """Predict each row using the model fit for its ``(group, bucket)``; ``NaN`` where no model was fit."""
        horizon_day = np.asarray(horizon_day)
        groups = X[self.group_col].to_numpy() if self.group_col is not None else np.zeros(len(X), dtype=np.int8)
        feature_cols = [c for c in X.columns if c != self.group_col]

        preds = np.full(len(X), np.nan, dtype=np.float64)
        unique_groups: List[Any] = [None] if self.group_col is None else list(pd.unique(groups))
        for group in unique_groups:
            group_mask = np.ones(len(X), dtype=bool) if group is None else (groups == group)
            for bucket in self.horizon_buckets:
                bucket_mask = group_mask & (horizon_day >= bucket[0]) & (horizon_day <= bucket[1])
                if not bucket_mask.any():
                    continue
                model = self.models_.get((group, bucket))
                if model is None:
                    continue
                preds[bucket_mask] = model.predict(X.loc[bucket_mask, feature_cols])
        return preds


__all__ = ["DirectHorizonBucketForecaster"]
