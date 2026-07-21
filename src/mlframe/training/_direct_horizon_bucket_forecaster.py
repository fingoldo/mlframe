"""Direct (non-recursive) horizon-bucket forecasting: independent per-bucket models, no lag-of-own-prediction.

Recursive multi-step forecasting (predict step 1, feed it back as a lag feature to predict step 2, ...)
accumulates error: a bad early-step prediction poisons every later step's input. An M5-forecasting 4th place
team deliberately avoided this, training INDEPENDENT direct-forecast models per (entity, horizon-bucket) using
only features known at forecast time (no self-referential lag), sidestepping the accumulation problem
entirely at the cost of training more models. This wrapper formalizes that recipe as the default-recommended
pattern over recursive forecasting.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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

    def predict(self, X: pd.DataFrame, horizon_day: np.ndarray, edge_blend_width: int = 0) -> np.ndarray:
        """Predict each row using the model fit for its ``(group, bucket)``; ``NaN`` where no model was fit.

        Parameters
        ----------
        edge_blend_width
            Opt-in boundary smoothing. ``0`` (default) reproduces the original hard-boundary behavior
            bit-identically. When ``>0``, rows within this many horizon-day steps of a shared boundary
            between two *consecutive* buckets get a linearly-weighted blend of both neighboring buckets'
            predictions (each still computed purely from that row's own features -- no lagged/recursive
            input crosses buckets, only the two independently-computed scalar predictions are averaged),
            reducing the discontinuity a hard bucket cutover otherwise creates right at the edge.
        """
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

            if edge_blend_width > 0:
                self._blend_bucket_edges(X, preds, horizon_day, group, group_mask, feature_cols, edge_blend_width)

        n_nan = int(np.isnan(preds).sum())
        if n_nan:
            logger.warning(
                "DirectHorizonBucketForecaster.predict: %d/%d row(s) got NaN -- no fitted model covers their "
                "(group, bucket) (out-of-range horizon_day, an unseen group at predict time, or NaN in "
                "group_col).",
                n_nan, len(preds),
            )
        return preds

    def _blend_bucket_edges(
        self,
        X: pd.DataFrame,
        preds: np.ndarray,
        horizon_day: np.ndarray,
        group: Any,
        group_mask: np.ndarray,
        feature_cols: List[str],
        edge_blend_width: int,
    ) -> None:
        """Blend predictions near each internal boundary between two consecutive, adjacent buckets.

        Mutates ``preds`` in place. Only touches rows whose original hard-boundary prediction came from
        one of the two buckets sharing that boundary; each side's prediction is still that model's own
        unmodified output on the row's own features -- only the final scalar combination changes.
        """
        sorted_buckets = sorted(self.horizon_buckets, key=lambda b: b[0])
        for left, right in zip(sorted_buckets, sorted_buckets[1:]):
            if right[0] != left[1] + 1:
                continue  # not adjacent -- no shared edge to smooth
            left_model = self.models_.get((group, left))
            right_model = self.models_.get((group, right))
            if left_model is None or right_model is None:
                continue
            boundary = left[1]  # last day of left bucket == boundary; right starts at boundary + 1

            # symmetric linear ramp: own-bucket weight is exactly 0.5 right at the boundary on both sides
            # (matching continuity across the edge) and rises to ~1 at the far end of the blend zone.
            left_zone = group_mask & (horizon_day > boundary - edge_blend_width) & (horizon_day <= boundary)
            if left_zone.any():
                other_pred = right_model.predict(X.loc[left_zone, feature_cols])
                dist = boundary - horizon_day[left_zone]  # 0 at boundary .. edge_blend_width-1 at far end
                own_weight = 0.5 + 0.5 * dist / edge_blend_width
                preds[left_zone] = own_weight * preds[left_zone] + (1 - own_weight) * other_pred

            right_zone = group_mask & (horizon_day >= boundary + 1) & (horizon_day < boundary + 1 + edge_blend_width)
            if right_zone.any():
                other_pred = left_model.predict(X.loc[right_zone, feature_cols])
                dist = horizon_day[right_zone] - (boundary + 1)  # 0 at boundary .. edge_blend_width-1 at far end
                own_weight = 0.5 + 0.5 * dist / edge_blend_width
                preds[right_zone] = own_weight * preds[right_zone] + (1 - own_weight) * other_pred


__all__ = ["DirectHorizonBucketForecaster"]
