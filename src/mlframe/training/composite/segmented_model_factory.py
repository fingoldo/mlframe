"""``SegmentedModelFactory``: one model per (cross of) segment keys, with incremental add/remove lifecycle.

Source: NASA Airport Configuration 1st place -- 120 separate CatBoost models trained per
(airport x lookahead-horizon) instead of one global model with airport as a categorical feature; the global
approach was discarded as too expensive and fragile to entity churn (a new/dropped airport shouldn't force
retraining every other airport's model).

Distinct from :class:`RegimeSplitEnsemble` (this session's earlier addition): that class routes by a
CONTINUOUS/COMPUTED ``regime_fn(X)`` (e.g. a rolling trend sign), requires a full refit of every segment
whenever anything changes, and has no notion of segment identity beyond the label itself. This class instead
segments by an EXPLICIT categorical key (or cross of keys, e.g. ``["airport", "horizon"]``, matching the
source's per-entity-per-horizon grid), and is built around a segment LIFECYCLE: :meth:`add_segment` /
:meth:`update_segment` / :meth:`remove_segment` touch exactly ONE segment's model, leaving every other
segment's fitted model untouched (same object, not refit) -- directly the "add/remove without retraining the
whole set" resilience to entity churn the source technique was built for.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

logger = logging.getLogger(__name__)


def _as_tuple(key: Any) -> Tuple[Any, ...]:
    """pandas groupby yields a bare scalar (not a 1-tuple) when grouping by a single column."""
    return key if isinstance(key, tuple) else (key,)


def _group_positions(X: pd.DataFrame, segment_keys: Sequence[str]) -> Dict[Tuple[Any, ...], np.ndarray]:
    """One ``groupby`` call (hash-based, O(n)) instead of a Python ``iterrows`` loop building per-row tuples
    plus one O(n) boolean-mask scan per segment (O(n_segments * n) total) -- see the module docstring's
    perf note. ``.indices`` gives positional (``iloc``-compatible) row indices per group key."""
    return {_as_tuple(key): idx for key, idx in X.groupby(list(segment_keys), sort=False).indices.items()}


class SegmentedModelFactory(BaseEstimator, RegressorMixin):
    """One model per (cross of) segment keys, with per-segment add/update/remove lifecycle management.

    Parameters
    ----------
    estimator_factory
        Zero-arg callable returning a fresh unfitted estimator, used as the default for every segment
        (overridable per-segment via ``hpo_search_fn``).
    segment_keys
        Column name(s) whose cross defines a segment (e.g. ``["airport", "horizon"]``). A single-column list
        gives one model per distinct value; multiple columns give one model per distinct COMBINATION.
    hpo_search_fn
        Optional ``callable(X_segment, y_segment) -> fitted_estimator``, called instead of
        ``estimator_factory().fit(...)`` for per-segment hyperparameter search. Falls back to the plain
        factory when None.
    min_segment_rows
        Segments with fewer than this many rows are skipped (their rows fall back to ``global_model_`` at
        predict time) -- avoids fitting a near-data-free model for a rarely-seen segment combination.

    Attributes
    ----------
    segment_models_
        ``{segment_key_tuple: fitted estimator}``.
    global_model_
        Fallback model fit on ALL rows, used for segments unseen at fit time or skipped via
        ``min_segment_rows``.
    """

    def __init__(
        self,
        estimator_factory: Callable[[], Any],
        segment_keys: Sequence[str],
        hpo_search_fn: Optional[Callable[[Any, np.ndarray], Any]] = None,
        min_segment_rows: int = 2,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.segment_keys = segment_keys
        self.hpo_search_fn = hpo_search_fn
        self.min_segment_rows = min_segment_rows

    def _drop_segment_cols(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=list(self.segment_keys))

    def _fit_one(self, X_segment: Any, y_segment: np.ndarray) -> Any:
        if self.hpo_search_fn is not None:
            return self.hpo_search_fn(X_segment, y_segment)
        model = clone(self.estimator_factory())
        model.fit(X_segment, y_segment)
        return model

    def fit(self, X: pd.DataFrame, y: Any) -> "SegmentedModelFactory":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SegmentedModelFactory: X must be a pandas DataFrame (segment_keys are column names).")
        y_arr = np.asarray(y, dtype=np.float64)
        self.segment_models_: Dict[Tuple[Any, ...], Any] = {}

        X_features = self._drop_segment_cols(X)
        for seg_key, idx in _group_positions(X, self.segment_keys).items():
            if idx.shape[0] < self.min_segment_rows:
                logger.info("SegmentedModelFactory: skipping segment %s (%d rows < min_segment_rows=%d)", seg_key, int(idx.shape[0]), self.min_segment_rows)
                continue
            self.segment_models_[seg_key] = self._fit_one(X_features.iloc[idx], y_arr[idx])

        self.global_model_ = clone(self.estimator_factory())
        self.global_model_.fit(X_features, y_arr)
        return self

    def add_segment(self, X_segment: Any, y_segment: Any, segment_key: Optional[Tuple[Any, ...]] = None) -> None:
        """Fit and register ONE new segment's model without touching any other segment's fitted model.

        ``segment_key`` is inferred from ``X_segment``'s ``segment_keys`` columns when not given explicitly
        (all rows of ``X_segment`` must share the same segment key in that case).
        """
        y_arr = np.asarray(y_segment, dtype=np.float64)
        if segment_key is None:
            keys = X_segment[list(self.segment_keys)].drop_duplicates()
            if len(keys) != 1:
                raise ValueError("add_segment: X_segment spans multiple segment keys; pass segment_key explicitly.")
            segment_key = tuple(keys.iloc[0])
        self.segment_models_[segment_key] = self._fit_one(self._drop_segment_cols(X_segment), y_arr)

    def update_segment(self, X_segment: Any, y_segment: Any, segment_key: Optional[Tuple[Any, ...]] = None) -> None:
        """Refit ONE existing (or new) segment's model in place; alias of :meth:`add_segment`."""
        self.add_segment(X_segment, y_segment, segment_key)

    def remove_segment(self, segment_key: Tuple[Any, ...]) -> None:
        """Drop a segment's model; its rows fall back to ``global_model_`` at predict time thereafter."""
        self.segment_models_.pop(segment_key, None)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n = X.shape[0]
        out = np.zeros(n, dtype=np.float64)
        X_features = self._drop_segment_cols(X)
        query_groups = _group_positions(X, self.segment_keys)

        seen_idx: List[np.ndarray] = []
        for seg_key, idx in query_groups.items():
            model = self.segment_models_.get(seg_key)
            if model is None:
                continue
            out[idx] = np.asarray(model.predict(X_features.iloc[idx]), dtype=np.float64)
            seen_idx.append(idx)

        seen_mask = np.zeros(n, dtype=bool)
        if seen_idx:
            seen_mask[np.concatenate(seen_idx)] = True
        unseen_mask = ~seen_mask
        if unseen_mask.any():
            out[unseen_mask] = np.asarray(self.global_model_.predict(X_features.iloc[np.flatnonzero(unseen_mask)]), dtype=np.float64)
        return out


__all__ = ["SegmentedModelFactory"]
