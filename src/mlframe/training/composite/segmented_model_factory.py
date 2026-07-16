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
    shrinkage_min_rows
        Opt-in hierarchical-shrinkage fallback (default ``None`` = disabled, behavior bit-identical to the
        plain skip-or-global-fallback above). When set, segments with ``min_segment_rows <= rows <
        shrinkage_min_rows`` still get their own fitted model (there IS signal, just not enough to trust
        alone), but predictions are partially pooled with a coarser fallback model via an empirical-Bayes
        weight ``w = rows / (rows + shrinkage_k)`` -- ``w * segment_pred + (1 - w) * fallback_pred``. Tiny
        segments (few rows) lean on the fallback; well-populated ones (many rows) lean on their own fit.
        Segments below ``min_segment_rows`` are still skipped entirely and route straight to the fallback,
        same as the non-shrinkage path. Must be ``>= min_segment_rows`` when set.
    shrinkage_parent_keys
        The coarser segment-key subset the fallback model is fit on (e.g. ``["airport"]`` when
        ``segment_keys=["airport", "horizon"]`` -- pools across horizons within an airport). Must be a
        proper subset of ``segment_keys``. Defaults to ``None``, meaning the fallback is ``global_model_``
        (pooling across ALL rows) -- only meaningful when ``shrinkage_min_rows`` is also set.
    shrinkage_k
        Smoothing constant in the pooling weight above; larger ``shrinkage_k`` pools more aggressively
        toward the fallback for a given row count. Only used when ``shrinkage_min_rows`` is set.

    Attributes
    ----------
    segment_models_
        ``{segment_key_tuple: fitted estimator}``.
    global_model_
        Fallback model fit on ALL rows, used for segments unseen at fit time, skipped via
        ``min_segment_rows``, or (when shrinkage is enabled with no ``shrinkage_parent_keys``) blended in
        for undersized segments.
    parent_models_
        ``{parent_key_tuple: fitted estimator}``, one per distinct value of ``shrinkage_parent_keys``.
        Empty unless both ``shrinkage_min_rows`` and ``shrinkage_parent_keys`` are set.
    shrinkage_weights_
        ``{segment_key_tuple: w}`` for segments undergoing partial pooling (rows in
        ``[min_segment_rows, shrinkage_min_rows)``). Empty unless ``shrinkage_min_rows`` is set.
    """

    def __init__(
        self,
        estimator_factory: Callable[[], Any],
        segment_keys: Sequence[str],
        hpo_search_fn: Optional[Callable[[Any, np.ndarray], Any]] = None,
        min_segment_rows: int = 2,
        shrinkage_min_rows: Optional[int] = None,
        shrinkage_parent_keys: Optional[Sequence[str]] = None,
        shrinkage_k: float = 10.0,
    ) -> None:
        self.estimator_factory = estimator_factory
        self.segment_keys = segment_keys
        self.hpo_search_fn = hpo_search_fn
        self.min_segment_rows = min_segment_rows
        self.shrinkage_min_rows = shrinkage_min_rows
        self.shrinkage_parent_keys = shrinkage_parent_keys
        self.shrinkage_k = shrinkage_k

    def _drop_segment_cols(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return ``X`` with the segment-key columns removed, leaving only model features."""
        return X.drop(columns=list(self.segment_keys))

    def _fit_one(self, X_segment: Any, y_segment: np.ndarray) -> Any:
        """Fit (or HPO-search) one segment's model on its own rows."""
        if self.hpo_search_fn is not None:
            return self.hpo_search_fn(X_segment, y_segment)
        model = clone(self.estimator_factory())
        model.fit(X_segment, y_segment)
        return model

    def _parent_key_of(self, seg_key: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Project a full segment key down to its coarser ``shrinkage_parent_keys`` sub-key."""
        assert self.shrinkage_parent_keys is not None
        idx_map = [list(self.segment_keys).index(k) for k in self.shrinkage_parent_keys]
        return tuple(seg_key[i] for i in idx_map)

    def fit(self, X: pd.DataFrame, y: Any) -> "SegmentedModelFactory":
        """Fit one model per segment (skipping too-small segments), optional parent/global pooling models for shrinkage."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SegmentedModelFactory: X must be a pandas DataFrame (segment_keys are column names).")
        if self.shrinkage_min_rows is not None and self.shrinkage_min_rows < self.min_segment_rows:
            raise ValueError("SegmentedModelFactory: shrinkage_min_rows must be >= min_segment_rows.")
        if self.shrinkage_parent_keys is not None and not set(self.shrinkage_parent_keys) < set(self.segment_keys):
            raise ValueError("SegmentedModelFactory: shrinkage_parent_keys must be a proper subset of segment_keys.")

        y_arr = np.asarray(y, dtype=np.float64)
        self.segment_models_: Dict[Tuple[Any, ...], Any] = {}
        self.parent_models_: Dict[Tuple[Any, ...], Any] = {}
        self.shrinkage_weights_: Dict[Tuple[Any, ...], float] = {}

        X_features = self._drop_segment_cols(X)
        for seg_key, idx in _group_positions(X, self.segment_keys).items():
            n_rows = int(idx.shape[0])
            if n_rows < self.min_segment_rows:
                logger.info("SegmentedModelFactory: skipping segment %s (%d rows < min_segment_rows=%d)", seg_key, n_rows, self.min_segment_rows)
                continue
            self.segment_models_[seg_key] = self._fit_one(X_features.iloc[idx], y_arr[idx])
            if self.shrinkage_min_rows is not None and n_rows < self.shrinkage_min_rows:
                self.shrinkage_weights_[seg_key] = n_rows / (n_rows + self.shrinkage_k)

        if self.shrinkage_min_rows is not None and self.shrinkage_parent_keys is not None:
            for parent_key, idx in _group_positions(X, self.shrinkage_parent_keys).items():
                self.parent_models_[parent_key] = self._fit_one(X_features.iloc[idx], y_arr[idx])

        self.global_model_ = clone(self.estimator_factory())
        self.global_model_.fit(X_features, y_arr)
        return self

    def _fallback_predict(self, seg_key: Tuple[Any, ...], X_seg_features: Any) -> np.ndarray:
        """Prediction from the coarser pooling model: the matching parent segment if configured and
        present, else the global model -- used both for fully-skipped segments and as the pooling partner
        for shrinkage-blended segments."""
        if self.shrinkage_parent_keys is not None:
            parent_model = self.parent_models_.get(self._parent_key_of(seg_key))
            if parent_model is not None:
                return np.asarray(parent_model.predict(X_seg_features), dtype=np.float64)
        return np.asarray(self.global_model_.predict(X_seg_features), dtype=np.float64)

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
        """Predict each row via its segment's model (with shrinkage/fallback blending) or the global model."""
        n = X.shape[0]
        out = np.zeros(n, dtype=np.float64)
        X_features = self._drop_segment_cols(X)
        query_groups = _group_positions(X, self.segment_keys)

        seen_idx: List[np.ndarray] = []
        for seg_key, idx in query_groups.items():
            model = self.segment_models_.get(seg_key)
            if model is None:
                continue
            X_seg = X_features.iloc[idx]
            seg_pred = np.asarray(model.predict(X_seg), dtype=np.float64)
            w = self.shrinkage_weights_.get(seg_key)
            if w is None:
                out[idx] = seg_pred
            else:
                # Partial pooling: undersized-but-fitted segment blended with its coarser fallback model,
                # rather than either trusting a data-starved per-segment fit outright or discarding it
                # entirely in favor of the fallback -- see class docstring.
                fallback_pred = self._fallback_predict(seg_key, X_seg)
                out[idx] = w * seg_pred + (1.0 - w) * fallback_pred
            seen_idx.append(idx)

        seen_mask = np.zeros(n, dtype=bool)
        if seen_idx:
            seen_mask[np.concatenate(seen_idx)] = True
        unseen_mask = ~seen_mask
        if unseen_mask.any():
            unseen_pos = np.flatnonzero(unseen_mask)
            if self.shrinkage_min_rows is not None and self.shrinkage_parent_keys is not None:
                # Fully-skipped segments (below min_segment_rows) still prefer the parent model over the
                # coarsest global fallback when shrinkage is configured -- same idea as the blended case,
                # just at weight 0 (no per-segment fit exists to blend in).
                for seg_key, idx in query_groups.items():
                    if seg_key in self.segment_models_:
                        continue
                    rows_here = np.intersect1d(idx, unseen_pos, assume_unique=True)
                    if rows_here.size:
                        out[rows_here] = self._fallback_predict(seg_key, X_features.iloc[rows_here])
            else:
                out[unseen_pos] = np.asarray(self.global_model_.predict(X_features.iloc[unseen_pos]), dtype=np.float64)
        return out


__all__ = ["SegmentedModelFactory"]
