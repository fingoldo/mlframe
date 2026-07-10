"""``CountWeightedBlendEnsemble``: blend an entity-specific model with a metadata/global model by observation count.

Source: KKBox Music Recommendation Challenge 1st place -- "Allowing an offset for users between final
representation and predicted results from metadata... model learns to rely more on user-embedding if data
sufficient, more on metadata if not." A soft cold-start fallback: when an entity has few training
observations, its own entity-specific model (an embedding, a per-entity submodel) is unreliable/overfit, so
the prediction should lean on a metadata/global model instead; when an entity is well-observed, lean on the
entity-specific model.

Generalizes the additive-count smoothing formula already used for scalar target-encoding
(``count / (count + k)``) to blend two FULL model outputs rather than just an encoded categorical value.

Distinct from :class:`SimilarityBlendEnsemble` (this session's earlier addition): that class weights by
embedding-distance similarity to the training set (a geometric "how close is this row to what we've seen"
signal). This class weights by raw per-ENTITY observation COUNT (a frequency/data-sufficiency signal) --
different evidence source, same underlying "continuous blend, not a hard route" pattern.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

logger = logging.getLogger(__name__)


class CountWeightedBlendEnsemble(BaseEstimator, RegressorMixin):
    """Blend ``entity_estimator`` (fit on all columns, e.g. an entity-embedding-aware model) and
    ``global_estimator`` (fit on metadata columns only) with weight ``count / (count + k)``.

    Parameters
    ----------
    entity_estimator
        sklearn-compatible estimator prototype fit on the FULL feature frame (including ``entity_col``, or
        columns derived from it, e.g. an entity embedding) -- the "reliable when well-observed" specialist.
    global_estimator
        sklearn-compatible estimator prototype fit on ``metadata_cols`` only (excluding ``entity_col`` and
        anything entity-specific) -- the "reliable fallback for sparse entities" generalist.
    entity_col
        Column identifying the entity (used to count training observations per entity; NOT necessarily fed
        to either estimator directly).
    metadata_cols
        Columns ``global_estimator`` is fit/predicted on (default: every column except ``entity_col``).
    k
        Smoothing constant in ``weight = count / (count + k)`` -- larger ``k`` requires more observations
        before trusting the entity-specific model; smaller ``k`` trusts it sooner.

    Attributes
    ----------
    entity_model_, global_model_
        The fitted clones.
    entity_counts_
        ``{entity_id: training row count}``, used to compute blend weights at predict time (an entity
        unseen at fit time gets count 0, i.e. weight 0 -- fully deferred to the global model).
    """

    def __init__(
        self,
        entity_estimator: Any,
        global_estimator: Any,
        entity_col: str,
        metadata_cols: Optional[Sequence[str]] = None,
        k: float = 10.0,
    ) -> None:
        self.entity_estimator = entity_estimator
        self.global_estimator = global_estimator
        self.entity_col = entity_col
        self.metadata_cols = metadata_cols
        self.k = k

    def _metadata_cols(self, X: pd.DataFrame) -> list[str]:
        return list(self.metadata_cols) if self.metadata_cols is not None else [c for c in X.columns if c != self.entity_col]

    def fit(self, X: pd.DataFrame, y: Any, sample_weight: Optional[np.ndarray] = None) -> "CountWeightedBlendEnsemble":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CountWeightedBlendEnsemble: X must be a pandas DataFrame.")
        y_arr = np.asarray(y, dtype=np.float64)
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}

        self.entity_model_ = clone(self.entity_estimator)
        self.entity_model_.fit(X, y_arr, **fit_kwargs)

        meta_cols = self._metadata_cols(X)
        self.global_model_ = clone(self.global_estimator)
        self.global_model_.fit(X[meta_cols], y_arr, **fit_kwargs)

        self.entity_counts_ = X[self.entity_col].value_counts().to_dict()
        return self

    def _blend_weight(self, X: pd.DataFrame) -> np.ndarray:
        counts = X[self.entity_col].map(self.entity_counts_).fillna(0).to_numpy(dtype=np.float64)
        return np.asarray(counts / (counts + self.k))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        w = self._blend_weight(X)
        pred_entity = np.asarray(self.entity_model_.predict(X), dtype=np.float64)
        pred_global = np.asarray(self.global_model_.predict(X[self._metadata_cols(X)]), dtype=np.float64)
        return w * pred_entity + (1.0 - w) * pred_global


__all__ = ["CountWeightedBlendEnsemble"]
