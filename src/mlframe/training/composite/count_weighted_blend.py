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
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

DEFAULT_K_GRID: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0)


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
        before trusting the entity-specific model; smaller ``k`` trusts it sooner. Ignored (but still
        stored, for ``get_params``/``sklearn`` clone compatibility) when ``auto_k`` is True.
    auto_k
        Opt-in (default False, i.e. bit-identical to the original fixed-``k`` behavior when unused). When
        True, ``k`` is *learned* at fit time: a small grid of candidate values (``k_grid``) is evaluated by
        ``k_cv``-fold cross-validation on the training rows -- for each fold, ``entity_estimator`` and
        ``global_estimator`` are cloned and refit on the fold's train split (with per-fold entity counts,
        so no leakage of a row's own count into its own held-out weight), predictions on the held-out fold
        are blended at every candidate ``k``, and the ``k`` with the lowest pooled held-out MSE across all
        folds is kept as ``k_``. The final ``entity_model_``/``global_model_`` used at predict time are
        still fit once on the FULL training data (as in the fixed-``k`` path) -- CV here only selects
        ``k_``, it does not change what the deployed submodels are.
    k_grid
        Candidate values searched when ``auto_k`` is True (default: ``DEFAULT_K_GRID``, 1 to 200).
    k_cv
        Number of CV folds used to score each candidate ``k`` when ``auto_k`` is True (default 3).
    random_state
        Seed for the CV fold shuffle when ``auto_k`` is True.

    Attributes
    ----------
    entity_model_, global_model_
        The fitted clones (always fit once on the full training data).
    entity_counts_
        ``{entity_id: training row count}``, used to compute blend weights at predict time (an entity
        unseen at fit time gets count 0, i.e. weight 0 -- fully deferred to the global model).
    k_
        The smoothing constant actually used at predict time: equal to ``k`` unless ``auto_k`` is True, in
        which case it is the CV-selected value.
    k_cv_scores_
        ``{k: pooled held-out MSE}`` for every candidate in ``k_grid``, set only when ``auto_k`` is True --
        useful for diagnosing how sensitive the blend is to ``k``.
    """

    def __init__(
        self,
        entity_estimator: Any,
        global_estimator: Any,
        entity_col: str,
        metadata_cols: Optional[Sequence[str]] = None,
        k: float = 10.0,
        auto_k: bool = False,
        k_grid: Optional[Sequence[float]] = None,
        k_cv: int = 3,
        random_state: Optional[int] = None,
    ) -> None:
        self.entity_estimator = entity_estimator
        self.global_estimator = global_estimator
        self.entity_col = entity_col
        self.metadata_cols = metadata_cols
        self.k = k
        self.auto_k = auto_k
        self.k_grid = k_grid
        self.k_cv = k_cv
        self.random_state = random_state

    def _metadata_cols(self, X: pd.DataFrame) -> list[str]:
        """Return the configured metadata columns, or all columns except the entity column by default."""
        return list(self.metadata_cols) if self.metadata_cols is not None else [c for c in X.columns if c != self.entity_col]

    def _select_k_via_cv(self, X: pd.DataFrame, y_arr: np.ndarray, meta_cols: list[str], fit_kwargs: dict[str, Any]) -> float:
        """Cross-validate the blend smoothing constant ``k`` over a grid and return the value with lowest mean squared error."""
        grid = list(self.k_grid) if self.k_grid is not None else list(DEFAULT_K_GRID)
        kf = KFold(n_splits=self.k_cv, shuffle=True, random_state=self.random_state)
        n = len(X)
        sq_err_by_k: dict[float, list[np.ndarray]] = {k: [] for k in grid}

        for train_idx, val_idx in kf.split(np.arange(n)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
            fold_fit_kwargs = dict(fit_kwargs)
            if "sample_weight" in fold_fit_kwargs:
                fold_fit_kwargs["sample_weight"] = np.asarray(fold_fit_kwargs["sample_weight"])[train_idx]

            fold_entity_model = clone(self.entity_estimator)
            fold_entity_model.fit(X_tr, y_tr, **fold_fit_kwargs)
            fold_global_model = clone(self.global_estimator)
            fold_global_model.fit(X_tr[meta_cols], y_tr, **fold_fit_kwargs)

            fold_counts = X_tr[self.entity_col].value_counts().to_dict()
            counts_val = X_val[self.entity_col].map(fold_counts).fillna(0).to_numpy(dtype=np.float64)
            pred_entity = np.asarray(fold_entity_model.predict(X_val), dtype=np.float64)
            pred_global = np.asarray(fold_global_model.predict(X_val[meta_cols]), dtype=np.float64)

            for k in grid:
                w = counts_val / (counts_val + k)
                pred = w * pred_entity + (1.0 - w) * pred_global
                sq_err_by_k[k].append((pred - y_val) ** 2)

        self.k_cv_scores_ = {k: float(np.mean(np.concatenate(errs))) for k, errs in sq_err_by_k.items()}
        return min(self.k_cv_scores_, key=lambda k: self.k_cv_scores_[k])

    def fit(self, X: pd.DataFrame, y: Any, sample_weight: Optional[np.ndarray] = None) -> "CountWeightedBlendEnsemble":
        """Fit the entity-level and global models (and, if ``auto_k``, the blend constant ``k``) on ``X``, ``y``."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CountWeightedBlendEnsemble: X must be a pandas DataFrame.")
        y_arr = np.asarray(y, dtype=np.float64)
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        meta_cols = self._metadata_cols(X)

        if self.auto_k:
            self.k_ = self._select_k_via_cv(X, y_arr, meta_cols, fit_kwargs)
        else:
            self.k_ = self.k

        self.entity_model_ = clone(self.entity_estimator)
        self.entity_model_.fit(X, y_arr, **fit_kwargs)

        self.global_model_ = clone(self.global_estimator)
        self.global_model_.fit(X[meta_cols], y_arr, **fit_kwargs)

        self.entity_counts_ = X[self.entity_col].value_counts().to_dict()
        return self

    def _blend_weight(self, X: pd.DataFrame) -> np.ndarray:
        """Return the per-row entity-vs-global blend weight from fitted entity counts and smoothing constant ``k_``."""
        counts = X[self.entity_col].map(self.entity_counts_).fillna(0).to_numpy(dtype=np.float64)
        return np.asarray(counts / (counts + self.k_))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Blend the entity-level and global model predictions weighted by per-row entity observation count."""
        w = self._blend_weight(X)
        pred_entity = np.asarray(self.entity_model_.predict(X), dtype=np.float64)
        pred_global = np.asarray(self.global_model_.predict(X[self._metadata_cols(X)]), dtype=np.float64)
        return w * pred_entity + (1.0 - w) * pred_global


__all__ = ["CountWeightedBlendEnsemble"]
