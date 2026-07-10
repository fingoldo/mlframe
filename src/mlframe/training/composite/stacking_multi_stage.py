"""``MultiStageMetaFeatureStacker``: auxiliary-target OOF meta-features feeding a primary-target model.

Source: Mechanisms of Action (MoA) prediction 1st place -- a 3-stage NN where stage 1 predicts auxiliary
("non-scored") targets, whose predictions become meta-features for stage 2 (predicting the real "scored"
targets), whose predictions in turn become meta-features for stage 3. Generalizes to any setting with a
richer/adjacent label space than the one being scored (multi-label bio/drug assay data, business telemetry
with more internal labels than the one that's actually evaluated).

Reuses :func:`mlframe.training.composite.ensemble.feature_stacking.composite_oof_predictions` as the leakage
-safe OOF engine (generic over any ``wrapper_factory``, already handles polars/pandas, groups, sample_weight)
rather than reimplementing K-fold OOF plumbing -- this class only adds the multi-auxiliary-target
orchestration and the QuantileTransform-then-concat step that primitive doesn't do on its own.

Repeating the stage-1-into-stage-2 pattern for a 3rd stage (as MoA's writeup does) is achieved by CHAINING
two instances -- fit a first stacker, then feed its stage-2 OOF predictions back in as one more auxiliary
target for a second stacker -- rather than baking recursion into this class, keeping each stage's leakage
discipline independently auditable.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from .ensemble.feature_stacking import composite_oof_predictions

logger = logging.getLogger(__name__)


class MultiStageMetaFeatureStacker(BaseEstimator):
    """Stage 1 (auxiliary targets) -> OOF meta-features -> Stage 2 (primary target).

    Parameters
    ----------
    stage1_estimator_factories
        ``{aux_target_name: zero-arg callable returning a fresh unfitted estimator}`` -- one factory per
        auxiliary target. Each is OOF-fit against its own auxiliary target via
        :func:`composite_oof_predictions`, producing a leakage-safe per-row prediction column.
    stage2_estimator
        sklearn-compatible estimator prototype (``fit(X, y)`` / ``predict(X)``), cloned at fit time. Trained
        on the ORIGINAL features plus every stage-1 meta-feature column.
    n_splits, random_state
        Passed through to each auxiliary target's :func:`composite_oof_predictions` call.
    quantile_transform
        If True (default, matching the MoA source technique), each stage-1 OOF prediction column is
        rank-mapped to a standard-normal distribution (``sklearn.preprocessing.QuantileTransformer``) before
        being handed to stage 2 -- smooths out miscalibrated/skewed stage-1 output scales so stage 2 sees a
        comparable distribution across auxiliary targets regardless of how each was trained.
    meta_feature_prefix
        Column-name prefix for the injected meta-features (``{prefix}_{aux_target_name}``).

    Attributes
    ----------
    stage1_models_
        ``{aux_target_name: fitted estimator}`` -- refit on the FULL training set (not OOF) once stage 1's
        OOF predictions have been consumed for stage-2 training, so :meth:`predict` on new rows has a
        real fitted model to call (an OOF-only model has no single "fitted on everything" version).
    stage2_model_
        The fitted stage-2 estimator.
    quantile_transformers_
        ``{aux_target_name: fitted QuantileTransformer}`` when ``quantile_transform=True``, else ``{}``.
    """

    def __init__(
        self,
        stage1_estimator_factories: Mapping[str, Callable[[], Any]],
        stage2_estimator: Any,
        n_splits: int = 5,
        random_state: int = 42,
        quantile_transform: bool = True,
        meta_feature_prefix: str = "meta",
    ) -> None:
        self.stage1_estimator_factories = stage1_estimator_factories
        self.stage2_estimator = stage2_estimator
        self.n_splits = n_splits
        self.random_state = random_state
        self.quantile_transform = quantile_transform
        self.meta_feature_prefix = meta_feature_prefix

    def _meta_column_name(self, aux_name: str) -> str:
        return f"{self.meta_feature_prefix}_{aux_name}"

    @staticmethod
    def _ensure_frame(X: Any) -> Any:
        """Wrap a plain ndarray into a pandas DataFrame (generic ``f{j}`` column names); pandas/polars
        frames pass through unchanged."""
        if isinstance(X, pd.DataFrame):
            return X
        try:
            import polars as pl
            if isinstance(X, pl.DataFrame):
                return X
        except ImportError:
            pass
        X_arr = np.asarray(X, dtype=np.float64)
        return pd.DataFrame(X_arr, columns=[f"f{j}" for j in range(X_arr.shape[1])])

    def fit(
        self,
        X: Any,
        y_primary: np.ndarray,
        y_auxiliary: Mapping[str, np.ndarray],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "MultiStageMetaFeatureStacker":
        if not self.stage1_estimator_factories:
            raise ValueError("stage1_estimator_factories must be non-empty.")
        missing = set(self.stage1_estimator_factories) - set(y_auxiliary)
        if missing:
            raise ValueError(f"y_auxiliary is missing targets required by stage1_estimator_factories: {sorted(missing)}")

        # composite_oof_predictions requires a pandas/polars frame (row-index-based fold slicing); a plain
        # ndarray is wrapped once here so callers can pass either.
        X = self._ensure_frame(X)

        self.stage1_models_: dict[str, Any] = {}
        self.quantile_transformers_: dict[str, Any] = {}
        meta_cols: dict[str, np.ndarray] = {}

        for aux_name, factory in self.stage1_estimator_factories.items():
            y_aux = np.asarray(y_auxiliary[aux_name])
            oof_pred = composite_oof_predictions(factory, X, y_aux, n_splits=self.n_splits, random_state=self.random_state)
            if self.quantile_transform:
                from sklearn.preprocessing import QuantileTransformer
                n_quantiles = min(1000, max(10, oof_pred.shape[0]))
                qt = QuantileTransformer(output_distribution="normal", n_quantiles=n_quantiles, random_state=self.random_state)
                # NaN-safe: composite_oof_predictions NaN-fills folds that failed to train; impute with the
                # OOF mean before fitting the transformer (QuantileTransformer itself rejects NaN input).
                finite_mask = np.isfinite(oof_pred)
                fill_value = float(oof_pred[finite_mask].mean()) if finite_mask.any() else 0.0
                oof_pred_filled = np.where(finite_mask, oof_pred, fill_value)
                meta_cols[self._meta_column_name(aux_name)] = qt.fit_transform(oof_pred_filled.reshape(-1, 1)).ravel()
                self.quantile_transformers_[aux_name] = qt
            else:
                meta_cols[self._meta_column_name(aux_name)] = oof_pred

            # Refit on the FULL training set so predict() on new rows has a real model to call.
            full_model = factory()
            full_model.fit(X, y_aux)
            self.stage1_models_[aux_name] = full_model

        X_meta = self._concat_meta(X, meta_cols)
        self.stage2_model_ = clone(self.stage2_estimator)
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        self.stage2_model_.fit(X_meta, y_primary, **fit_kwargs)
        return self

    def _build_meta_features_at_predict(self, X: Any) -> dict[str, np.ndarray]:
        meta_cols: dict[str, np.ndarray] = {}
        for aux_name, model in self.stage1_models_.items():
            pred = np.asarray(model.predict(X), dtype=np.float64).reshape(-1)
            if self.quantile_transform:
                qt = self.quantile_transformers_[aux_name]
                pred = qt.transform(pred.reshape(-1, 1)).ravel()
            meta_cols[self._meta_column_name(aux_name)] = pred
        return meta_cols

    @staticmethod
    def _concat_meta(X: Any, meta_cols: dict[str, np.ndarray]) -> Any:
        try:
            import polars as pl
            if isinstance(X, pl.DataFrame):
                return X.with_columns(**{name: pl.Series(vals) for name, vals in meta_cols.items()})
        except ImportError:
            pass
        if isinstance(X, pd.DataFrame):
            out = X.copy()
            for name, vals in meta_cols.items():
                out[name] = vals
            return out
        # Plain ndarray: append meta columns as extra numeric columns.
        X_arr = np.asarray(X, dtype=np.float64)
        meta_arr = np.column_stack([meta_cols[name] for name in meta_cols])
        return np.concatenate([X_arr, meta_arr], axis=1)

    def predict(self, X: Any) -> np.ndarray:
        X = self._ensure_frame(X)
        meta_cols = self._build_meta_features_at_predict(X)
        X_meta = self._concat_meta(X, meta_cols)
        return np.asarray(self.stage2_model_.predict(X_meta))


__all__ = ["MultiStageMetaFeatureStacker"]
