"""``GroupedBlockStacker``: one OOF submodel per redundant/correlated feature block, stacked by a meta-model.

Source: Santander Value Prediction 2nd place -- "fit one model for each group of features [113 groups of 40],
but only for rows that have at least one nonzero element [in that group]... stacked all 113 out-of-fold
predictions and ran a second level fit." A production analog: sensor arrays / repeated measurement blocks
where each block is individually informative but many rows have a whole block missing/degenerate (a sensor
offline, a measurement type not collected for that entity), so a single global model wastes rows where any
one feature is invalid, while per-block submodels each use every row where THEIR OWN block is valid.

Feature-group source: mlframe's DCD (Dynamic Cluster Discovery, `feature_selection.filters
._dynamic_cluster_discovery`) already detects correlated/redundant feature groups during MRMR's greedy
selection loop, exposed post-fit via ``MRMR.dcd_["cluster_anchors_names"]`` (anchor column name -> list of
member column names). This class does NOT reach into DCD's internals directly (that machinery is tightly
coupled to MRMR's live greedy-selection state, not something to import into an unrelated meta-estimator) --
instead it takes a plain ``{group_name: [column_names]}`` mapping as input, which a caller can populate FROM
``dcd_summary(...)["cluster_anchors_names"]`` after an MRMR fit, or from any domain-known grouping (e.g. a
known sensor-array layout). This keeps the two systems decoupled while still directly serving the idea's ask.

Reuses :func:`mlframe.training.composite.ensemble.feature_stacking.composite_oof_predictions` as the
leakage-safe OOF engine per group (restricted to that group's valid rows), rather than reimplementing K-fold
OOF plumbing.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from .ensemble.feature_stacking import composite_oof_predictions

logger = logging.getLogger(__name__)


def _default_valid_row_mask(X_group: np.ndarray) -> np.ndarray:
    """A row is valid for a group if it has at least one nonzero, finite element in that group's columns --
    the exact criterion from the source technique ("only for rows that have at least one nonzero element")."""
    finite = np.isfinite(X_group)
    nonzero = finite & (X_group != 0)
    return np.asarray(nonzero.any(axis=1))


def _select_group(X: Any, columns: Sequence[Any]) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        return np.asarray(X[list(columns)].to_numpy(dtype=np.float64))
    try:
        import polars as pl
        if isinstance(X, pl.DataFrame):
            return X.select(list(columns)).to_numpy().astype(np.float64)
    except ImportError:
        pass
    return np.asarray(X, dtype=np.float64)[:, [int(c) for c in columns]]


class GroupedBlockStacker(BaseEstimator, RegressorMixin):
    """Per-feature-group OOF submodels (restricted to each group's valid rows), stacked by a meta-model.

    Parameters
    ----------
    feature_groups
        ``{group_name: [column names or indices]}`` -- one submodel is fit per group, using only that
        group's columns. Typically sourced from DCD's ``cluster_anchors_names`` after an MRMR fit (see
        module docstring), but any grouping works.
    submodel_factory
        Zero-arg callable returning a fresh unfitted estimator, used for EVERY group (clone-per-group via a
        fresh factory call, not ``sklearn.clone`` of a single shared prototype, so each group's model is
        fully independent).
    meta_estimator
        sklearn-compatible estimator prototype trained on the stacked per-group OOF prediction columns.
        Cloned at fit time.
    valid_row_predicate
        ``callable(X_group_array) -> (n,) bool array``. Defaults to :func:`_default_valid_row_mask` (at
        least one nonzero finite value in the group), matching the source technique exactly.
    invalid_row_fill
        How an invalid row's meta-feature is filled for a group it doesn't qualify for: ``"group_mean"``
        (default, that group's own OOF-prediction mean over its valid rows) or a float constant.
    n_splits, random_state
        Passed through to each group's :func:`composite_oof_predictions` call.

    Attributes
    ----------
    submodels_
        ``{group_name: fitted estimator}``, refit on ALL of that group's valid rows (not OOF) for predict().
    group_valid_rates_
        ``{group_name: fraction of training rows valid for that group}`` (diagnostic).
    group_fill_values_
        ``{group_name: fallback meta-feature value for invalid rows}``.
    """

    def __init__(
        self,
        feature_groups: Mapping[str, Sequence[Any]],
        submodel_factory: Callable[[], Any],
        meta_estimator: Any,
        valid_row_predicate: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        invalid_row_fill: Any = "group_mean",
        n_splits: int = 5,
        random_state: int = 42,
    ) -> None:
        self.feature_groups = feature_groups
        self.submodel_factory = submodel_factory
        self.meta_estimator = meta_estimator
        self.valid_row_predicate = valid_row_predicate
        self.invalid_row_fill = invalid_row_fill
        self.n_splits = n_splits
        self.random_state = random_state

    def _mask_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        return self.valid_row_predicate if self.valid_row_predicate is not None else _default_valid_row_mask

    def fit(self, X: Any, y: Any, sample_weight: Optional[np.ndarray] = None) -> "GroupedBlockStacker":
        if not self.feature_groups:
            raise ValueError("feature_groups must be non-empty.")
        y_arr = np.asarray(y, dtype=np.float64)
        n = y_arr.shape[0]
        mask_fn = self._mask_fn()

        self.submodels_: dict[str, Any] = {}
        self.group_valid_rates_: dict[str, float] = {}
        self.group_fill_values_: dict[str, float] = {}
        meta_cols: dict[str, np.ndarray] = {}

        for group_name, columns in self.feature_groups.items():
            X_group = _select_group(X, columns)
            valid_mask = np.asarray(mask_fn(X_group), dtype=bool)
            self.group_valid_rates_[group_name] = float(valid_mask.mean()) if n else 0.0

            oof_col = np.full(n, np.nan, dtype=np.float64)
            if valid_mask.sum() >= max(2, self.n_splits):
                X_group_valid = pd.DataFrame(X_group[valid_mask], columns=[str(c) for c in columns])
                oof_valid = composite_oof_predictions(self.submodel_factory, X_group_valid, y_arr[valid_mask], n_splits=self.n_splits, random_state=self.random_state)
                oof_col[valid_mask] = oof_valid

                full_model = self.submodel_factory()
                full_model.fit(X_group_valid, y_arr[valid_mask])
                self.submodels_[group_name] = full_model
            else:
                logger.warning("GroupedBlockStacker: group '%s' has only %d valid rows (<max(2,n_splits)); skipped.", group_name, int(valid_mask.sum()))
                self.submodels_[group_name] = None

            finite_oof = oof_col[np.isfinite(oof_col)]
            fill_value = float(finite_oof.mean()) if finite_oof.size else (float(self.invalid_row_fill) if isinstance(self.invalid_row_fill, (int, float)) else 0.0)
            self.group_fill_values_[group_name] = fill_value
            oof_col = np.where(np.isfinite(oof_col), oof_col, fill_value)
            meta_cols[f"meta_{group_name}"] = oof_col

        meta_X = pd.DataFrame(meta_cols)
        self.meta_model_ = clone(self.meta_estimator)
        fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
        self.meta_model_.fit(meta_X, y_arr, **fit_kwargs)
        return self

    def predict(self, X: Any) -> np.ndarray:
        meta_cols: dict[str, np.ndarray] = {}
        for group_name, columns in self.feature_groups.items():
            X_group = _select_group(X, columns)
            model = self.submodels_.get(group_name)
            fill_value = self.group_fill_values_[group_name]
            if model is None:
                meta_cols[f"meta_{group_name}"] = np.full(X_group.shape[0], fill_value, dtype=np.float64)
                continue
            mask_fn = self._mask_fn()
            valid_mask = np.asarray(mask_fn(X_group), dtype=bool)
            pred = np.full(X_group.shape[0], fill_value, dtype=np.float64)
            if valid_mask.any():
                X_group_df = pd.DataFrame(X_group[valid_mask], columns=[str(c) for c in columns])
                pred[valid_mask] = np.asarray(model.predict(X_group_df), dtype=np.float64)
            meta_cols[f"meta_{group_name}"] = pred

        meta_X = pd.DataFrame(meta_cols)
        return np.asarray(self.meta_model_.predict(meta_X))


__all__ = ["GroupedBlockStacker"]
