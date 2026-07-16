"""Row-level feature-importance passthrough for ``compute_row_level_then_average_predictions``.

The parent function aggregates a row-level model's predictions to entity level, but by design throws away
the fitted model(s) doing the work -- callers can see the aggregate OOF score moved, but not WHICH child-row
features actually drove it. This module fits/extracts importances from the SAME row-level model class the
caller already supplies via ``model_factory``, so diagnosing "which feature is responsible" doesn't require
a separate, possibly-inconsistent side analysis.

Mode A (OOF): the parent's ``composite_oof_predictions`` call already trains one model per ``GroupKFold``
fold internally but doesn't expose them, so this module reruns the identical fold split (``GroupKFold`` on
the same ``entity_ids``/``n_splits`` -- deterministic, no ``random_state`` since ``GroupKFold`` has none) to
collect and mean-average per-fold importances. This duplicates the row-level fit cost when opted in; the
main prediction path is untouched.

Mode B (external query): the parent already fits exactly one model on the full ``X_rows`` for prediction --
importances are extracted from that SAME fitted model at zero extra fit cost.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


def _feature_names(X_rows: Any) -> list[str]:
    """Return column names for ``X_rows``, falling back to positional ``f0, f1, ...`` for plain arrays."""
    if isinstance(X_rows, pd.DataFrame):
        return [str(c) for c in X_rows.columns]
    if isinstance(X_rows, pl.DataFrame):
        return [str(c) for c in X_rows.columns]
    n_cols = np.asarray(X_rows).shape[1]
    return [f"f{i}" for i in range(n_cols)]


def _subset_rows(X_rows: Any, idx: np.ndarray) -> Any:
    """Select rows of ``X_rows`` (pandas, polars, or array-like) by integer positions ``idx``."""
    if isinstance(X_rows, pl.DataFrame):
        mask = np.isin(np.arange(X_rows.height), idx, assume_unique=True)
        return X_rows.filter(pl.Series(mask))
    if isinstance(X_rows, pd.DataFrame):
        return X_rows.iloc[idx].reset_index(drop=True)
    return np.asarray(X_rows)[idx]


def extract_model_importance(model: Any, feature_names: Sequence[str]) -> np.ndarray:
    """Pull a ``(n_features,)`` importance vector out of a fitted row-level model.

    Tries ``feature_importances_`` (tree ensembles) first, then ``abs(coef_)`` (linear models) as the
    magnitude-of-effect proxy. Raises ``AttributeError`` for model classes exposing neither -- callers decide
    whether to skip that fold or fail loudly.
    """
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=np.float64).reshape(-1)
    elif hasattr(model, "coef_"):
        imp = np.abs(np.asarray(model.coef_, dtype=np.float64)).reshape(-1)
    else:
        raise AttributeError(
            f"extract_model_importance: model {type(model).__name__} exposes neither 'feature_importances_' "
            "nor 'coef_'; cannot compute row-level feature importance for it."
        )
    if imp.shape[0] != len(feature_names):
        raise ValueError(f"extract_model_importance: importance length {imp.shape[0]} != {len(feature_names)} feature names.")
    return imp


def compute_row_level_feature_importance_oof(
    model_factory: Any,
    X_rows: Any,
    y_arr: np.ndarray,
    entity_arr: np.ndarray,
    n_splits: int,
) -> pl.DataFrame:
    """Mode A: refit the identical ``GroupKFold`` split used internally by ``composite_oof_predictions`` on
    ``entity_arr``, average each fold's row-level model importance, and return one row per feature sorted by
    descending mean importance."""
    from sklearn.model_selection import GroupKFold  # lazy, mirrors composite_oof_predictions

    feature_names = _feature_names(X_rows)
    n = y_arr.size
    indices = np.arange(n)
    kf = GroupKFold(n_splits=int(n_splits))
    sums = np.zeros(len(feature_names), dtype=np.float64)
    n_folds_used = 0
    for train_idx, _val_idx in kf.split(indices, y_arr, entity_arr):
        X_train = _subset_rows(X_rows, train_idx)
        model = model_factory()
        model.fit(X_train, y_arr[train_idx])
        try:
            sums += extract_model_importance(model, feature_names)
            n_folds_used += 1
        except AttributeError as exc:
            logger.warning("compute_row_level_feature_importance_oof: skipping a fold, %s", exc)
    if n_folds_used == 0:
        raise AttributeError(
            "compute_row_level_feature_importance_oof: the row-level model exposes neither 'feature_importances_' "
            "nor 'coef_' in any fold; feature-importance passthrough is unsupported for this model_factory."
        )
    mean_importance = sums / n_folds_used
    return pl.DataFrame({"feature": feature_names, "importance": mean_importance}).sort("importance", descending=True)


def compute_row_level_feature_importance_single_model(model: Any, X_rows: Any) -> pl.DataFrame:
    """Mode B: extract importance from the single model the caller already fit on the full ``X_rows``."""
    feature_names = _feature_names(X_rows)
    importance = extract_model_importance(model, feature_names)
    return pl.DataFrame({"feature": feature_names, "importance": importance}).sort("importance", descending=True)


__all__ = ["extract_model_importance", "compute_row_level_feature_importance_oof", "compute_row_level_feature_importance_single_model"]
