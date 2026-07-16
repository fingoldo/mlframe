"""Model-family-aware outlier handling policy: skip capping for tree models, engineer a flag instead.

Tree-based models (LightGBM/XGBoost/CatBoost/RandomForest) split on thresholds and are largely resilient to
extreme outlier VALUES -- capping/removing them ahead of training typically loses real signal for no
robustness benefit. Linear/distance-based models (LinearRegression, SVM, KNN, neural nets) have no such
resilience and genuinely need outlier handling. Rather than applying one blanket outlier policy regardless of
the downstream estimator, this dispatches: tree models get an outlier-SCORE feature only (no value
modification); everything else gets the existing capping/rejection treatment.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd

from mlframe.preprocessing.outliers import count_num_outofranges

# Class-name substrings (case-sensitive, matched against type(model).__name__ and its MRO) identifying
# tree-based estimators that are resilient to raw outlier values -- checked by substring so this covers
# sklearn/lightgbm/xgboost/catboost variants (e.g. "LGBMClassifier", "XGBRegressor", "CatBoostClassifier",
# "RandomForestRegressor", "ExtraTreesClassifier", "GradientBoostingRegressor") without an exhaustive enum.
_TREE_MODEL_NAME_MARKERS = (
    "LGBM", "XGB", "CatBoost", "RandomForest", "ExtraTrees", "GradientBoosting",
    "DecisionTree", "HistGradientBoosting", "AdaBoost",
)


def _unwrap_final_estimator(model: Any, _depth: int = 0) -> Any:
    """Recursively resolve the real estimator wrapped by common sklearn meta-estimator patterns.

    Handles ``Pipeline`` (last step), and the ``estimator``/``base_estimator`` attribute used by
    ``CalibratedClassifierCV``, ``BaggingRegressor``, ``AdaBoostClassifier``, ``MultiOutputRegressor``, etc.
    Bails out after a few hops (meta-estimators don't nest deeply in practice) to avoid infinite recursion on
    a pathological ``self``-referencing attribute.
    """
    if _depth >= 5:
        return model
    steps = getattr(model, "steps", None)
    if steps:
        return _unwrap_final_estimator(steps[-1][1], _depth + 1)
    for attr in ("estimator", "base_estimator"):
        inner = getattr(model, attr, None)
        if inner is not None and inner is not model:
            return _unwrap_final_estimator(inner, _depth + 1)
    return model


def is_tree_based_model(model: Any, unwrap_pipeline: bool = False) -> bool:
    """Detect a tree-based estimator by scanning its class MRO's names for known tree-family markers.

    Parameters
    ----------
    model
        The estimator (instance or class) to classify.
    unwrap_pipeline
        Opt-in. When ``True``, also resolve common sklearn meta-estimator wrappers (``Pipeline``,
        ``CalibratedClassifierCV``, ``BaggingRegressor``, etc. via their ``steps``/``estimator``/
        ``base_estimator`` attributes) and classify the real underlying estimator, so a caller passing a
        fitted ``Pipeline(steps=[..., ("clf", LGBMClassifier())])`` gets correctly routed to the tree-family
        policy instead of silently falling through to capping. Default ``False`` preserves the prior exact
        behavior (direct MRO check on ``model`` only).
    """

    def _mro_matches(obj: Any) -> bool:
        """Return whether any class in ``obj``'s MRO name-matches a known tree-family marker."""
        mro_names = [cls.__name__ for cls in type(obj).__mro__]
        return any(marker in name for name in mro_names for marker in _TREE_MODEL_NAME_MARKERS)

    if _mro_matches(model):
        return True
    if unwrap_pipeline:
        unwrapped = _unwrap_final_estimator(model)
        if unwrapped is not model:
            return _mro_matches(unwrapped)
    return False


def apply_outlier_policy(
    X: pd.DataFrame,
    model: Any,
    columns: Optional[list] = None,
    cap_quantiles: Tuple[float, float] = (0.01, 0.99),
    unwrap_pipeline: bool = False,
) -> pd.DataFrame:
    """Apply a model-family-aware outlier policy: cap for non-tree models, flag-only for tree models.

    Parameters
    ----------
    X
        Feature frame.
    model
        The downstream estimator instance (or class) whose family determines the policy; detected via
        :func:`is_tree_based_model`.
    columns
        Numeric columns to consider; defaults to every numeric column in ``X``.
    cap_quantiles
        ``(low, high)`` quantile bounds passed to capping for non-tree models.
    unwrap_pipeline
        Opt-in, forwarded to :func:`is_tree_based_model`. When ``True``, a ``model`` wrapped in a sklearn
        ``Pipeline``/``CalibratedClassifierCV``/etc. is classified by its real underlying estimator instead
        of the wrapper class -- avoids silently applying the wrong (capping) policy to a tree model just
        because it was passed inside a pipeline. Default ``False`` preserves the prior exact behavior.

    Returns
    -------
    pd.DataFrame
        For a tree-based model: ``X`` UNCHANGED plus one new ``outlier_score`` column (a naive per-row
        outlier score across ``columns``, values NOT modified). For a non-tree model: ``X`` with
        ``columns`` capped to ``cap_quantiles`` (no new column).
    """
    if columns is None:
        columns = [c for c in X.select_dtypes(include=[np.number]).columns]

    if is_tree_based_model(model, unwrap_pipeline=unwrap_pipeline):
        out = X.copy()
        col_arr = X[columns].to_numpy(dtype=np.float64)
        mins = np.nanquantile(col_arr, cap_quantiles[0], axis=0)
        maxs = np.nanquantile(col_arr, cap_quantiles[1], axis=0)
        out_of_range_counts = count_num_outofranges(col_arr, mins, maxs)
        out["outlier_score"] = out_of_range_counts.astype(np.float64) / max(len(columns), 1)
        return out

    # one batched DataFrame.quantile() call across all columns instead of two per-column Series.quantile()
    # calls each -- the quantile partition itself dominates either way, but batching still measured ~18%
    # faster at n=1M/50 cols (2.65s vs 3.22s) by avoiding repeated per-column dispatch overhead.
    out = X.copy()
    bounds = X[columns].quantile([cap_quantiles[0], cap_quantiles[1]])
    out[columns] = out[columns].clip(lower=bounds.loc[cap_quantiles[0]], upper=bounds.loc[cap_quantiles[1]], axis=1)
    return out


__all__ = ["is_tree_based_model", "apply_outlier_policy"]
