"""``predicted_group_aggregate_feature``: leakage-safe cross-sectional "macro factor" as a per-row feature.

Source: Optiver Realized Volatility Prediction 3rd place -- trained a separate model to predict the average
volatility across ALL stock_ids for each time_id, then used that PREDICTED macro value as an input feature
to the main per-stock model. Generalizes to any panel with a cross-sectional grouping (time_id, date,
batch): fleet-wide average sensor reading feeding per-device models, market-wide average return feeding
per-asset models, store-wide average feeding per-SKU models.

Leakage discipline: using the group's REALIZED (true) aggregate as a feature for its own member rows would
leak (a member row's own value contributes to the group aggregate it then gets to see as a feature -- worse,
using it at inference time is impossible since the aggregate isn't known until every member is known, which
circularly includes the row being predicted). Instead, an AUXILIARY model is trained to PREDICT the group
aggregate from group-level features (not the realized value), OOF-safe so no group's own rows leak into its
own predicted aggregate, and that PREDICTION -- not the ground truth -- is broadcast back as a feature.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl

from .ensemble.feature_stacking import composite_oof_predictions

logger = logging.getLogger(__name__)

_VALID_AGGS = ("mean", "median", "std", "min", "max")


def _default_group_feature_agg(X_row: pd.DataFrame) -> pd.Series:
    """Mean of every numeric column, the natural default cross-sectional group-level feature summary."""
    return X_row.select_dtypes(include=[np.number]).mean()


def _oof_multi_target_predictions(
    wrapper_factory: Callable[[], Any],
    X_group: pd.DataFrame,
    Y_group: np.ndarray,
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    """K-fold OOF predictions for a MULTI-output target matrix ``Y_group`` (n_groups, n_stats).

    One ``wrapper.fit(X_train, Y_train)`` call per fold trains on ALL requested statistics at once --
    same fold split and design matrix ``composite_oof_predictions`` would use per-statistic, but the
    estimator amortizes its per-fold factorization/tree-build cost across every target column instead
    of repeating it once per statistic. Requires an estimator whose ``fit``/``predict`` support native
    multi-output regression (e.g. ``LinearRegression``, tree-based regressors) -- callers using an
    estimator without multi-output support should keep calling this function once per statistic instead.
    """
    from sklearn.model_selection import KFold  # lazy, mirrors composite_oof_predictions

    n, n_targets = Y_group.shape
    out = np.full((n, n_targets), np.nan, dtype=np.float64)
    kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
    indices = np.arange(n)
    for train_idx, val_idx in kf.split(indices):
        X_train = X_group.iloc[train_idx].reset_index(drop=True)
        X_val = X_group.iloc[val_idx].reset_index(drop=True)
        try:
            w = wrapper_factory()
            w.fit(X_train, Y_group[train_idx])
            fold_preds = np.asarray(w.predict(X_val), dtype=np.float64)
            if fold_preds.ndim == 1:
                fold_preds = fold_preds.reshape(-1, 1)
            out[val_idx] = fold_preds
        except Exception as fold_err:
            logger.warning(
                "[_oof_multi_target_predictions] fold failed (val rows %d-%d): %s. NaN-filled.",
                int(val_idx.min()), int(val_idx.max()), fold_err,
            )
    return out


def predicted_group_aggregate_feature(
    X_row: Any,
    y_row: np.ndarray,
    group_ids: np.ndarray,
    macro_estimator_factory: Callable[[], Any],
    group_feature_agg_fn: Callable[[pd.DataFrame], pd.Series] = _default_group_feature_agg,
    agg: str = "mean",
    n_splits: int = 5,
    random_state: int = 42,
    column_name: str = "predicted_group_aggregate",
    aggs: Optional[Sequence[str]] = None,
) -> pl.DataFrame:
    """Train an OOF-safe auxiliary model on group-level aggregated features/target, broadcast its
    predictions back to a per-original-row feature column.

    Parameters
    ----------
    X_row
        Per-row feature frame (pandas), ``n_rows`` rows.
    y_row
        ``(n_rows,)`` target.
    group_ids
        ``(n_rows,)`` cross-sectional group id per row (e.g. ``time_id``).
    macro_estimator_factory
        Zero-arg callable returning a fresh unfitted estimator, fit on group-level rows (one row per unique
        group) to predict the group's target aggregate from the group's aggregated features.
    group_feature_agg_fn
        ``callable(X_row_for_one_group) -> pd.Series`` -- how to summarize a group's member rows into one
        feature row. Defaults to the mean of every numeric column.
    agg
        How to aggregate ``y_row`` within a group to build the auxiliary target: ``"mean"`` or ``"median"``.
    n_splits, random_state
        Passed to the group-level OOF CV (over unique groups, not over the original rows).
    column_name
        Name of the returned broadcast feature column (or column-name prefix when ``aggs`` is given).
    aggs
        Opt-in: a list of aggregation statistics (any of ``"mean"``, ``"median"``, ``"std"``, ``"min"``,
        ``"max"``) to compute in ONE call. When given, ``agg`` is ignored and the single expensive OOF fit
        step is shared across every requested statistic -- one ``macro_estimator_factory().fit(X_train,
        Y_train)`` call per fold trains a multi-output target matrix (one column per statistic) instead of
        repeating a full per-statistic K-fold OOF fit. Requires ``macro_estimator_factory`` to build an
        estimator whose ``fit``/``predict`` support native multi-output regression (e.g.
        ``sklearn.linear_model.LinearRegression``, tree-based regressors) -- an estimator without
        multi-output support will raise inside ``fit``/``predict``; fall back to calling this function once
        per statistic in that case. Returns one column per statistic, named ``f"{column_name}__{stat}"``.
        Default ``None`` preserves the original single-``agg`` behavior exactly (bit-identical).

    Returns
    -------
    pl.DataFrame
        ``aggs is None``: single column ``column_name``. ``aggs`` given: one ``f"{column_name}__{stat}"``
        column per requested statistic. Always one row per ORIGINAL row (same length/order as ``X_row``).
    """
    if agg not in ("mean", "median"):
        raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")
    if not isinstance(X_row, pd.DataFrame):
        raise TypeError("predicted_group_aggregate_feature: X_row must be a pandas DataFrame.")
    if aggs is not None:
        bad = [a for a in aggs if a not in _VALID_AGGS]
        if bad:
            raise ValueError(f"aggs entries must be one of {_VALID_AGGS}, got invalid: {bad}")
        if len(aggs) == 0:
            raise ValueError("aggs must be a non-empty sequence when given.")

    group_arr = np.asarray(group_ids)
    y_arr = np.asarray(y_row, dtype=np.float64)

    # A per-group boolean-mask filter (df[df["_group_id"] == g]) rescans the FULL n-row frame for every one
    # of n_groups groups -- O(n_groups * n) total. pandas' hash-based groupby is O(n) -- ~75x faster measured
    # at n_groups=5,000/n=100,000 (11.3s -> ~0.15s order of magnitude; see bench_group_aggregate_macro.py).
    unique_groups = pd.unique(group_arr)
    if group_feature_agg_fn is _default_group_feature_agg:
        # Fully vectorized fast path: one groupby-mean call instead of one Python callback per group
        # (measured ~15x faster than groupby.apply(_default_group_feature_agg) at n_groups=5,000/n=100,000).
        X_group = X_row.select_dtypes(include=[np.number]).groupby(group_arr, sort=False).mean().reindex(unique_groups)
    else:
        grouped = X_row.groupby(group_arr, sort=False)
        X_group = grouped.apply(group_feature_agg_fn, include_groups=False).reindex(unique_groups)

    if aggs is None:
        y_grouped = pd.Series(y_arr).groupby(group_arr).agg(agg).reindex(unique_groups)
        y_group = y_grouped.to_numpy(dtype=np.float64)

        if len(unique_groups) >= max(2, n_splits):
            oof_group_pred = composite_oof_predictions(macro_estimator_factory, X_group, y_group, n_splits=n_splits, random_state=random_state)
        else:
            logger.warning("predicted_group_aggregate_feature: only %d unique groups (<max(2,n_splits)); falling back to the global target mean for every group.", len(unique_groups))
            oof_group_pred = np.full(len(unique_groups), float(y_arr.mean()))

        group_to_pred = dict(zip(unique_groups, oof_group_pred))
        fallback = float(np.nanmean(oof_group_pred)) if np.isfinite(oof_group_pred).any() else float(y_arr.mean())
        broadcast = np.array([group_to_pred.get(g, fallback) for g in group_arr], dtype=np.float64)
        broadcast = np.where(np.isfinite(broadcast), broadcast, fallback)

        return pl.DataFrame({column_name: broadcast})

    # Multi-statistic opt-in path: shared X_group / fold split, one multi-output OOF fit.
    y_series = pd.Series(y_arr)
    Y_group = np.column_stack([y_series.groupby(group_arr).agg(a).reindex(unique_groups).to_numpy(dtype=np.float64) for a in aggs])

    if len(unique_groups) >= max(2, n_splits):
        oof_matrix = _oof_multi_target_predictions(macro_estimator_factory, X_group, Y_group, n_splits=n_splits, random_state=random_state)
    else:
        logger.warning("predicted_group_aggregate_feature: only %d unique groups (<max(2,n_splits)); falling back to the global target statistic for every group.", len(unique_groups))
        global_stats = pd.Series(y_arr).agg(list(aggs)).to_numpy(dtype=np.float64)
        oof_matrix = np.tile(global_stats, (len(unique_groups), 1))

    out_columns: dict[str, np.ndarray] = {}
    for j, a in enumerate(aggs):
        col = oof_matrix[:, j]
        group_to_pred = dict(zip(unique_groups, col))
        fallback = float(np.nanmean(col)) if np.isfinite(col).any() else float(pd.Series(y_arr).agg(a))
        broadcast = np.array([group_to_pred.get(g, fallback) for g in group_arr], dtype=np.float64)
        broadcast = np.where(np.isfinite(broadcast), broadcast, fallback)
        out_columns[f"{column_name}__{a}"] = broadcast

    return pl.DataFrame(out_columns)


__all__ = ["predicted_group_aggregate_feature"]
