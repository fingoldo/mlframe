"""``NeighborAggregateFeatures``: OOF-safe k-nearest-neighbor aggregation of arbitrary columns.

Source ideas: Home Credit 1st-place's ``neighbors_target_mean_500`` (mean TARGET of the 500 nearest
neighbors, defined by a small set of strong continuous raw features) -- the single highest-importance
feature in that competition; and Optiver Realized Volatility 1st-place's neighbor aggregation of
realized-volatility/size across similar time-ids/stock-ids (360 of ~600 features, the main score driver).

Distinct from :mod:`nn_oof_target_mean`, which embeds rows via 3 FITTED BASELINE MODELS (LGB depth-3,
LGB depth-5, Ridge/LogReg) before running kNN, and aggregates only the target with a fixed 3-stat bundle.
This module instead runs kNN directly on a caller-chosen RAW feature subset (matching Phil's literal
mechanism and Optiver's stock/time-id similarity aggregation, and avoiding a 3-model-fit cost per fold when
the caller already trusts a raw feature subset as the similarity space) and aggregates ANY caller-specified
column(s) -- not just the target -- which is what Optiver's "aggregate RV/size across similar ids" case
needs and ``nn_oof_target_mean`` cannot do.

Leakage discipline mirrors ``nn_oof_target_mean``'s two modes:
* Mode A (``X_query=None``): per outer fold (from ``splitter``), the kNN index is built ONLY on the fold's
  train rows; each val row's neighbor stats are computed from train-row values only -- no val row ever sees
  its own value or a neighbor computed from itself.
* Mode B (``X_query`` given): a single index over the full ``X_train``, queried by ``X_query`` (a genuinely
  held-out set, e.g. the real test set at inference time).

kNN backend is :func:`mlframe.feature_engineering.transformer._knn_helper.knn_search` (HNSW above the
per-host-tuned crossover, else exact sklearn ``NearestNeighbors``) -- no new kNN kernel needed.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import polars as pl

from ._knn_helper import knn_search
from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_DEFAULT_K_VALUES: tuple[int, ...] = (5, 10, 20, 40)
_DEFAULT_STATS: tuple[str, ...] = ("mean", "std")

_STAT_FUNCS = {
    "mean": lambda arr: arr.mean(axis=1),
    "std": lambda arr: arr.std(axis=1),
    "median": lambda arr: np.median(arr, axis=1),
    "min": lambda arr: arr.min(axis=1),
    "max": lambda arr: arr.max(axis=1),
}


_WEIGHTED_EPS = 1e-6


def _aggregate_one(
    X_index: np.ndarray,
    X_query: np.ndarray,
    agg_values: Mapping[str, np.ndarray],
    k_values: Sequence[int],
    stats: Sequence[str],
    dtype: type,
    column_prefix: str,
    distance_weighted: bool = False,
) -> dict[str, np.ndarray]:
    """Find each query row's kNN in ``X_index``, then for every (column, k, stat) combo compute the stat
    over the neighbors' values of that column.

    ``distance_weighted``: additionally emit a ``{prefix}_{col}_k{k}_wmean`` column per (column, k) --
    an inverse-distance-weighted mean (closer neighbors contribute more) instead of the plain unweighted
    mean. Opt-in; existing columns/values are unchanged (bit-identical) when False.
    """
    k_max = min(max(k_values), X_index.shape[0])
    neighbor_dists, neighbor_idx = knn_search(X_index, X_query, k_max)  # (n_q, k_max) each

    out: dict[str, np.ndarray] = {}
    for col_name, values in agg_values.items():
        neighbor_vals = np.asarray(values, dtype=np.float64)[neighbor_idx]  # (n_q, k_max)
        for k in k_values:
            k_eff = min(int(k), k_max)
            sliced = neighbor_vals[:, :k_eff]
            for stat_name in stats:
                stat_fn = _STAT_FUNCS[stat_name]
                out[f"{column_prefix}_{col_name}_k{k}_{stat_name}"] = stat_fn(sliced).astype(dtype, copy=False)
            if distance_weighted:
                sliced_dists = np.asarray(neighbor_dists, dtype=np.float64)[:, :k_eff]
                weights = 1.0 / (sliced_dists + _WEIGHTED_EPS)
                weights /= weights.sum(axis=1, keepdims=True)
                wmean = (sliced * weights).sum(axis=1)
                out[f"{column_prefix}_{col_name}_k{k}_wmean"] = wmean.astype(dtype, copy=False)
    return out


def compute_neighbor_aggregate_features(
    X_train: np.ndarray,
    agg_values: Mapping[str, np.ndarray],
    X_query: Optional[np.ndarray] = None,
    splitter: Optional[Any] = None,
    *,
    seed: int,
    k_values: Sequence[int] = _DEFAULT_K_VALUES,
    stats: Sequence[str] = _DEFAULT_STATS,
    standardize: bool = True,
    distance_weighted: bool = False,
    column_prefix: str = "nbr",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """OOF-safe k-nearest-neighbor aggregation of ``agg_values`` columns, in ``X_train``'s feature space.

    Parameters
    ----------
    X_train
        ``(n_train, d)`` numeric feature subset defining the neighbor-similarity space (a small set of
        strong continuous features, per the source ideas -- NOT the full feature matrix).
    agg_values
        ``{column_name: (n_train,) array}`` -- one or more columns to aggregate over each query row's
        neighbors (the target, or any other feature, e.g. realized-volatility/size in the Optiver case).
    X_query
        ``(n_query, d)`` -- Mode B: a genuinely held-out query set (e.g. real test rows). When None, Mode A
        (``splitter`` required) computes OOF features for ``X_train`` itself.
    splitter
        A CV splitter (``.split(X_train)`` -> ``(train_idx, val_idx)`` pairs) for Mode A.
    k_values
        Neighbor-count values to emit stats for (default ``(5, 10, 20, 40)``).
    stats
        Which aggregate stats to emit per (column, k): any of ``"mean"``, ``"std"``, ``"median"``,
        ``"min"``, ``"max"`` (default ``("mean", "std")``).
    standardize
        Robust-scale ``X_train``/``X_query`` before the kNN search (default True) -- puts features on
        comparable scales so no single raw-unit feature dominates the neighbor distance.
    distance_weighted
        When True, additionally emit an inverse-distance-weighted mean (``{prefix}_{col}_k{k}_wmean``)
        per (column, k) -- closer neighbors contribute more than farther ones within the same k-window,
        instead of every neighbor counting equally as in the plain ``"mean"`` stat. Opt-in (default
        False): the existing ``stats``-driven columns are computed identically either way.

    Returns
    -------
    pl.DataFrame
        ``len(agg_values) * len(k_values) * len(stats)`` columns, named
        ``{column_prefix}_{agg_col}_k{k}_{stat}``, plus one ``..._wmean`` column per (agg_col, k) when
        ``distance_weighted=True``.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    unknown_stats = set(stats) - set(_STAT_FUNCS)
    if unknown_stats:
        raise ValueError(f"Unknown stats: {sorted(unknown_stats)}; expected any of {sorted(_STAT_FUNCS)}")

    X_train_f = np.asarray(X_train, dtype=np.float32)
    k_list = tuple(int(k) for k in k_values)
    n_train = X_train_f.shape[0]
    for col_name, values in agg_values.items():
        arr = np.asarray(values)
        if arr.shape[0] != n_train:
            raise ValueError(f"agg_values[{col_name!r}] has {arr.shape[0]} rows, expected {n_train} (matching X_train)")

    def _scale(Xt: np.ndarray, Xq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit a RobustScaler on Xt and apply it to both Xt and Xq, or pass them through unchanged if standardize is off."""
        if not standardize:
            return Xt, Xq
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler().fit(Xt)
        return scaler.transform(Xt).astype(np.float32), scaler.transform(Xq).astype(np.float32)

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        Xt_s, Xq_s = _scale(X_train_f, Xq)
        cols = _aggregate_one(Xt_s, Xq_s, agg_values, k_list, stats, dtype, column_prefix, distance_weighted)
        return pl.DataFrame(cols)

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_stat_cols = len(stats) + (1 if distance_weighted else 0)
    n_features = len(agg_values) * len(k_list) * n_stat_cols
    col_order: list[str] = []
    out: np.ndarray = np.zeros((n_train, n_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_agg_values = {name: np.asarray(vals)[train_idx] for name, vals in agg_values.items()}
        Xt_s, Xq_s = _scale(X_train_f[train_idx], X_train_f[val_idx])
        fold_cols = _aggregate_one(Xt_s, Xq_s, fold_agg_values, k_list, stats, dtype, column_prefix, distance_weighted)
        if not col_order:
            col_order = list(fold_cols.keys())
        for j, name in enumerate(col_order):
            out[val_idx, j] = fold_cols[name]
        logger.info("neighbor_aggregate_features: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame({name: out[:, j] for j, name in enumerate(col_order)})


__all__ = ["compute_neighbor_aggregate_features"]
