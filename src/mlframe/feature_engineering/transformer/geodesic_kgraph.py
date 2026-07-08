"""Geodesic distance via kNN graph (multi-source Dijkstra).

Iter 85 mechanism. Geometric agent's #3 ranked.

For binary: per train row compute geodesic distance to nearest opposite-class train row via kNN graph
(k=10) shortest path (scipy.sparse.csgraph.dijkstra). For each query: K NN train rows in raw X, aggregate
their geodesic distances. 5 features.

For regression: same but distance = to nearest top-quintile y train row from each bottom-quintile y row,
and vice versa.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_geodesic_kgraph_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    k_graph: int = 10,
    k_query: int = 16,
    standardize: bool = True,
    column_prefix: str = "geo",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Geodesic distance features via kNN graph.

    Output: 5 features per query — mean_geo, max_geo, min_geo, std_geo, median_geo.
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features_out = 5

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        n_t = Xt_s.shape[0]
        # Build kNN graph on train.
        nn_graph = NearestNeighbors(n_neighbors=k_graph, n_jobs=-1).fit(Xt_s)
        dists, idxs = nn_graph.kneighbors(Xt_s)
        # Build sparse graph: edge weight = distance.
        rows = np.repeat(np.arange(n_t), k_graph)
        cols = idxs.ravel()
        weights = dists.ravel()
        graph = csr_matrix((weights, (rows, cols)), shape=(n_t, n_t))
        # Symmetrize
        graph = graph.maximum(graph.T)

        # Identify "target" rows (opposite class for binary, top/bottom y quintile for regression).
        if task == "binary":
            target_indices = np.where(y_t > 0.5)[0]  # positive class as source
        else:
            q1 = np.quantile(y_t, 0.20)
            target_indices = np.where(y_t <= q1)[0]  # bottom quintile y as source

        # Multi-source Dijkstra from target_indices.
        if target_indices.size > 0:
            geo_dist_train = dijkstra(graph, indices=target_indices, return_predecessors=False, directed=False, limit=np.inf)
            # Take min over sources for each row.
            min_geo_per_train = geo_dist_train.min(axis=0)
            # Replace inf with max non-inf value
            finite_max = float(min_geo_per_train[np.isfinite(min_geo_per_train)].max()) if np.any(np.isfinite(min_geo_per_train)) else 1.0
            min_geo_per_train = np.where(np.isfinite(min_geo_per_train), min_geo_per_train, finite_max).astype(np.float32)
        else:
            min_geo_per_train = np.zeros(n_t, dtype=np.float32)

        # For each query, find K NN in train (raw X), aggregate min_geo_per_train.
        k_q_eff = min(k_query, n_t)
        nn_q = NearestNeighbors(n_neighbors=k_q_eff, n_jobs=-1).fit(Xt_s)
        _, q_idx = nn_q.kneighbors(Xq_s)
        nbr_geo = min_geo_per_train[q_idx]  # (n_q, k_q_eff)
        mean_geo = nbr_geo.mean(axis=1).astype(np.float32)
        max_geo = nbr_geo.max(axis=1).astype(np.float32)
        min_geo = nbr_geo.min(axis=1).astype(np.float32)
        std_geo = nbr_geo.std(axis=1).astype(np.float32) + 1e-9
        median_geo = np.median(nbr_geo, axis=1).astype(np.float32)
        return np.column_stack([mean_geo, max_geo, min_geo, std_geo, median_geo])

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_mean"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_max"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_min"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_std"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_median"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out: np.ndarray = np.zeros((n_train, n_features_out), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx])
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("geodesic_kgraph: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
