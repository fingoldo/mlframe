"""K-means-cluster-SMOTE: SMOTE interpolation within positive-class K-means subclusters (sampling-family, beyond-frozen).

Iter 44 mechanism. Refines iter 33 vanilla SMOTE by interpolating ONLY within positive-class K-means subclusters — not across them.

Motivation: vanilla SMOTE picks pairs of positives uniformly, so when positives have multiple distinct sub-modes, SMOTE can interpolate across them, creating virtuals
in low-density "gap" regions between modes. Cluster-SMOTE first segments positives into K-means subclusters, then interpolates within each cluster — preserves
sub-mode geometry.

Mechanism (binary):
1. Fit K-means on positive-class rows (k=3 by default).
2. For each cluster, run SMOTE interpolation among its members (a × b convex combinations within cluster).
3. Combine real + cluster-SMOTE virtuals.
4. Per query: distance features (k=1,3,5,10) + signed log-gap vs real-negatives.

For regression: K-means on top-quintile-y rows.

Why beyond-frozen sampling family: K-means itself is a fitted model (gradient-style Lloyd's algorithm). The cluster assignment is LEARNED — different from purely
hand-crafted SMOTE.

Leakage discipline: K-means + SMOTE refit per fold from train-fold positives only.

Cost: K-means O(K · n_pos · d) + per-cluster SMOTE. Fast.

Reference: Douzas, Bacao, Last 2018 — K-Means SMOTE.

The published K-Means-SMOTE uses cluster-density weighting (more virtuals in dense clusters); we simplify to uniform virtuals-per-cluster.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_K_SCALES = (1, 3, 5, 10)


def _smote_within_cluster(X_cluster: np.ndarray, n_synthetic: int, k_neighbors: int, seed: int) -> np.ndarray:
    """SMOTE-interpolate within a single cluster of positives."""
    n_cluster = X_cluster.shape[0]
    if n_cluster < 2:
        return np.tile(X_cluster, (n_synthetic // max(1, n_cluster) + 1, 1))[:n_synthetic].astype(np.float32)
    from sklearn.neighbors import NearestNeighbors
    k_used = min(k_neighbors + 1, n_cluster)
    nn = NearestNeighbors(n_neighbors=k_used).fit(X_cluster)
    _dists, ids = nn.kneighbors(X_cluster)
    rng = np.random.default_rng(seed)
    out = np.zeros((n_synthetic, X_cluster.shape[1]), dtype=np.float32)
    for i in range(n_synthetic):
        src_idx = rng.integers(0, n_cluster)
        candidates = ids[src_idx, 1:k_used]
        if candidates.size == 0:
            out[i] = X_cluster[src_idx]
            continue
        nbr_idx = candidates[rng.integers(0, candidates.size)]
        alpha = rng.random()
        out[i] = X_cluster[src_idx] + alpha * (X_cluster[nbr_idx] - X_cluster[src_idx])
    return out


def _cluster_smote_synthesize(X_pos: np.ndarray, n_clusters: int, n_synthetic_total: int, k_neighbors: int, seed: int) -> np.ndarray:
    """Fit K-means on positives, then SMOTE within each cluster proportional to cluster size."""
    from sklearn.cluster import KMeans
    n_pos = X_pos.shape[0]
    if n_pos < n_clusters * 2:
        # Too few positives for K-means; fall back to flat SMOTE.
        return _smote_within_cluster(X_pos, n_synthetic_total, k_neighbors, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=3, max_iter=50)
        labels = km.fit_predict(X_pos)
    pieces = []
    for c in range(n_clusters):
        members = X_pos[labels == c]
        if members.shape[0] < 2:
            continue
        # Proportional allocation: this cluster's share of virtuals = its share of positives.
        share = int(n_synthetic_total * members.shape[0] / max(n_pos, 1))
        if share < 5:
            continue
        cluster_seed = seed + c * 37
        virtuals = _smote_within_cluster(members, share, k_neighbors=k_neighbors, seed=cluster_seed)
        pieces.append(virtuals)
    if not pieces:
        return _smote_within_cluster(X_pos, n_synthetic_total, k_neighbors, seed)
    return np.concatenate(pieces, axis=0).astype(np.float32)


def _kth_nearest_dists(X_subset: np.ndarray, X_query: np.ndarray, k_max: int) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors
    n_sub = X_subset.shape[0]
    if n_sub == 0:
        return np.full((X_query.shape[0], len(_K_SCALES)), 1e6, dtype=np.float32)
    k_request = min(k_max, n_sub)
    nn = NearestNeighbors(n_neighbors=k_request, algorithm="auto", n_jobs=-1).fit(X_subset)
    dists, _ids = nn.kneighbors(X_query)
    out = np.zeros((X_query.shape[0], len(_K_SCALES)), dtype=np.float32)
    for col_idx, k in enumerate(_K_SCALES):
        eff_k = min(k, k_request)
        out[:, col_idx] = dists[:, eff_k - 1]
    return out


def compute_cluster_smote_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    n_clusters: int = 3,
    oversample: float = 10.0,
    k_smote: int = 5,
    q_high: float = 0.8,
    standardize: bool = True,
    column_prefix: str = "csmote",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """K-means-cluster-SMOTE virtual positive distance features.

    Output: 8 columns per row.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _slice(X_sub: np.ndarray, y_sub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if task == "binary":
            pos = y_sub > 0.5
            return X_sub[pos], X_sub[~pos]
        y_hi = np.quantile(y_sub, q_high)
        y_lo = np.quantile(y_sub, 1.0 - q_high)
        return X_sub[y_sub >= y_hi], X_sub[y_sub <= y_lo]

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        Xt_pos, Xt_neg = _slice(Xt_s, y_t)
        if Xt_pos.shape[0] < 2 or Xt_neg.shape[0] < 2:
            return np.zeros((Xq_s.shape[0], 2 * len(_K_SCALES)), dtype=np.float32)
        n_synthetic = max(50, int(Xt_pos.shape[0] * oversample))
        n_clusters_eff = max(2, min(n_clusters, Xt_pos.shape[0] // 4))
        X_synth_pos = _cluster_smote_synthesize(Xt_pos, n_clusters=n_clusters_eff, n_synthetic_total=n_synthetic, k_neighbors=k_smote, seed=fold_seed)
        X_virtual_pos = np.concatenate([Xt_pos, X_synth_pos], axis=0)
        pos_d = _kth_nearest_dists(X_virtual_pos, Xq_s, max(_K_SCALES))
        neg_d = _kth_nearest_dists(Xt_neg, Xq_s, max(_K_SCALES))
        log_gap = np.log(np.maximum(neg_d, 1e-9)) - np.log(np.maximum(pos_d, 1e-9))
        return np.concatenate([pos_d, log_gap], axis=1).astype(np.float32)

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        for j, k in enumerate(_K_SCALES):
            cols[f"{column_prefix}_pos_k{k}"] = feats[:, j].astype(dtype, copy=False)
        for j, k in enumerate(_K_SCALES):
            cols[f"{column_prefix}_loggap_k{k}"] = feats[:, len(_K_SCALES) + j].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, 2 * len(_K_SCALES)), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("cluster_smote: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
