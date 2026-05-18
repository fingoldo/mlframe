"""SMOTE-synthetic positive / quantile distance features: virtual minority-class density via interpolation.

Iter 33 mechanism. Addresses rare-positive mammography (1.3%) where the real positive cloud is too sparse for `cdist` (iter 27) to capture continuous boundary geometry.

Mechanism (binary):
1. Apply SMOTE-style synthesis on positive-class train rows: for each real positive, sample k_neighbors=5 of its nearest positive neighbors, interpolate at random
   convex combinations to generate `n_synthetic = real_positives × oversample` virtual positives.
2. For each query row, compute distances to k=1,3,5,10-th nearest VIRTUAL positive — 4 features.
3. Plus signed log-gap vs real-negative-distance — 4 features. Total: 8 features per row.

For regression: replace "positive class" with "top-quintile-y" rows; SMOTE-interpolate among them to imagine the dense version of the rare upper-tail.

Why CB-blind: CB cannot synthesize examples — its splits operate on the observed empirical X distribution. Virtual positives expose density geometry that real-data
kNN (iter 27 cdist) cannot reach when only ~52 real positives exist.

Why this differs from cdist (iter 27):
- cdist: distance to REAL nearest positive instance — preserves outliers but extremely sparse signal in 1.3%-positive regime.
- SMOTE-distance: distance to NEAREST INTERPOLATED positive — smooth, continuous "imagined positive cloud" coverage.

Leakage discipline: SMOTE refit per fold on train-fold positives only; val rows queried against the virtual cloud.

Cost: SMOTE generates n_synthetic = 5 × n_positives virtual rows per fold (~260 synthetic for mammography). kNN search on (n_synthetic + n_real_negatives) against
n_query. Fast.

References:
- Chawla et al. 2002 — SMOTE.
- Synthetic-data-derived features for imbalanced classification (not commonly used as a frozen FE primitive for boostings).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_K_SCALES = (1, 3, 5, 10)


def _smote_synthesize(X_minority: np.ndarray, n_synthetic: int, k_neighbors: int, seed: int) -> np.ndarray:
    """Generate n_synthetic virtual minority-class rows by SMOTE-style convex interpolation.

    For each iteration: pick a random minority row, find its k_neighbors nearest minority rows, pick one at random, interpolate at random alpha ∈ [0, 1].
    """
    n_min = X_minority.shape[0]
    if n_min < 2:
        return X_minority.copy()
    from sklearn.neighbors import NearestNeighbors
    k_used = min(k_neighbors + 1, n_min)
    nn = NearestNeighbors(n_neighbors=k_used).fit(X_minority)
    _dists, ids = nn.kneighbors(X_minority)  # (n_min, k_used) — self is at index 0
    rng = np.random.default_rng(seed)
    out = np.zeros((n_synthetic, X_minority.shape[1]), dtype=np.float32)
    for i in range(n_synthetic):
        src_idx = rng.integers(0, n_min)
        neighbor_candidates = ids[src_idx, 1:k_used]  # exclude self
        if neighbor_candidates.size == 0:
            out[i] = X_minority[src_idx]
            continue
        nbr_idx = neighbor_candidates[rng.integers(0, neighbor_candidates.size)]
        alpha = rng.random()
        out[i] = X_minority[src_idx] + alpha * (X_minority[nbr_idx] - X_minority[src_idx])
    return out.astype(np.float32)


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


def compute_smote_distance_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    oversample: float = 5.0,
    k_smote: int = 5,
    q_high: float = 0.8,
    standardize: bool = True,
    column_prefix: str = "smote",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """SMOTE-synthetic positive distance features for binary classification (or top-quintile-y for regression).

    Output: 8 columns per row — 4 distances to k-th nearest virtual positive + 4 signed log-gaps against real-negative distances.
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
        n_synthetic = max(2 * Xt_pos.shape[0], int(Xt_pos.shape[0] * oversample))
        X_synth_pos = _smote_synthesize(Xt_pos, n_synthetic=n_synthetic, k_neighbors=k_smote, seed=fold_seed)
        # Combine REAL + SYNTHETIC positives for virtual cloud.
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
        logger.info("smote_distance: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
