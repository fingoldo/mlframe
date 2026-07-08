"""Multi-scale SMOTE: vanilla SMOTE at multiple k_neighbors interpolation scales.

Iter 47 mechanism. Extends iter 33 vanilla SMOTE the same way iter 45 extended iter 41 BGM — by running the winning mechanism at multiple resolution scales.

SMOTE's k_neighbors parameter controls how broadly virtuals interpolate:
- Small k (e.g. 3): tight interpolation, virtuals stay close to source positives' tight neighborhood — narrow virtual cloud, conservative.
- Medium k (e.g. 8): moderate interpolation breadth.
- Large k (e.g. 15): broad interpolation, virtuals can span larger gaps — wide virtual cloud, exploratory.

By running SMOTE at all three scales and exposing per-scale distance features, the boosting picks the most-informative scale per region of X.

Mechanism (binary):
1. For each k in {3, 8, 15}, generate K × n_pos × oversample SMOTE virtuals with that interpolation scale.
2. Per scale, compute distance features (k=1,3,5,10) + signed log-gap vs real-negatives.
3. Total: 3 scales × 8 features = 24 features per row.

For regression: SMOTE on top-quintile-y rows.

Why this should work: multi-scale BGM (iter 45) was a record-setter (+3.6pp LGB PR_AUC) — the multi-resolution principle works in the sampling family.

Leakage discipline: SMOTE generated per scale per fold from train-fold positives only.

Cost: 3 × iter-33's SMOTE cost. Each SMOTE is sub-second; total ~1-2 sec per fold.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_K_SCALES = (1, 3, 5, 10)
_SMOTE_K_DEFAULT = (3, 8, 15)


def _smote_with_k(X_minority: np.ndarray, n_synthetic: int, k_neighbors: int, seed: int) -> np.ndarray:
    """Vanilla SMOTE with explicit k_neighbors."""
    n_min = X_minority.shape[0]
    if n_min < 2:
        return X_minority.copy() if n_min > 0 else np.zeros((0, X_minority.shape[1] if n_min > 0 else 1), dtype=np.float32)
    from sklearn.neighbors import NearestNeighbors
    k_used = min(k_neighbors + 1, n_min)
    nn = NearestNeighbors(n_neighbors=k_used).fit(X_minority)
    _dists, ids = nn.kneighbors(X_minority)
    rng = np.random.default_rng(seed)
    out = np.zeros((n_synthetic, X_minority.shape[1]), dtype=np.float32)
    for i in range(n_synthetic):
        src_idx = rng.integers(0, n_min)
        candidates = ids[src_idx, 1:k_used]
        if candidates.size == 0:
            out[i] = X_minority[src_idx]
            continue
        nbr_idx = candidates[rng.integers(0, candidates.size)]
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


def compute_multiscale_smote_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    smote_k_scales: Tuple[int, ...] = _SMOTE_K_DEFAULT,
    oversample: float = 5.0,
    q_high: float = 0.8,
    standardize: bool = True,
    column_prefix: str = "mss",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Multi-scale SMOTE distance features.

    Output: ``len(smote_k_scales) * 2 * len(_K_SCALES)`` columns. Default 3 × 2 × 4 = 24 features.
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
            return np.zeros((Xq_s.shape[0], 2 * len(_K_SCALES) * len(smote_k_scales)), dtype=np.float32)
        neg_d = _kth_nearest_dists(Xt_neg, Xq_s, max(_K_SCALES))
        n_synthetic = max(50, int(Xt_pos.shape[0] * oversample))
        all_feats = []
        for scale_idx, k_smote in enumerate(smote_k_scales):
            X_synth = _smote_with_k(Xt_pos, n_synthetic=n_synthetic, k_neighbors=k_smote, seed=fold_seed + scale_idx * 13)
            X_virtual_pos = np.concatenate([Xt_pos, X_synth], axis=0)
            pos_d = _kth_nearest_dists(X_virtual_pos, Xq_s, max(_K_SCALES))
            log_gap = np.log(np.maximum(neg_d, 1e-9)) - np.log(np.maximum(pos_d, 1e-9))
            all_feats.append(pos_d)
            all_feats.append(log_gap)
        return np.asarray(np.concatenate(all_feats, axis=1).astype(np.float32))

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        col_idx = 0
        for k_smote in smote_k_scales:
            tag = f"sk{k_smote}"
            for k in _K_SCALES:
                cols[f"{column_prefix}_{tag}_pos_k{k}"] = feats[:, col_idx].astype(dtype, copy=False)
                col_idx += 1
            for k in _K_SCALES:
                cols[f"{column_prefix}_{tag}_loggap_k{k}"] = feats[:, col_idx].astype(dtype, copy=False)
                col_idx += 1
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f, seed)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    n_features = 2 * len(_K_SCALES) * len(smote_k_scales)
    out: np.ndarray = np.zeros((n_train, n_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("multiscale_smote: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
