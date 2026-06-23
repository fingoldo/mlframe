"""Density-weighted SMOTE: sample source positives with probability inversely proportional to local kNN density.

Iter 50 mechanism. BEYOND-FROZEN learned-weighting in sampling family.

Vanilla SMOTE (iter 33) samples source positives uniformly. Density-weighted SMOTE biases the source-positive selection toward isolated/sparse positives — these
typically represent rare positive-class patterns that the boosting struggles with most. Oversampling sparse regions gives the boosting more virtuals where it
needs them.

Mechanism (binary):
1. Compute local density per real positive: density_i = 1 / mean(top-k=5 NN distances among positives).
2. Sample probability w_i ∝ 1 / density_i  (inverse-density weighting).
3. SMOTE-interpolate: pick source positive with weight w_i, find its kNN among positives, interpolate.
4. Combine real + density-weighted-SMOTE virtuals.
5. Per query: distance features (k=1,3,5,10) + signed log-gap.

For regression: density-weight by top-quintile-y rows.

Why beyond-frozen: per-positive sampling weight is LEARNED from data (local density estimate), not hand-crafted uniform.

Why this should help on mammography rare-positive:
- 52 real positives may have ~5-10 sparse "outlier-like" positives that vanilla SMOTE ignores in proportion.
- Density-weighted SMOTE generates more virtuals around these sparse positives, surfacing their rare-pattern structure.

Leakage discipline: positives' density computed per fold from train-fold positives only.

Cost: kNN on n_pos rows + weighted SMOTE; ~0.5 sec per fold.

References:
- ADASYN (He et al. 2008) — similar idea: adaptive synthetic sampling weighted by per-positive difficulty (negative-neighbor fraction).
- Density-weighted SMOTE simplifies ADASYN: use POSITIVE-density not negative-neighbor fraction.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_K_SCALES = (1, 3, 5, 10)


def _density_weighted_smote_synthesize(X_minority: np.ndarray, n_synthetic: int, k_neighbors: int, seed: int) -> np.ndarray:
    """Sample source positives with inverse-density weight, then SMOTE-interpolate."""
    n_min = X_minority.shape[0]
    if n_min < 2:
        return X_minority.copy() if n_min > 0 else np.zeros((0, X_minority.shape[1] if n_min > 0 else 1), dtype=np.float32)
    from sklearn.neighbors import NearestNeighbors
    k_used = min(k_neighbors + 1, n_min)
    nn = NearestNeighbors(n_neighbors=k_used).fit(X_minority)
    dists, ids = nn.kneighbors(X_minority)
    # Per-positive density: 1 / mean(top-k NN distances). Sparse positives have small density.
    mean_knn_dist = dists[:, 1:].mean(axis=1) + 1e-9  # exclude self at column 0
    density = 1.0 / mean_knn_dist  # high density = dense region
    # Inverse-density weight for sampling: low density → high weight.
    weights = mean_knn_dist  # already proportional to 1/density (since density = 1/mean_dist)
    weights = weights / weights.sum()  # normalize to probability distribution
    rng = np.random.default_rng(seed)
    src = rng.choice(n_min, size=n_synthetic, p=weights)
    # Draw (nbr, alpha) in the exact per-iteration order the PCG64 stream produced them (alpha truncated to
    # float32 to mirror the legacy float32 store-rounding), then hoist the gather + convex interpolation into a
    # single vectorized pass. Bit-identical to the row loop; ~1.4x faster on the synthesize step.
    nbr = np.empty(n_synthetic, dtype=np.int64)
    alpha = np.empty(n_synthetic, dtype=np.float32)
    for i in range(n_synthetic):
        candidates = ids[src[i], 1:k_used]
        if candidates.size == 0:
            nbr[i] = src[i]
            alpha[i] = np.float32(0.0)
            continue
        nbr[i] = candidates[rng.integers(0, candidates.size)]
        alpha[i] = rng.random()
    x_src = X_minority[src]
    return (x_src + alpha[:, None] * (X_minority[nbr] - x_src)).astype(np.float32)


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


def compute_density_weighted_smote_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    oversample: float = 10.0,
    k_smote: int = 5,
    q_high: float = 0.8,
    standardize: bool = True,
    column_prefix: str = "dwsmote",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Density-weighted SMOTE distance features.

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
        X_synth_pos = _density_weighted_smote_synthesize(Xt_pos, n_synthetic=n_synthetic, k_neighbors=k_smote, seed=fold_seed)
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
        logger.info("density_weighted_smote: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
