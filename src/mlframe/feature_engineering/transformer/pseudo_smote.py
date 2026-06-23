"""Pseudo-label-filtered SMOTE virtuals: generate vanilla SMOTE virtuals, filter by aux LGB confidence.

Iter 43 mechanism. BEYOND-FROZEN additive — uses aux LGB's gradient-trained predictions as a filter on synthesized virtuals.

Two-stage pipeline:
1. Generate vanilla SMOTE virtuals from positive-class subset (intra-positive convex interpolation, iter 33 style). Generate K × oversample virtuals for liberal initial pool.
2. Train aux LGB on (X_train_fold, y_train_fold).
3. Score the virtuals with the aux LGB.
4. Keep only virtuals with predicted ``P(y=1) >= confidence_threshold`` (default 0.7).
5. Per query: distances to k-th nearest filtered-virtual + signed log-gap vs real-negative distances.

Why this is structurally different from prior virtual mechanisms:
- iter 33 SMOTE: every interpolation kept, no filtering.
- iter 34 Borderline-SMOTE: pre-filter on real positives only (which positives to interpolate FROM).
- iter 43 pseudo-SMOTE: post-filter on virtuals (which virtuals to KEEP after generation).

The aux LGB's filter ensures virtuals fall in regions the aux model considers likely positive. This is a "self-consistent SMOTE" — virtuals must be predicted positive
by the same model trained on real data. Removes SMOTE virtuals that wander into the negative manifold.

Why CB-blind:
- CB cannot fit aux LGB internally during its own training.
- The filter creates a virtual cloud SHAPE that depends on the aux LGB's decision boundary — a complex nonlinear filter CB has no way to compute.

Leakage discipline: aux LGB refit per fold from train-fold rows only. Virtuals generated and filtered per fold.

Cost: SMOTE O(n_pos × oversample) + aux LGB fit O(N × n_estimators × depth²) + LGB predict on virtuals + kNN. Aux LGB fit dominates (~1-2 sec per fold).

References:
- Chawla 2002 — SMOTE.
- Han, Wang, Mao 2005 — Borderline-SMOTE (pre-filtering source positives).
- "Pseudo-labeling" semi-supervised technique (Lee 2013) — confidence-based filtering of model predictions, here adapted to synthetic data.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_K_SCALES = (1, 3, 5, 10)


def _smote_synthesize_intra(X_minority: np.ndarray, n_synthetic: int, k_neighbors: int, seed: int) -> np.ndarray:
    """SMOTE-interpolate among minority subset (same as iter 33 vanilla SMOTE)."""
    n_min = X_minority.shape[0]
    if n_min < 2:
        return X_minority.copy() if n_min > 0 else np.zeros((0, 0), dtype=np.float32)
    from sklearn.neighbors import NearestNeighbors
    k_used = min(k_neighbors + 1, n_min)
    nn = NearestNeighbors(n_neighbors=k_used).fit(X_minority)
    _dists, ids = nn.kneighbors(X_minority)
    rng = np.random.default_rng(seed)
    # Draw (src, nbr, alpha) in the exact interleaved per-iteration order the PCG64 stream produced before,
    # then gather + convex-interpolate as one vectorized pass — bit-identical to the row loop. The draw order
    # is load-bearing: batching the draws would change WHICH neighbours interpolate and break selection downstream.
    src = np.empty(n_synthetic, dtype=np.int64)
    nbr = np.empty(n_synthetic, dtype=np.int64)
    alpha = np.empty(n_synthetic, dtype=np.float32)
    for i in range(n_synthetic):
        s = rng.integers(0, n_min)
        candidates = ids[s, 1:k_used]
        src[i] = s
        if candidates.size == 0:
            nbr[i] = s
            alpha[i] = np.float32(0.0)
            continue
        nbr[i] = candidates[rng.integers(0, candidates.size)]
        alpha[i] = rng.random()
    x_src = X_minority[src]
    return (x_src + alpha[:, None] * (X_minority[nbr] - x_src)).astype(np.float32)


def _fit_aux_lgb_and_filter(X_train: np.ndarray, y_train: np.ndarray, virtuals: np.ndarray, task: str, seed: int, threshold: float, n_estimators: int = 200, max_depth: int = 4) -> np.ndarray:
    """Train aux LGB and keep only virtuals with predicted P(y=1) >= threshold (binary) or top-quantile predicted (regression)."""
    import lightgbm as lgb
    params = dict(
        n_estimators=n_estimators, max_depth=max_depth, num_leaves=2 ** max_depth,
        learning_rate=0.05, random_state=seed, n_jobs=-1, verbose=-1, min_data_in_leaf=5,
    )
    if task == "binary":
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(virtuals)[:, 1]
        keep_mask = proba >= threshold
    else:
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(virtuals)
        # For regression, keep virtuals predicted in top quintile (analogous to confidence threshold).
        pred_threshold = float(np.quantile(model.predict(X_train), 0.8))
        keep_mask = pred >= pred_threshold
    filtered = virtuals[keep_mask]
    if filtered.shape[0] < 10:
        # Fallback: if too few pass filter, lower threshold to keep top-N most confident.
        # Wave 62 (2026-05-20): lexsort with row-index tiebreak for deterministic
        # top-K across runs on tied proba/pred values.
        if task == "binary":
            top_k = min(max(10, len(virtuals) // 10), len(virtuals))
            top_idx = np.lexsort((np.arange(len(proba)), -proba))[:top_k]
            filtered = virtuals[top_idx]
        else:
            top_k = min(max(10, len(virtuals) // 10), len(virtuals))
            top_idx = np.lexsort((np.arange(len(pred)), -pred))[:top_k]
            filtered = virtuals[top_idx]
    return filtered


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


def compute_pseudo_smote_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    oversample: float = 10.0,
    k_smote: int = 5,
    confidence_threshold: float = 0.7,
    q_high: float = 0.8,
    standardize: bool = True,
    column_prefix: str = "psmote",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Pseudo-label-filtered SMOTE virtuals: confidence-filtered convex-interpolation virtuals.

    Output: 8 columns per row — 4 distances to k-th nearest filtered-virtual + 4 signed log-gaps vs real-negatives.
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
        # Generate liberal initial SMOTE pool.
        n_synthetic = max(100, int(Xt_pos.shape[0] * oversample * 2))
        raw_virtuals = _smote_synthesize_intra(Xt_pos, n_synthetic=n_synthetic, k_neighbors=k_smote, seed=fold_seed)
        # Filter via aux LGB confidence.
        filtered_virtuals = _fit_aux_lgb_and_filter(Xt_s, y_t, raw_virtuals, task=task, seed=fold_seed, threshold=confidence_threshold)
        # Combine real + filtered virtuals.
        X_virtual_pos = np.concatenate([Xt_pos, filtered_virtuals], axis=0)
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
        logger.info("pseudo_smote: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
