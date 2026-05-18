"""MIXUP-boundary virtual distance features: synthesize across class boundary by positive-negative interpolation.

Iter 35 mechanism. Extends iter 33 vanilla SMOTE (intra-positive) and iter 34 borderline-SMOTE (filtered intra-positive) by crossing the class boundary itself.

Mechanism (binary):
1. Sample random (positive, negative) pairs.
2. For each pair, generate virtual at ``alpha * positive + (1 - alpha) * negative`` where ``alpha ~ Uniform(0.6, 0.9)``.
3. These virtuals lie on the POSITIVE side of the decision boundary — they represent "what a positive would look like if pulled slightly toward the negative cloud".
4. Per query: distances to k=1,3,5,10-th nearest mixup-virtual + signed log-gap vs real-negative distances.

Why MIXUP-boundary differs from SMOTE:
- SMOTE (iter 33): virtual = α × pos1 + (1 − α) × pos2 — both endpoints positive, intra-class.
- MIXUP-boundary (iter 35): virtual = α × pos + (1 − α) × neg — endpoints cross class boundary, captures boundary geometry.

For mammography rare-positive: SMOTE thickens the positive cloud; MIXUP-boundary fills the boundary-zone where positives push against negatives. CB needs both kinds of signal — the iter 33 win plus boundary geometry should give complementary information.

Reference: Zhang et al. 2018 — mixup. Originally for NN training augmentation; here repurposed as a frozen feature generator.

Leakage discipline: virtual generation per fold from train-fold pos/neg slices only.

Cost: per fold, generate n_synthetic = 5 × n_positives virtual rows. kNN search on the virtual pool against query rows. Fast.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_K_SCALES = (1, 3, 5, 10)


def _mixup_boundary(X_pos: np.ndarray, X_neg: np.ndarray, n_synthetic: int, alpha_low: float, alpha_high: float, seed: int) -> np.ndarray:
    """Generate n_synthetic virtual rows by random positive-negative convex combination at alpha ∈ [alpha_low, alpha_high]."""
    n_pos, n_neg = X_pos.shape[0], X_neg.shape[0]
    if n_pos == 0 or n_neg == 0:
        return np.zeros((0, X_pos.shape[1] if n_pos > 0 else X_neg.shape[1]), dtype=np.float32)
    rng = np.random.default_rng(seed)
    pos_idx = rng.integers(0, n_pos, size=n_synthetic)
    neg_idx = rng.integers(0, n_neg, size=n_synthetic)
    alphas = rng.uniform(alpha_low, alpha_high, size=n_synthetic).astype(np.float32)
    virtuals = alphas[:, None] * X_pos[pos_idx] + (1.0 - alphas[:, None]) * X_neg[neg_idx]
    return virtuals.astype(np.float32)


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


def compute_mixup_boundary_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    n_synthetic_multiplier: float = 5.0,
    alpha_low: float = 0.6,
    alpha_high: float = 0.9,
    q_high: float = 0.8,
    standardize: bool = True,
    column_prefix: str = "mixup",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """MIXUP-boundary virtual distance features.

    Output: 8 columns per row — 4 distances to k-th nearest mixup-virtual + 4 signed log-gaps vs real-negative.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if not (0.5 < alpha_low < alpha_high < 1.0):
        raise ValueError(f"alpha range must satisfy 0.5 < alpha_low < alpha_high < 1.0; got [{alpha_low}, {alpha_high}].")

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
        n_synthetic = max(50, int(Xt_pos.shape[0] * n_synthetic_multiplier))
        X_mixup = _mixup_boundary(Xt_pos, Xt_neg, n_synthetic=n_synthetic, alpha_low=alpha_low, alpha_high=alpha_high, seed=fold_seed)
        # Combine real positives + mixup virtuals to capture both cores and boundary.
        X_virtual_pos = np.concatenate([Xt_pos, X_mixup], axis=0)
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
        logger.info("mixup_boundary: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
