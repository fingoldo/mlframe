"""ADASYN-style boundary-weighted SMOTE: source-positive sampling weight ∝ fraction of negative neighbors.

Iter 51 mechanism. Faithful implementation of He et al. 2008 ADASYN (Adaptive Synthetic Sampling).

Mechanism (binary):
1. For each real positive, compute fraction of negative-class rows among its top-k_global kNN in the FULL dataset.
2. Source-positive sampling weight w_i = neg_fraction_i / Σ_j neg_fraction_j.
3. SMOTE-interpolate, picking source positives weighted by w. Positives near negatives (high neg_fraction) get more virtuals.
4. Combine real + ADASYN-weighted-SMOTE virtuals.
5. Per query: distance features.

For regression: source weight = fraction of bottom-quintile-y rows among kNN (analogue).

Difference from iter 34 Borderline-SMOTE: iter 34 uses HARD cutoff (keep positives with neg_fraction > 0.5); iter 51 uses GRADIENT (weight by neg_fraction). Soft version
should retain more diversity than hard cutoff.

Difference from iter 50 density-weighted: iter 50 weights by INVERSE LOCAL DENSITY among positives (sparse positives); iter 51 weights by NEGATIVE FRACTION (boundary positives).

Different sub-population of positives gets oversampled in each mechanism:
- iter 50 dwsmote: sparse positives (deep in positive region but far from each other).
- iter 51 adasyn: boundary positives (near negatives).

Leakage discipline: kNN computed per fold from train-fold rows only.

Cost: kNN search + weighted sampling + SMOTE; ~0.5 sec per fold.

Reference: He et al. 2008 — ADASYN.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_K_SCALES = (1, 3, 5, 10)


def _adasyn_synthesize(X_minority: np.ndarray, X_full: np.ndarray, y_binary_full: np.ndarray, n_synthetic: int, k_smote: int, k_global: int, seed: int) -> np.ndarray:
    """ADASYN-style weighted SMOTE: source-positive weight = fraction of negative neighbors in full dataset."""
    n_min = X_minority.shape[0]
    if n_min < 2:
        return X_minority.copy() if n_min > 0 else np.zeros((0, X_minority.shape[1] if n_min > 0 else 1), dtype=np.float32)
    from sklearn.neighbors import NearestNeighbors
    # Find kNN of each positive in the FULL dataset to compute neg_fraction per positive.
    nn_full = NearestNeighbors(n_neighbors=k_global + 1).fit(X_full)
    _d_full, ids_full = nn_full.kneighbors(X_minority)
    neg_fraction = (y_binary_full[ids_full[:, 1:]] <= 0.5).mean(axis=1)
    # ADASYN weight: ∝ neg_fraction (with epsilon to avoid all-zero).
    weights = neg_fraction + 1e-6
    weights = weights / weights.sum()
    # kNN among positives for interpolation candidates.
    nn_pos = NearestNeighbors(n_neighbors=min(k_smote + 1, n_min)).fit(X_minority)
    _d_pos, ids_pos = nn_pos.kneighbors(X_minority)
    rng = np.random.default_rng(seed)
    src = rng.choice(n_min, size=n_synthetic, p=weights)
    # Draw (nbr, alpha) in the exact per-iteration order the PCG64 stream produced them (alpha truncated to
    # float32 to mirror the legacy float32 store-rounding), then hoist the gather + convex interpolation into a
    # single vectorized pass. Bit-identical to the row loop; ~1.5x faster on the synthesize step.
    k_hi = min(k_smote + 1, n_min)
    nbr = np.empty(n_synthetic, dtype=np.int64)
    alpha = np.empty(n_synthetic, dtype=np.float32)
    for i in range(n_synthetic):
        candidates = ids_pos[src[i], 1:k_hi]
        if candidates.size == 0:
            nbr[i] = src[i]
            alpha[i] = np.float32(0.0)
            continue
        nbr[i] = candidates[rng.integers(0, candidates.size)]
        alpha[i] = rng.random()
    x_src = X_minority[src]
    return np.asarray((x_src + alpha[:, None] * (X_minority[nbr] - x_src)).astype(np.float32))


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


def compute_adasyn_smote_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    oversample: float = 10.0,
    k_smote: int = 5,
    k_global: int = 10,
    q_high: float = 0.8,
    standardize: bool = True,
    column_prefix: str = "adasyn",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """ADASYN-style boundary-weighted SMOTE distance features.

    Output: 8 columns per row.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _slice_and_binarize(X_sub: np.ndarray, y_sub: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if task == "binary":
            pos_mask = y_sub > 0.5
            y_binary = pos_mask.astype(np.float32)
        else:
            y_hi = np.quantile(y_sub, q_high)
            pos_mask = y_sub >= y_hi
            y_binary = pos_mask.astype(np.float32)
        return X_sub[pos_mask], X_sub[~pos_mask], y_binary

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, fold_seed: int) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        Xt_pos, Xt_neg, y_binary = _slice_and_binarize(Xt_s, y_t)
        if Xt_pos.shape[0] < 2 or Xt_neg.shape[0] < 2:
            return np.zeros((Xq_s.shape[0], 2 * len(_K_SCALES)), dtype=np.float32)
        n_synthetic = max(50, int(Xt_pos.shape[0] * oversample))
        X_synth_pos = _adasyn_synthesize(Xt_pos, Xt_s, y_binary, n_synthetic=n_synthetic, k_smote=k_smote, k_global=k_global, seed=fold_seed)
        X_virtual_pos = np.concatenate([Xt_pos, X_synth_pos], axis=0)
        pos_d = _kth_nearest_dists(X_virtual_pos, Xq_s, max(_K_SCALES))
        neg_d = _kth_nearest_dists(Xt_neg, Xq_s, max(_K_SCALES))
        log_gap = np.log(np.maximum(neg_d, 1e-9)) - np.log(np.maximum(pos_d, 1e-9))
        return np.asarray(np.concatenate([pos_d, log_gap], axis=1).astype(np.float32))

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
    out: np.ndarray = np.zeros((n_train, 2 * len(_K_SCALES)), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx], int(seed) + fold_idx * 100)
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("adasyn_smote: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
