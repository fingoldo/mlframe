"""CutMix-style virtual distance features: hard feature swap between positive and negative class examples.

Iter 36 mechanism. Complements iter 33 SMOTE (intra-positive convex combo) and iter 35 MIXUP (cross-class convex combo) with a HARD-SWAP analog.

Mechanism (binary):
1. For each real positive, generate K virtuals by replacing a random subset (size cut_size × d) of its features with values from a randomly-chosen negative.
2. The resulting virtual has some features from positive-class and others from negative-class.
3. Per query: distances to k=1,3,5,10-th nearest CutMix-virtual + signed log-gap vs real-negatives.

Differs from MIXUP (iter 35) which uses convex combinations: CutMix uses HARD axis-aligned swaps. The result respects feature decorrelation — interpolation in CutMix preserves
each individual feature's marginal distribution (a value from either pos or neg class for each axis independently), where MIXUP creates feature values that are AVERAGES never seen
in real data.

For CB specifically: CutMix virtuals might land in regions where individual features look "positive-like" on some axes but "negative-like" on others — exposing axis-conditional class
geometry that CB's symmetric oblivious trees struggle with (CB splits one feature at a time; CutMix virtuals are mixtures across axes).

Reference: Yun et al. 2019 — CutMix data augmentation. Originally for CV; here repurposed as a frozen FE primitive.

Output: 8 features per row — 4 distances + 4 signed log-gaps.

Leakage discipline: virtual generation per fold from train-fold pos/neg slices only.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_K_SCALES = (1, 3, 5, 10)


def _cutmix_synthesize(X_pos: np.ndarray, X_neg: np.ndarray, n_synthetic: int, cut_fraction: float, seed: int) -> np.ndarray:
    """Generate n_synthetic CutMix virtuals.

    For each virtual: pick a random positive and a random negative; replace ~cut_fraction × d of the positive's features with the negative's values.
    """
    n_pos, n_neg = X_pos.shape[0], X_neg.shape[0]
    if n_pos == 0 or n_neg == 0:
        return np.zeros((0, X_pos.shape[1] if n_pos > 0 else X_neg.shape[1]), dtype=np.float32)
    rng = np.random.default_rng(seed)
    d = X_pos.shape[1]
    n_cut = max(1, int(round(cut_fraction * d)))
    pos_idx = rng.integers(0, n_pos, size=n_synthetic)
    neg_idx = rng.integers(0, n_neg, size=n_synthetic)
    out = X_pos[pos_idx].copy()
    # For each virtual, randomly select cut_fraction × d features to overwrite with negative-class values.
    for i in range(n_synthetic):
        cut_axes = rng.choice(d, size=n_cut, replace=False)
        out[i, cut_axes] = X_neg[neg_idx[i], cut_axes]
    return np.asarray(out.astype(np.float32))


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


def compute_cutmix_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    n_synthetic_multiplier: float = 5.0,
    cut_fraction: float = 0.3,
    q_high: float = 0.8,
    standardize: bool = True,
    column_prefix: str = "cutmix",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """CutMix-style hard-swap virtual distance features.

    Output: 8 columns per row — 4 distances to k-th nearest CutMix-virtual + 4 signed log-gaps vs real-negative distances.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if not (0.0 < cut_fraction < 1.0):
        raise ValueError(f"cut_fraction must be in (0, 1); got {cut_fraction}.")

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
        X_cutmix = _cutmix_synthesize(Xt_pos, Xt_neg, n_synthetic=n_synthetic, cut_fraction=cut_fraction, seed=fold_seed)
        X_virtual_pos = np.concatenate([Xt_pos, X_cutmix], axis=0)
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
        logger.info("cutmix: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
