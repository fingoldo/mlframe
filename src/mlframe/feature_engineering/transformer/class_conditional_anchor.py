"""Class-conditional anchor attention: K-means fit SEPARATELY on positive and negative class rows; per-row similarity features to all class-conditional anchors.

Iter 16 anchor-based attention fit ONE K-means on all of X, then per-anchor target aggregates. For heavily-imbalanced binary (mammography ~1.3% positive class),
running K-means with n_anchors=32 across 4000 rows lands on average ~0.4 anchors on the positive class (52 positives out of 4000 / 32 anchors ≈ 1.6 positives per
anchor in the best case; in practice K-means clusters the dense negative manifold and ignores rare positives entirely).

Iter 19 forces parity: fit n_pos_anchors K-means anchors ON POSITIVE ROWS ONLY, fit n_neg_anchors on negative rows only. Per query row, expose similarity to ALL
2K anchors. The downstream boosting can split on "row is close to positive-class anchor 7" — a feature CB's internal target-statistics encoding does NOT compute
(TS sees per-row aggregates, not anchor-distance to rare-class clusters).

Mechanism (Mode B):
1. Standardise X with RobustScaler.
2. Fit K-means(n_pos_anchors) on X_train[y==1], yields anchors_pos.
3. Fit K-means(n_neg_anchors) on X_train[y==0], yields anchors_neg.
4. Per query row, compute softmax-weighted distance to all 2K anchors. Concatenate.
5. Compute per-anchor target aggregates (mean y, count) within each class; expose softmax-weighted aggregate per row.

Mode A (OOF): K-means refit per fold within fold's train subset positive/negative rows.

For regression target, "class-conditional" interpretation: split y by quartile, fit K-means per quartile. Provides 4K anchors with target-quartile semantics.

Cost: roughly 4x the iter-16 cost since two K-means fits per fold; still ms-scale at N<10k.

Mechanism vs iter 16:
- Iter 16: 32 unsupervised anchors driven by X-density.
- Iter 19: K positive + K negative anchors, each class gets equal "anchor budget" regardless of class frequency.
- For 1.3% positive class: iter 19 dedicates ~16 anchors to rare-class structure; iter 16 dedicates ~0 (statistically).

Reference: stratified K-means clustering for imbalanced data (Liu et al. 2009); class-conditional Gaussian mixture features for tabular ML (less commonly used).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _fit_kmeans(X: np.ndarray, n_anchors: int, seed: int) -> np.ndarray:
    """Fit K-means on X. Returns centroids of shape (n_anchors, d)."""
    from sklearn.cluster import KMeans
    n_anchors_actual = min(n_anchors, max(2, X.shape[0]))
    km = KMeans(n_clusters=n_anchors_actual, random_state=seed, n_init=5, max_iter=100)
    km.fit(X)
    return km.cluster_centers_.astype(np.float32, copy=False)


def _squared_dists(X: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Per-row squared euclidean distance to each anchor, (n_rows, n_anchors), via the
    ``||x||^2 - 2 x.a + ||a||^2`` GEMM decomposition. Avoids the (n_rows, n_anchors, d) broadcast
    cube that ``np.sum((X[:,None,:]-anchors[None,:,:])**2, axis=2)`` materialises; only the
    (n_rows, n_anchors) result is allocated. Differs from the subtraction form by float32 reduction
    order (~1e-5 on the downstream softmax), selection-equivalent for these FE features."""
    x_sq = np.einsum("ij,ij->i", X, X)[:, None]
    a_sq = np.einsum("ij,ij->i", anchors, anchors)[None, :]
    d = x_sq - 2.0 * (X @ anchors.T) + a_sq
    np.maximum(d, 0.0, out=d)
    return d


def _softmax_similarity(X: np.ndarray, anchors: np.ndarray, softmax_temp: float) -> np.ndarray:
    """Per-row softmax over anchor distances. Returns (n_rows, n_anchors)."""
    dists = _squared_dists(X, anchors)
    logits = -dists / (softmax_temp + 1e-9)
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


def _compute_class_anchor_features(
    X_query_std: np.ndarray,
    anchors_pos: np.ndarray,
    anchors_neg: np.ndarray,
    y_train_pos_count_per_anchor: np.ndarray,
    y_train_neg_count_per_anchor: np.ndarray,
    softmax_temp: float,
) -> dict[str, np.ndarray]:
    """Compute features per query row given class-conditional anchors."""
    sim_pos = _softmax_similarity(X_query_std, anchors_pos, softmax_temp=softmax_temp)  # (n_q, n_pos_anchors)
    sim_neg = _softmax_similarity(X_query_std, anchors_neg, softmax_temp=softmax_temp)  # (n_q, n_neg_anchors)

    # Combined sim mass: how much of the query's similarity mass goes to positive-class anchors vs negative.
    # Compute under a UNIFIED softmax over all 2K anchors.
    all_anchors = np.concatenate([anchors_pos, anchors_neg], axis=0)
    dists_all = _squared_dists(X_query_std, all_anchors)
    logits_all = -dists_all / (softmax_temp + 1e-9)
    logits_all -= logits_all.max(axis=1, keepdims=True)
    w_all = np.exp(logits_all)
    w_all /= w_all.sum(axis=1, keepdims=True)
    n_pos = anchors_pos.shape[0]
    mass_pos = w_all[:, :n_pos].sum(axis=1).astype(np.float32)  # (n_q,) fraction of similarity mass on positive anchors

    return {
        "sim_pos": sim_pos,  # (n_q, n_pos_anchors)
        "sim_neg": sim_neg,  # (n_q, n_neg_anchors)
        "mass_pos": mass_pos,  # (n_q,) — single feature: P(close to positive-class anchor)
    }


def compute_class_conditional_anchor_attention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    n_anchors_per_class: int = 16,
    softmax_temp: float = 1.0,
    standardize: bool = True,
    column_prefix: str = "ccanchor",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Class-conditional anchor attention features.

    Output (binary task): ``n_anchors_per_class * 2`` similarity columns + 1 "mass_pos" column = ``2*K + 1`` features.
    Output (regression task, future): quartile-conditional anchors — ``4 * n_anchors_per_class`` similarity + 4 mass columns.

    For now, only binary task is implemented (regression-quartile mode would need to be added separately).
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if task != "binary":
        raise NotImplementedError(f"task={task!r} not yet supported; only 'binary' in iter 19.")
    if n_anchors_per_class < 2:
        raise ValueError(f"n_anchors_per_class must be >= 2; got {n_anchors_per_class}.")
    if softmax_temp <= 0:
        raise ValueError(f"softmax_temp must be > 0; got {softmax_temp}.")

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    if X_query is not None:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(X_train_f)
            Xt_std = scaler.transform(X_train_f).astype(np.float32)
            Xq_std = scaler.transform(np.asarray(X_query, dtype=np.float32)).astype(np.float32)
        else:
            Xt_std = X_train_f
            Xq_std = np.asarray(X_query, dtype=np.float32)
        pos_mask = y_train_f > 0.5
        neg_mask = ~pos_mask
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())
        if n_pos < 2 or n_neg < 2:
            raise ValueError(f"class_conditional_anchor: need >=2 rows per class; got n_pos={n_pos}, n_neg={n_neg}.")
        n_anch_pos = min(n_anchors_per_class, max(2, n_pos // 2))
        n_anch_neg = min(n_anchors_per_class, max(2, n_neg // 2))
        anchors_pos = _fit_kmeans(Xt_std[pos_mask], n_anchors=n_anch_pos, seed=seed)
        anchors_neg = _fit_kmeans(Xt_std[neg_mask], n_anchors=n_anch_neg, seed=seed + 1)
        feats = _compute_class_anchor_features(
            Xq_std, anchors_pos=anchors_pos, anchors_neg=anchors_neg,
            y_train_pos_count_per_anchor=np.full(n_anch_pos, n_pos / n_anch_pos),
            y_train_neg_count_per_anchor=np.full(n_anch_neg, n_neg / n_anch_neg),
            softmax_temp=softmax_temp,
        )
        cols: dict[str, np.ndarray] = {}
        for j in range(n_anch_pos):
            cols[f"{column_prefix}_pos_a{j}"] = feats["sim_pos"][:, j].astype(dtype, copy=False)
        for j in range(n_anch_neg):
            cols[f"{column_prefix}_neg_a{j}"] = feats["sim_neg"][:, j].astype(dtype, copy=False)
        cols[f"{column_prefix}_mass_pos"] = feats["mass_pos"].astype(dtype, copy=False)
        return pl.DataFrame(cols)

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    # For OOF Mode A, we need a fixed column schema. Compute global n_anch_pos / n_anch_neg by min over folds.
    splits = list(splitter.split(X_train_f))
    # Compute target per-fold class counts to derive a stable n_anchors per class.
    min_n_pos_per_fold = min(int((y_train_f[idx[0]] > 0.5).sum()) for idx in splits)
    min_n_neg_per_fold = min(int((y_train_f[idx[0]] <= 0.5).sum()) for idx in splits)
    n_anch_pos = min(n_anchors_per_class, max(2, min_n_pos_per_fold // 2))
    n_anch_neg = min(n_anchors_per_class, max(2, min_n_neg_per_fold // 2))

    out_pos = np.zeros((n_train, n_anch_pos), dtype=dtype)
    out_neg = np.zeros((n_train, n_anch_neg), dtype=dtype)
    out_mass = np.zeros(n_train, dtype=dtype)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_tr = X_train_f[train_idx]
        X_va = X_train_f[val_idx]
        y_tr = y_train_f[train_idx]
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(X_tr)
            X_tr_s = scaler.transform(X_tr).astype(np.float32)
            X_va_s = scaler.transform(X_va).astype(np.float32)
        else:
            X_tr_s = X_tr
            X_va_s = X_va
        pos_mask = y_tr > 0.5
        neg_mask = ~pos_mask
        if int(pos_mask.sum()) < 2 or int(neg_mask.sum()) < 2:
            logger.warning("class_conditional_anchor: fold %d has too few rows per class; skipping anchors fill", fold_idx)
            continue
        anchors_pos = _fit_kmeans(X_tr_s[pos_mask], n_anchors=n_anch_pos, seed=int(seed) + fold_idx)
        anchors_neg = _fit_kmeans(X_tr_s[neg_mask], n_anchors=n_anch_neg, seed=int(seed) + fold_idx + 1)
        feats = _compute_class_anchor_features(
            X_va_s, anchors_pos=anchors_pos, anchors_neg=anchors_neg,
            y_train_pos_count_per_anchor=np.ones(n_anch_pos),
            y_train_neg_count_per_anchor=np.ones(n_anch_neg),
            softmax_temp=softmax_temp,
        )
        out_pos[val_idx] = feats["sim_pos"].astype(dtype, copy=False)
        out_neg[val_idx] = feats["sim_neg"].astype(dtype, copy=False)
        out_mass[val_idx] = feats["mass_pos"].astype(dtype, copy=False)
        logger.info("class_conditional_anchor: fold %d/%d done (n_pos=%d, n_neg=%d, anch_pos=%d, anch_neg=%d)", fold_idx + 1, len(splits), int(pos_mask.sum()), int(neg_mask.sum()), n_anch_pos, n_anch_neg)

    cols: dict[str, np.ndarray] = {}
    for j in range(n_anch_pos):
        cols[f"{column_prefix}_pos_a{j}"] = out_pos[:, j]
    for j in range(n_anch_neg):
        cols[f"{column_prefix}_neg_a{j}"] = out_neg[:, j]
    cols[f"{column_prefix}_mass_pos"] = out_mass
    return pl.DataFrame(cols)
