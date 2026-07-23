"""Anchor-based attention: K-means anchors in X-space, per-row soft assignment + per-anchor target aggregation.

Differs from row-attention (per-query top-k over the full train pool): anchors are a FIXED small set of K centroids learned from X. Each row's features are
(a) softmax-weighted similarity to all K anchors, (b) the per-anchor target aggregates (y_mean / y_std). Output is (n_rows, K * (1 + n_aggregates)).

Why this complements row-attention:
- Row-attention is local-per-query (top-k nearest, can miss global structure when local neighbourhood is degenerate).
- Anchor-attention is GLOBAL: each row gets a fingerprint of where it sits relative to the K modes of the X-distribution.
- For datasets with a few latent clusters (different risk profiles in diabetes, different mass categories in mammography), anchor features expose cluster-membership
  directly. Boostings can then split on "row is close to anchor 7" without having to rediscover that anchor via repeated axis-aligned splits.

Leakage discipline:
- Mode A (X_query is None): K-means is refit per fold on X_train[train_idx]; per-anchor y aggregates computed from y_train[train_idx] only; val-fold rows scored
  against the fold's anchors (anchors NEVER see val-fold rows or their y).
- Mode B (X_query is given): K-means fit once on full X_train; aggregates from full y_train; query rows scored against fixed anchors. Train-only key-bank discipline.

Hyperparams:
- n_anchors: 16-64 typically (much smaller than k in row-attention since anchors are global).
- softmax_temp: controls hard-vs-soft assignment to anchors. Lower temp -> hard clustering features.
- aggregate: per-anchor y statistics (y_mean, y_std, count, y_q10, y_q90).

Reference: this is essentially "soft K-means target encoding" — equivalent to a single-layer attention over a learned codebook (the centroids), where the codebook
is trained UNSUPERVISED (K-means) but the values (per-anchor y stats) come from the targets. Closes a gap in the frozen-mechanism toolkit (no existing iter
implemented anchor-style features).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_VALID_AGGS = ("y_mean", "y_std", "count", "y_q10", "y_q90")


def _squared_dists(X: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Per-row squared euclidean distance to each anchor, (n_rows, n_anchors), via the
    ``||x||^2 - 2 x.a + ||a||^2`` GEMM decomposition. Avoids the (n_rows, n_anchors, d) broadcast
    cube that ``np.sum((X[:,None,:]-anchors[None,:,:])**2, axis=2)`` materialises; only the
    (n_rows, n_anchors) result is allocated. Differs from the subtraction form by float32 reduction
    order (~1e-5 on the downstream softmax, argmin-equivalent), selection-equivalent for these FE features."""
    x_sq = np.einsum("ij,ij->i", X, X)[:, None]
    a_sq = np.einsum("ij,ij->i", anchors, anchors)[None, :]
    d = x_sq - 2.0 * (X @ anchors.T) + a_sq
    np.maximum(d, 0.0, out=d)
    return np.asarray(d)


def _fit_anchors(X: np.ndarray, n_anchors: int, seed: int) -> np.ndarray:
    """Fit K-means anchors on X. Falls back to MiniBatchKMeans for large N."""
    from sklearn.cluster import KMeans, MiniBatchKMeans
    n_samples = X.shape[0]
    if n_samples > 20_000:
        km = MiniBatchKMeans(n_clusters=n_anchors, random_state=seed, n_init=3, batch_size=2048, max_iter=100)
    else:
        km = KMeans(n_clusters=n_anchors, random_state=seed, n_init=5, max_iter=100)
    km.fit(X)
    return np.asarray(km.cluster_centers_.astype(np.float32, copy=False))


def _compute_anchor_aggregates(y_train: np.ndarray, assignments: np.ndarray, n_anchors: int, aggregates: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Compute per-anchor target aggregates given hard assignment train_rows->nearest_anchor.

    Returns dict of arrays, each of shape (n_anchors,). Anchors with empty membership get y_mean=global_mean, y_std=global_std (degenerate-safe fallback).
    """
    global_mean = float(y_train.mean())
    global_std = float(y_train.std() + 1e-9)
    out: dict[str, np.ndarray] = {}
    for agg in aggregates:
        out[agg] = np.zeros(n_anchors, dtype=np.float32)
    for a in range(n_anchors):
        mask = assignments == a
        if not mask.any():
            for agg in aggregates:
                if agg == "y_mean":
                    out[agg][a] = global_mean
                elif agg == "y_std":
                    out[agg][a] = global_std
                elif agg == "count":
                    out[agg][a] = 0.0
                elif agg in ("y_q10", "y_q90"):
                    out[agg][a] = global_mean
            continue
        y_a = y_train[mask]
        for agg in aggregates:
            if agg == "y_mean":
                out[agg][a] = float(y_a.mean())
            elif agg == "y_std":
                out[agg][a] = float(y_a.std() + 1e-9)
            elif agg == "count":
                out[agg][a] = float(mask.sum())
            elif agg == "y_q10":
                out[agg][a] = float(np.quantile(y_a, 0.1))
            elif agg == "y_q90":
                out[agg][a] = float(np.quantile(y_a, 0.9))
    return out


def _score_rows_against_anchors(X: np.ndarray, anchors: np.ndarray, anchor_aggs: dict[str, np.ndarray], softmax_temp: float, n_anchors: int) -> dict[str, np.ndarray]:
    """For each row in X, compute softmax-weighted similarity to all anchors AND per-anchor target features (broadcast soft-assignment * aggregate)."""
    # Negative squared euclidean as similarity; softmax over anchors per row.
    dists = _squared_dists(X, anchors)  # (n, K)
    logits = -dists / (softmax_temp + 1e-9)
    logits -= logits.max(axis=1, keepdims=True)  # numerical stability
    exp_l = np.exp(logits)
    weights = exp_l / exp_l.sum(axis=1, keepdims=True)  # (n, K)

    out: dict[str, np.ndarray] = {"similarity": weights.astype(np.float32)}  # (n, K)
    for agg, agg_vec in anchor_aggs.items():
        # Each row's aggregate feature = sum over anchors of (weight * anchor's agg value).
        out[agg] = (weights @ agg_vec).astype(np.float32)  # (n,)
    return out


def compute_anchor_attention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    n_anchors: int = 32,
    softmax_temp: float = 1.0,
    aggregate: tuple[str, ...] = ("y_mean", "y_std"),
    standardize: bool = True,
    column_prefix: str = "anchor",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Anchor-based attention features.

    Mode A (X_query is None): OOF on X_train via ``splitter``. Anchors refit per-fold on X_train[train_idx]; aggregates from y_train[train_idx]; val rows scored.
    Mode B (X_query given): Anchors fit on full X_train; aggregates from full y_train; X_query rows scored. Splitter ignored.

    Output columns (per row): {n_anchors} similarity columns + {n_anchors * len(aggregate)} per-anchor aggregate columns.
    Total = n_anchors * (1 + len(aggregate)) features.

    For n_anchors=32, aggregate=('y_mean','y_std'): 32 + 64 = 96 columns. Roughly comparable in width to a 4-head row-attention with k=32.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if not aggregate:
        raise ValueError("aggregate must be a non-empty tuple.")
    for agg in aggregate:
        if agg not in _VALID_AGGS:
            raise ValueError(f"aggregate item {agg!r} not in {_VALID_AGGS}.")
    if n_anchors < 2:
        raise ValueError(f"n_anchors must be >= 2; got {n_anchors}.")
    if softmax_temp <= 0:
        raise ValueError(f"softmax_temp must be > 0; got {softmax_temp}.")

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    if standardize:
        from sklearn.preprocessing import RobustScaler
        scaler_full = RobustScaler().fit(X_train_f)
    else:
        scaler_full = None

    def _build_output(scores_per_row: dict[str, np.ndarray], n_rows: int) -> dict[str, np.ndarray]:
        """Slice the per-anchor similarity matrix and per-row soft-aggregate scores into a name-tagged output dict (``{prefix}_sim_a{k}`` per anchor, ``{prefix}_{agg}_soft`` per aggregate), cast to the requested output dtype."""
        out: dict[str, np.ndarray] = {}
        sim = scores_per_row["similarity"]  # (n_rows, K)
        for a in range(n_anchors):
            out[f"{column_prefix}_sim_a{a}"] = sim[:, a].astype(dtype, copy=False)
        for agg in aggregate:
            agg_vec = scores_per_row[agg]  # (n_rows,)
            out[f"{column_prefix}_{agg}_soft"] = agg_vec.astype(dtype, copy=False)
        return out

    if X_query is not None:
        # Mode B: fit once on full X_train.
        Xt_std = scaler_full.transform(X_train_f).astype(np.float32) if standardize else X_train_f
        Xq_std = scaler_full.transform(np.asarray(X_query, dtype=np.float32)).astype(np.float32) if standardize else np.asarray(X_query, dtype=np.float32)
        anchors = _fit_anchors(Xt_std, n_anchors=n_anchors, seed=seed)
        # Hard assignment of train rows to nearest anchor for aggregate computation.
        train_dists = _squared_dists(Xt_std, anchors)
        # Wave 21 P1: np.argmin returns 0 on all-NaN rows (numpy>=1.18),
        # silently bucketing NaN-bearing rows under anchor 0 and
        # contaminating anchor 0's per-row aggregate. Use np.nanargmin,
        # which raises only when an entire row is all-NaN; that case is
        # data corruption deserving a loud error rather than a silent
        # bucket-to-0.
        train_assign = np.nanargmin(train_dists, axis=1)
        anchor_aggs = _compute_anchor_aggregates(y_train_f, train_assign, n_anchors=n_anchors, aggregates=aggregate)
        scores = _score_rows_against_anchors(Xq_std, anchors, anchor_aggs, softmax_temp=softmax_temp, n_anchors=n_anchors)
        cols = _build_output(scores, n_rows=Xq_std.shape[0])
        return pl.DataFrame(cols)

    # Mode A: OOF per-fold.
    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out_cols: dict[str, np.ndarray] = {}
    for a in range(n_anchors):
        out_cols[f"{column_prefix}_sim_a{a}"] = np.zeros(n_train, dtype=dtype)
    for agg in aggregate:
        out_cols[f"{column_prefix}_{agg}_soft"] = np.zeros(n_train, dtype=dtype)

    splits = list(splitter.split(X_train_f))
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
        anchors = _fit_anchors(X_tr_s, n_anchors=n_anchors, seed=int(seed) + fold_idx)
        train_dists = _squared_dists(X_tr_s, anchors)
        # Same NaN-poisoned-row bucket-to-0 bug Mode B's np.nanargmin above already guards against --
        # plain np.argmin returns 0 on an all-NaN row, silently contaminating anchor 0's aggregate.
        train_assign = np.nanargmin(train_dists, axis=1)
        anchor_aggs = _compute_anchor_aggregates(y_tr, train_assign, n_anchors=n_anchors, aggregates=aggregate)
        scores = _score_rows_against_anchors(X_va_s, anchors, anchor_aggs, softmax_temp=softmax_temp, n_anchors=n_anchors)
        sim = scores["similarity"]
        for a in range(n_anchors):
            out_cols[f"{column_prefix}_sim_a{a}"][val_idx] = sim[:, a].astype(dtype, copy=False)
        for agg in aggregate:
            out_cols[f"{column_prefix}_{agg}_soft"][val_idx] = scores[agg].astype(dtype, copy=False)
        logger.info("anchor_attention: fold %d/%d done (n_train=%d, n_val=%d, n_anchors=%d)", fold_idx + 1, len(splits), len(train_idx), len(val_idx), n_anchors)

    return pl.DataFrame(out_cols)
