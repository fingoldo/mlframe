"""Local lift / local-PR_AUC features: per-row local-rank-quality vs global, designed to break the CB-on-mammography ceiling.

Iter 24 mechanism. Motivation: CatBoost's internal ordered target statistics compute ``E[y | feature]`` per column, but they CANNOT compute *local-rank-quality of
neighbours*: the rate at which positive-class examples concentrate near this row's neighbourhood, normalised by the global base rate. For heavily-imbalanced
binary (mammography 1.3% positive), the global base rate is ~0.013 and the *local* positive rate around a true-positive row can be 0.10-0.30 — a 7-23× lift.
Boostings on top of these features can split on "row is in a 20× local-positive-density region".

Three frozen features per row:
1. ``local_lift`` = mean(y_kNN(i)) / global_pos_rate. Encoding strength of positive concentration locally.
2. ``local_pr_auc`` = local Average Precision (sklearn AP / step-PR convention, NOT trapezoidal) over the sorted (distance, y) sequence of top-k neighbours. Captures the ranking quality of distance-as-classifier.
3. ``local_top1_y`` = y of the closest neighbour (sharp marker; signal-rich for nearest-neighbour structure).

For regression target: ``local_lift`` becomes ``mean(y_kNN) / mean(y_global)`` (so values >1 mean above-average, <1 below); ``local_pr_auc`` becomes a Spearman-rank
correlation between distance and y over the top-k.

Leakage discipline: kNN built on standardised X_train[train_idx] per fold; val rows queried.

References:
- Local-density-ratio scoring (Sugiyama & Suzuki 2012) is conceptually similar.
- Per-row LOF/density features (Breunig 2000) — LOF measures rank-quality of distance as outlier-classifier.
- Local-AUC (McClish 1989) — region-restricted AUC of a classifier.

Cost: O(N · k · log k) per fold for kNN + per-query sort. ms-scale at N<10k.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._intel_patch import try_patch_sklearn
from ._knn_helper import knn_search
from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _local_lift_and_pr_auc(y_neighbors: np.ndarray, dists: np.ndarray, y_global_mean: float, task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-query: local lift, local Average Precision (or Spearman for regression), local top-1 y.

    y_neighbors: (n_q, k) — sorted by ascending distance (closest first).
    dists: (n_q, k) — corresponding distances.
    """
    n_q, k = y_neighbors.shape
    # Local mean of y.
    local_mean = y_neighbors.mean(axis=1)
    # Lift: ratio to global; safe-guarded for zero global mean.
    lift = local_mean / max(y_global_mean, 1e-9)

    # Local top-1 y (closest neighbour's target).
    top1 = y_neighbors[:, 0]

    if task == "binary":
        # Local PR_AUC: treat distance as a "negative score" (closer = higher score). Sort by ascending dist (already done), label = y.
        # PR_AUC = area under precision-recall curve where positives ranked by similarity (1/dist).
        # precision_i = cumulative_positives / (i+1), recall_i = cumulative_positives / total_positives (step-sum AP below, NOT trapezoidal -- see that comment).
        cum_pos = np.cumsum(y_neighbors, axis=1)
        total_pos = cum_pos[:, -1]  # may be 0 for some queries
        i_arange = (np.arange(k) + 1).astype(np.float32)
        precision = cum_pos / i_arange[None, :]
        recall = np.where(total_pos[:, None] > 0, cum_pos / np.maximum(total_pos[:, None], 1e-9), 0.0)
        # Local AVERAGE PRECISION (sklearn ``average_precision_score`` convention): AP = sum_n (R_n - R_{n-1}) * P_n -- a step / left-Riemann sum, deliberately NOT trapezoidal (trapezoidal interpolation between PR points is optimistic and is the reason sklearn dropped ``auc(recall, precision)``). The neighbours are already ranked by ascending distance == descending similarity, so this is exactly AP over the local neighbourhood.
        d_recall = np.diff(recall, axis=1, prepend=0.0)
        pr_auc = (d_recall * precision).sum(axis=1)
        # Queries with no positives in their neighborhood get pr_auc = 0; mark with local_mean (sentinel: zero local mean correlates with zero PR_AUC).
        pr_auc = np.where(total_pos > 0, pr_auc, 0.0)
    else:
        # Regression: Spearman-rank correlation of (-distance) with y over kNN.
        # Convert each row to ranks then compute Pearson.
        rng_idx = np.arange(k)
        dist_ranks = np.tile(rng_idx, (n_q, 1)).astype(np.float32)  # already sorted ascending dist, so dist ranks are 0..k-1
        # y ranks within each row: single argsort + scatter instead of double argsort (bit-identical, ~1.7-
        # 1.9x faster -- the second argsort was pure waste, the inverse permutation of the first argsort IS
        # the rank vector).
        _order = np.argsort(y_neighbors, axis=1)
        y_ranks = np.empty_like(_order)
        np.put_along_axis(y_ranks, _order, np.broadcast_to(np.arange(k), _order.shape), axis=1)
        y_ranks = y_ranks.astype(np.float32)
        # Pearson of (-dist_ranks, y_ranks) = - Pearson(dist_ranks, y_ranks). Higher dist_rank = farther; negate to make "similarity score".
        sim_ranks = (k - 1) - dist_ranks
        a = sim_ranks - sim_ranks.mean(axis=1, keepdims=True)
        b = y_ranks - y_ranks.mean(axis=1, keepdims=True)
        num = (a * b).sum(axis=1)
        denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1) + 1e-9)
        pr_auc = num / denom  # Spearman correlation in [-1, 1]

    return lift.astype(np.float32), pr_auc.astype(np.float32), top1.astype(np.float32)


def compute_local_lift_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    k: int = 32,
    standardize: bool = True,
    column_prefix: str = "loclift",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Local lift / PR_AUC / top1-y features.

    Output: 3 columns per row — ``{prefix}_lift``, ``{prefix}_pr_auc`` (or Spearman for regression), ``{prefix}_top1``.
    """
    seed = require_seed(seed)
    try_patch_sklearn()
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if task not in ("binary", "regression"):
        raise ValueError(f"task must be 'binary' or 'regression'; got {task!r}.")
    if k < 4:
        raise ValueError(f"k must be >= 4; got {k}.")

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    y_global_mean = float(y_train_f.mean()) if y_train_f.size > 0 else 0.0

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray, global_mean: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standardize (optional) and kNN-search ``Xq`` against ``Xt``, then reduce each query's neighbor labels to (lift, pr_auc/spearman, top1) via ``_local_lift_and_pr_auc``. Shared by both the train-only (self-kNN) and train+query call sites below."""
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        # _knn_helper auto-dispatches to hnswlib at N>=50000, falls back to sklearn otherwise.
        dists, ids = knn_search(Xt_s, Xq_s, k=k)
        y_neighbors = y_t[ids]  # (n_q, k_used)
        return _local_lift_and_pr_auc(y_neighbors, dists, global_mean, task=task)

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        lift, pr_auc, top1 = _process(X_train_f, Xq, y_train_f, y_global_mean)
        return pl.DataFrame({
            f"{column_prefix}_lift": lift.astype(dtype, copy=False),
            f"{column_prefix}_pr_auc": pr_auc.astype(dtype, copy=False),
            f"{column_prefix}_top1": top1.astype(dtype, copy=False),
        })

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out_lift: np.ndarray = np.zeros(n_train, dtype=dtype)
    out_pr: np.ndarray = np.zeros(n_train, dtype=dtype)
    out_top1: np.ndarray = np.zeros(n_train, dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_tr = X_train_f[train_idx]
        X_va = X_train_f[val_idx]
        y_tr = y_train_f[train_idx]
        fold_global_mean = float(y_tr.mean()) if y_tr.size > 0 else 0.0
        lift, pr_auc, top1 = _process(X_tr, X_va, y_tr, fold_global_mean)
        out_lift[val_idx] = lift.astype(dtype, copy=False)
        out_pr[val_idx] = pr_auc.astype(dtype, copy=False)
        out_top1[val_idx] = top1.astype(dtype, copy=False)
        logger.info("local_lift: fold %d/%d done (n_train=%d, n_val=%d, k=%d)", fold_idx + 1, len(splits), len(train_idx), len(val_idx), k)

    return pl.DataFrame({
        f"{column_prefix}_lift": out_lift,
        f"{column_prefix}_pr_auc": out_pr,
        f"{column_prefix}_top1": out_top1,
    })
