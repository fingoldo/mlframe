"""RF/GBDT-proximity attention: use boosting's own leaf-indicator co-occurrence as the similarity metric for kNN target aggregation.

Iter 1 used boosting leaves as ORDINAL FEATURES (failed: -13% to -19% R² across all 3 boostings). Iter 17 uses the same leaves as a DISTANCE metric instead —
two rows are "close" if they end up in the same leaf in many of the auxiliary boosting's trees. The proximity is then used to weight a target aggregate, returning
a row-attention output that captures "what y looks like for rows that the auxiliary boosting treats similarly".

Why this is the correct way to use leaves: iter 1's ordinal-feature use competed with the downstream boosting's own partitioning. Leaf-as-distance instead INFORMS
the kNN target encoding — the auxiliary boosting provides a target-aware similarity metric that pure Euclidean / cosine cannot capture (since the auxiliary model
has learned which axes matter for y). The downstream boosting then sees ONLY the target aggregate (a single column per head), not the noisy ordinal-leaf encoding.

Mechanism:
1. Fit an auxiliary LGB on (X_train, y_train) with ``n_aux_trees`` trees and small ``max_depth`` (default 4 — controls leaf granularity).
2. For each row, get its leaf assignment per tree: ``L[i, t] ∈ {0, ..., n_leaves_t - 1}``.
3. One-hot encode leaves into a sparse high-dimensional embedding: ``S = (N, sum_t n_leaves_t)`` with 1 at column ``offset_t + L[i, t]`` per tree.
4. Compute approximate kNN in S-space using L2-normalised cosine similarity (each row has ``n_aux_trees`` non-zeros, so dot products are tractable).
5. Run softmax-weighted target aggregation on the kNN, same as row-attention stage-4.

Mode A (OOF): aux LGB refit per fold on ``X_train[train_idx]``, leaves derived from the train-fold's fitted model, val-fold rows queried against the train-fold's
leaf-embedding bank.

Mode B (X_query): aux LGB fit once on full X_train, query rows pushed through the fitted model to get leaf assignments, scored against the train bank.

Reference: random-forest proximity matrix (Breiman 2001) is the unsupervised version of this — we use a supervised gradient-boosted variant. The leaf-indicator
distance is also similar to TabNet's "supervised similarity" (Arik & Pfister 2019) but frozen and ~100x cheaper.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl
from scipy import sparse as sp

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _fit_aux_lgb(X: np.ndarray, y: np.ndarray, task: str, n_estimators: int, max_depth: int, seed: int):
    """Fit an auxiliary LightGBM model and return it with the n_estimators tree count actually used."""
    import lightgbm as lgb
    params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        num_leaves=2 ** max_depth,
        learning_rate=0.05,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        min_data_in_leaf=5,
    )
    if task == "regression":
        model = lgb.LGBMRegressor(**params)
    else:
        model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    return model


def _leaf_embedding(model, X: np.ndarray, num_leaves: int) -> sp.csr_matrix:
    """Return sparse one-hot leaf embedding of X under the fitted aux LGB.

    Shape: (n_rows, n_trees * num_leaves). Each row has exactly ``n_trees`` non-zeros (one per tree). Fixed feature dim across calls (uses the model's
    ``num_leaves`` config) so embeddings from different X (bank vs query) share the same column space.

    L2-normalised so that cosine similarity = (1/n_trees) * (number of trees where both rows are in same leaf) ∈ [0, 1].
    """
    leaves = model.predict(X, pred_leaf=True)  # (n_rows, n_trees) int
    n_rows, n_trees = leaves.shape
    # Fixed per-tree offsets — total_features = n_trees * num_leaves, regardless of which leaves are populated by this particular X.
    offsets = (np.arange(n_trees, dtype=np.int64) * num_leaves)
    total_features = int(n_trees * num_leaves)

    rows = np.repeat(np.arange(n_rows, dtype=np.int64), n_trees)
    cols = (offsets[None, :] + leaves).ravel().astype(np.int64)
    data = np.full(n_rows * n_trees, 1.0 / np.sqrt(n_trees), dtype=np.float32)
    S = sp.csr_matrix((data, (rows, cols)), shape=(n_rows, total_features), dtype=np.float32)
    return S


def _topk_proximity(S_query: sp.csr_matrix, S_bank: sp.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine similarity from each query row against the bank rows.

    Returns (topk_ids, topk_sims) shapes (n_query, k).

    Uses dense product of sparse matrices. For N_query=4000, N_bank=4000, n_features~3000 (300 trees × 10 leaves), dense N×N is 16M floats ~64MB, tractable.
    """
    # CPX38 FUTURE (bench: _benchmarks rf_proximity probe, n=4000/d=12/200 trees): the "dense
    # materialization of a sparse top-k" premise does NOT hold here — with ~200 aux trees every
    # query/bank pair shares >=1 leaf, so sim is 100% DENSE (measured density=1.000). The dense
    # block is load-bearing: the per-row argpartition below needs every column, and there is no
    # sparsity to exploit. argpartition is already in use and cheap (61-176ms vs the 0.9-3.0s sparse
    # matmul, which is the real cost and must produce all pairs for top-k regardless). A sparse@dense
    # BLAS variant was ~2x on the matmul but requires materializing the full dense bank
    # (n_bank*n_features) — a larger, unconditional dense allocation, out of CPX38's stated scope.
    # Dot product yields cosine similarity since both inputs are L2-normalised.
    sim_dense = (S_query @ S_bank.T).toarray()  # (n_query, n_bank) float32
    # Top-k per row.
    n_query = sim_dense.shape[0]
    n_bank = sim_dense.shape[1]
    if k >= n_bank:
        k = n_bank
    # argpartition for top-k indices (unsorted).
    part_idx = np.argpartition(-sim_dense, kth=k - 1, axis=1)[:, :k]  # (n_query, k)
    # Sort the top-k by similarity for stable downstream softmax. kind="stable" makes the neighbour order
    # deterministic among tied similarities (RF proximities are quantised co-occurrence ratios, so ties are
    # common); the softmax aggregate itself is order-invariant, but a stable order keeps the emitted
    # topk_ids reproducible.
    row_idx = np.arange(n_query)[:, None]
    part_sims = sim_dense[row_idx, part_idx]
    sort_idx = np.argsort(-part_sims, axis=1, kind="stable")
    topk_ids = part_idx[row_idx, sort_idx]
    topk_sims = part_sims[row_idx, sort_idx]
    return topk_ids.astype(np.int64), topk_sims.astype(np.float32)


def _softmax_aggregate(topk_ids: np.ndarray, topk_sims: np.ndarray, y_bank: np.ndarray, softmax_temp: float) -> tuple[np.ndarray, np.ndarray]:
    """Softmax over top-k similarities, then weighted y_mean and y_std."""
    logits = topk_sims / (softmax_temp + 1e-9)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    y_neighbors = y_bank[topk_ids]  # (n_query, k)
    y_mean = (weights * y_neighbors).sum(axis=1).astype(np.float32)
    y_var = (weights * (y_neighbors - y_mean[:, None]) ** 2).sum(axis=1)
    y_std = np.sqrt(y_var + 1e-9).astype(np.float32)
    return y_mean, y_std


def compute_rf_proximity_attention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["regression", "binary"] = "regression",
    n_aux_trees: int = 200,
    aux_max_depth: int = 4,
    k: int = 32,
    softmax_temp: float = 1.0,
    column_prefix: str = "rfprox",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """RF/GBDT-proximity attention features.

    Output columns: ``{column_prefix}_y_mean`` and ``{column_prefix}_y_std`` (2 columns total).

    Mode A: aux LGB refit per fold, val-fold rows queried against train-fold leaf bank.
    Mode B: aux LGB fit once on full X_train; X_query rows queried against the bank.

    For boostings: this gives them ONE extra feature ("what y looks like for rows the aux LGB groups with you") instead of iter-1's ~200 leaf-index columns. The
    single-feature form avoids competing with the downstream's own splits while still injecting target-aware similarity information.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if task not in ("regression", "binary"):
        raise ValueError(f"task must be 'regression' or 'binary'; got {task!r}.")
    if k < 2:
        raise ValueError(f"k must be >= 2; got {k}.")

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    if X_query is not None:
        X_query_f = np.asarray(X_query, dtype=np.float32)
        model = _fit_aux_lgb(X_train_f, y_train_f, task=task, n_estimators=n_aux_trees, max_depth=aux_max_depth, seed=seed)
        num_leaves = 2 ** aux_max_depth
        S_train = _leaf_embedding(model, X_train_f, num_leaves=num_leaves)
        S_query = _leaf_embedding(model, X_query_f, num_leaves=num_leaves)
        topk_ids, topk_sims = _topk_proximity(S_query, S_train, k=k)
        y_mean, y_std = _softmax_aggregate(topk_ids, topk_sims, y_bank=y_train_f, softmax_temp=softmax_temp)
        return pl.DataFrame({
            f"{column_prefix}_y_mean": y_mean.astype(dtype, copy=False),
            f"{column_prefix}_y_std": y_std.astype(dtype, copy=False),
        })

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    y_mean_out = np.zeros(n_train, dtype=dtype)
    y_std_out = np.zeros(n_train, dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_tr = X_train_f[train_idx]
        X_va = X_train_f[val_idx]
        y_tr = y_train_f[train_idx]
        model = _fit_aux_lgb(X_tr, y_tr, task=task, n_estimators=n_aux_trees, max_depth=aux_max_depth, seed=int(seed) + fold_idx)
        num_leaves = 2 ** aux_max_depth
        S_tr = _leaf_embedding(model, X_tr, num_leaves=num_leaves)
        S_va = _leaf_embedding(model, X_va, num_leaves=num_leaves)
        topk_ids, topk_sims = _topk_proximity(S_va, S_tr, k=k)
        y_mean_v, y_std_v = _softmax_aggregate(topk_ids, topk_sims, y_bank=y_tr, softmax_temp=softmax_temp)
        y_mean_out[val_idx] = y_mean_v.astype(dtype, copy=False)
        y_std_out[val_idx] = y_std_v.astype(dtype, copy=False)
        logger.info("rf_proximity: fold %d/%d done (n_train=%d, n_val=%d, trees=%d, depth=%d)", fold_idx + 1, len(splits), len(train_idx), len(val_idx), n_aux_trees, aux_max_depth)

    return pl.DataFrame({
        f"{column_prefix}_y_mean": y_mean_out,
        f"{column_prefix}_y_std": y_std_out,
    })
