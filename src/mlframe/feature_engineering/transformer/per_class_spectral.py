"""Per-class spectral attention: Laplacian eigenvectors of kNN-graph built SEPARATELY on positive and negative class subgraphs (binary classification only).

Iter 18 (`compute_spectral_attention`) builds one kNN graph on all of X and extracts global Laplacian eigenvectors — captures global manifold modes but, on
heavily-imbalanced binary (mammography ~1.3% positive), the dominant modes are driven by the majority class and the positive cluster geometry is invisible to the
eigendecomposition (52 positives out of 4000 contribute negligible Laplacian mass).

Iter 21 fixes this by computing TWO graphs (one per class) and TWO sets of eigenvectors (one per class). Each row gets a feature vector of "where does it sit in
the positive-class manifold's spectral coordinates" PLUS "where in the negative-class manifold's spectral coordinates". For rare positives, the positive-class
subgraph has very different geometry from the full graph and exposes structure that whole-graph spectral cannot.

Out-of-sample query rows are projected via TWO Nyström extensions (one against the positive subgraph's eigenvectors, one against the negative subgraph's).

Mode A: per fold, two K-means-style class subsets refit → two graphs → two eigendecomps → val rows projected via Nyström into both spectral coordinate systems.

Cost: 2× iter-18's cost; still well within budget at N<10k.

Reference: class-conditional spectral clustering (Mavroforakis et al. 2017); supervised Laplacian eigenmaps with side information.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input
from .spectral_attention import _build_knn_graph, _eigvecs_from_graph, _nystrom_extend

logger = logging.getLogger(__name__)


def _compute_spectral_for_subset(
    X_subset_std: np.ndarray,
    X_query_std: np.ndarray,
    n_eigvecs: int,
    k_graph: int,
) -> np.ndarray:
    """Build kNN graph on a class-restricted X_subset, eigendecompose, Nyström-project X_query."""
    n_sub = X_subset_std.shape[0]
    if n_sub < k_graph + 2:
        # Too few rows; return zeros (degenerate).
        return np.zeros((X_query_std.shape[0], n_eigvecs), dtype=np.float32)
    k_used = min(k_graph, n_sub - 1)
    W, _ids, _dists = _build_knn_graph(X_subset_std, k_graph=k_used)
    eigvals, eigvecs, d_inv_sqrt = _eigvecs_from_graph(W, n_eigvecs=min(n_eigvecs, n_sub - 2))
    # Pad eigvals/eigvecs if subset returned fewer than n_eigvecs.
    actual_e = eigvecs.shape[1]
    eigvecs_normalised = eigvecs * d_inv_sqrt[:, None]
    embedding = _nystrom_extend(X_subset_std, X_query_std, eigvals=eigvals, eigvecs_normalised=eigvecs_normalised, d_inv_sqrt=d_inv_sqrt, k_graph=k_used)
    if actual_e < n_eigvecs:
        padded = np.zeros((embedding.shape[0], n_eigvecs), dtype=np.float32)
        padded[:, :actual_e] = embedding
        embedding = padded
    return embedding


def compute_per_class_spectral_attention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    n_eigvecs_per_class: int = 4,
    k_graph: int = 10,
    standardize: bool = True,
    column_prefix: str = "pcspec",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Per-class spectral attention (binary classification only).

    Output: ``2 * n_eigvecs_per_class`` columns — half for positive-class subgraph eigvecs, half for negative-class subgraph eigvecs.

    Mode A: per fold, refit class subsets on train-fold; project val-fold rows via Nyström.
    Mode B: fit once on full X_train; project X_query via Nyström.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if n_eigvecs_per_class < 1:
        raise ValueError(f"n_eigvecs_per_class must be >= 1; got {n_eigvecs_per_class}.")

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        pos_mask = y_t > 0.5
        neg_mask = ~pos_mask
        emb_pos = _compute_spectral_for_subset(Xt_s[pos_mask], Xq_s, n_eigvecs=n_eigvecs_per_class, k_graph=k_graph)
        emb_neg = _compute_spectral_for_subset(Xt_s[neg_mask], Xq_s, n_eigvecs=n_eigvecs_per_class, k_graph=k_graph)
        return np.concatenate([emb_pos, emb_neg], axis=1)  # (n_q, 2*n_eigvecs_per_class)

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        emb = _process(X_train_f, Xq, y_train_f)
        cols = {}
        for j in range(n_eigvecs_per_class):
            cols[f"{column_prefix}_pos_e{j}"] = emb[:, j].astype(dtype, copy=False)
        for j in range(n_eigvecs_per_class):
            cols[f"{column_prefix}_neg_e{j}"] = emb[:, n_eigvecs_per_class + j].astype(dtype, copy=False)
        return pl.DataFrame(cols)

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, 2 * n_eigvecs_per_class), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        emb = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx])
        out[val_idx] = emb.astype(dtype, copy=False)
        logger.info("per_class_spectral: fold %d/%d done (n_pos=%d, n_neg=%d)", fold_idx + 1, len(splits), int((y_train_f[train_idx] > 0.5).sum()), int((y_train_f[train_idx] <= 0.5).sum()))

    cols = {}
    for j in range(n_eigvecs_per_class):
        cols[f"{column_prefix}_pos_e{j}"] = out[:, j]
    for j in range(n_eigvecs_per_class):
        cols[f"{column_prefix}_neg_e{j}"] = out[:, n_eigvecs_per_class + j]
    return pl.DataFrame(cols)
