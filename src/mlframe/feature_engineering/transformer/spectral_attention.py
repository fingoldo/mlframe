"""Spectral attention: kNN-graph Laplacian eigenvectors as global manifold-mode features.

Iter 17 (RF-proximity) and earlier row-attention iters all operate LOCALLY — each query row sees its top-k neighbours and aggregates within them. They miss GLOBAL
manifold structure: rows can be connected through long chains of neighbours that no single top-k query captures.

Spectral attention captures global structure via the standard manifold-learning route: build a kNN graph on X, compute its normalised graph Laplacian
``L = I - D^{-1/2} W D^{-1/2}``, take the bottom-k non-trivial eigenvectors. Each eigenvector is a "slow mode" of the data — values change smoothly along the
manifold. Used directly as features, they expose global cluster / manifold orientation that local attention cannot derive.

Mechanism:
1. Build kNN graph G on X_train (k_graph neighbours per row, default 10) with Gaussian-weighted edges ``exp(-d^2 / 2σ^2)``.
2. Compute symmetric normalised Laplacian ``L_sym = I - D^{-1/2} G D^{-1/2}``.
3. Eigendecompose for the bottom ``n_eigvecs + 1`` eigenvalues (skip the trivial constant eigenvector at λ=0).
4. The next ``n_eigvecs`` eigenvectors are the "spectral coordinates" — output as features.

For OUT-OF-SAMPLE query rows (Mode B / Mode A val-fold):
- Use Nyström extension: for each query, compute its kNN affinity into the train bank, then project = (1/λ_j) * Σ_i W_{query,i} * φ_j(i) where φ_j is the j-th
  train eigenvector. This extends discrete eigenvectors to continuous query points.

Mode A discipline: graph + eigenvectors refit per fold on X_train[train_idx]; val-fold rows projected via Nyström. y_train is NOT used in eigenvector computation
(unsupervised), so there is no target-leakage concern — the projection is target-free, the only leakage risk would be passing val-fold X into the eigendecomposition,
which Mode A's per-fold refit prevents.

Why this complements existing iters:
- Row-attention with PLS/importance projection: target-aware but LOCAL.
- Spectral: target-free but GLOBAL.
- Combined (spectral + row-attention or spectral + RFF): expose both global manifold orientation AND local target structure.

Reference: spectral embedding / Laplacian eigenmaps (Belkin-Niyogi 2003), diffusion maps (Coifman 2006). Standard manifold-learning technique re-purposed as a
frozen feature-engineering step for boostings.

Cost: ~O(N * k_graph) for graph construction, O(N * n_eigvecs) for Lanczos eigendecomposition (sparse). At N=4000, total ~3-5 seconds.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import polars as pl
from scipy import sparse as sp

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _build_knn_graph(X: np.ndarray, k_graph: int, sigma: Optional[float] = None) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Build symmetric kNN graph with Gaussian-weighted edges.

    Returns (W, knn_ids, knn_dists) where W is a (N, N) sparse affinity matrix, knn_ids and knn_dists are (N, k_graph) for Nyström extension.

    Sigma defaults to median pairwise kNN distance (data-driven scale).
    """
    from sklearn.neighbors import NearestNeighbors
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k_graph + 1, algorithm="auto", n_jobs=-1).fit(X)
    dists, ids = nn.kneighbors(X)
    dists = dists[:, 1:]  # drop self
    ids = ids[:, 1:]

    if sigma is None or sigma <= 0:
        # Median of the k-th NN distance — Belkin-Niyogi heuristic.
        sigma = float(np.median(dists[:, -1])) + 1e-9

    weights = np.exp(-(dists**2) / (2 * sigma**2)).astype(np.float32)
    # Build (N, N) sparse — symmetrise.
    rows = np.repeat(np.arange(n, dtype=np.int64), k_graph)
    cols = ids.ravel().astype(np.int64)
    data = weights.ravel()
    W = sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    W = W.maximum(W.T)  # symmetric kNN graph
    return W, ids, dists


def _eigvecs_from_graph(W: sp.csr_matrix, n_eigvecs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute bottom (n_eigvecs+1) eigenvectors of the symmetric normalised Laplacian; drop the trivial constant eigvec.

    Returns (eigvals, eigvecs) — eigvals shape (n_eigvecs,), eigvecs shape (N, n_eigvecs).
    """
    from scipy.sparse.linalg import eigsh
    n = W.shape[0]
    d = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
    d_inv_sqrt = 1.0 / np.sqrt(d + 1e-9)
    # L_sym = I - D^{-1/2} W D^{-1/2}; eigendecompose A = D^{-1/2} W D^{-1/2} (LARGEST eigvals of A == SMALLEST of L_sym).
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    A = D_inv_sqrt @ W @ D_inv_sqrt
    A = (A + A.T) * 0.5  # numerical symmetrisation
    k_needed = min(n_eigvecs + 1, n - 1)
    try:
        eigvals_A, eigvecs_A = eigsh(A, k=k_needed, which="LM")  # largest of A
    except Exception:
        # Fall back to dense if Lanczos fails (small N or pathological structure).
        A_dense = A.toarray().astype(np.float64)
        full_eigvals, full_eigvecs = np.linalg.eigh(A_dense)
        eigvals_A = full_eigvals[-k_needed:]
        eigvecs_A = full_eigvecs[:, -k_needed:]
    # Sort descending (largest eigval of A = smallest of L = trivial first). kind="stable" gives a
    # deterministic index tiebreak for DEGENERATE (tied) eigenvalues -- common for graph Laplacians -- so
    # which eigenvector becomes feature-k does not drift on tied eigenvalues.
    order = np.argsort(-eigvals_A, kind="stable")
    eigvals_A = eigvals_A[order]
    eigvecs_A = eigvecs_A[:, order]
    # Skip the trivial constant eigvec (corresponds to eigval ~ 1 of A).
    return eigvals_A[1 : n_eigvecs + 1].astype(np.float32), eigvecs_A[:, 1 : n_eigvecs + 1].astype(np.float32), d_inv_sqrt


def _nystrom_extend(X_train: np.ndarray, X_query: np.ndarray, eigvals: np.ndarray, eigvecs_normalised: np.ndarray, d_inv_sqrt: np.ndarray, k_graph: int, sigma: Optional[float] = None) -> np.ndarray:
    """Project X_query rows onto the train eigenvectors via Nyström extension.

    For each query q, compute Gaussian-weighted affinity to its k_graph nearest train rows, then:
        φ_j(q) = (1 / λ_j) * Σ_i W(q, i) * d_i^{-1/2} * eigvec_j[i] * d_q^{-1/2}

    Returns (n_query, n_eigvecs) embedding.
    """
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k_graph, algorithm="auto", n_jobs=-1).fit(X_train)
    dists, ids = nn.kneighbors(X_query)
    if sigma is None or sigma <= 0:
        sigma = float(np.median(dists[:, -1])) + 1e-9
    affinity = np.exp(-(dists**2) / (2 * sigma**2)).astype(np.float32)  # (n_q, k_graph)
    d_q = affinity.sum(axis=1) + 1e-9
    d_q_inv_sqrt = 1.0 / np.sqrt(d_q)

    # Form a (n_q, n_train) sparse weight matrix (each row has k_graph entries) for fast multiply.
    n_q = X_query.shape[0]
    rows = np.repeat(np.arange(n_q, dtype=np.int64), k_graph)
    cols = ids.ravel().astype(np.int64)
    data = (affinity * d_inv_sqrt[ids]).ravel()  # apply d_i^{-1/2} per neighbour
    W_q = sp.csr_matrix((data, (rows, cols)), shape=(n_q, X_train.shape[0]), dtype=np.float32)

    raw = W_q @ eigvecs_normalised  # (n_q, n_eigvecs) — Σ_i affinity * d_i^{-1/2} * eigvec_j[i]
    # Apply d_q^{-1/2} per query and divide by eigvalue.
    embedding = raw * d_q_inv_sqrt[:, None]
    eigvals_safe = np.maximum(eigvals, 1e-6)
    embedding = embedding / eigvals_safe[None, :]
    return embedding.astype(np.float32)


def compute_spectral_attention(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    n_eigvecs: int = 8,
    k_graph: int = 10,
    standardize: bool = True,
    column_prefix: str = "spec",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Spectral attention: Laplacian eigenvectors of kNN graph as global manifold features.

    Output: ``n_eigvecs`` columns named ``{column_prefix}_e0 ... {column_prefix}_e{n_eigvecs-1}``.

    Note: y_train is NOT used in the spectral decomposition (unsupervised), but passed for API symmetry with other transformer-FE functions. Mode A still refits
    per fold to avoid passing val-fold X into the train eigendecomposition (which would leak X-distribution information across the split).
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if n_eigvecs < 1:
        raise ValueError(f"n_eigvecs must be >= 1; got {n_eigvecs}.")
    if k_graph < 2:
        raise ValueError(f"k_graph must be >= 2; got {k_graph}.")

    X_train_f = np.asarray(X_train, dtype=np.float32)

    def _process_train_query(Xt: np.ndarray, Xq: np.ndarray) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        W, _ids, _dists = _build_knn_graph(Xt_s, k_graph=k_graph)
        eigvals, eigvecs, d_inv_sqrt = _eigvecs_from_graph(W, n_eigvecs=n_eigvecs)
        eigvecs_normalised = eigvecs * d_inv_sqrt[:, None]
        embedding = _nystrom_extend(Xt_s, Xq_s, eigvals=eigvals, eigvecs_normalised=eigvecs_normalised, d_inv_sqrt=d_inv_sqrt, k_graph=k_graph)
        return embedding

    if X_query is not None:
        X_query_f = np.asarray(X_query, dtype=np.float32)
        embedding = _process_train_query(X_train_f, X_query_f)
        cols = {f"{column_prefix}_e{j}": embedding[:, j].astype(dtype, copy=False) for j in range(n_eigvecs)}
        return pl.DataFrame(cols)

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, n_eigvecs), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_tr = X_train_f[train_idx]
        X_va = X_train_f[val_idx]
        embedding = _process_train_query(X_tr, X_va)
        out[val_idx] = embedding.astype(dtype, copy=False)
        logger.info("spectral_attention: fold %d/%d done (n_train=%d, n_val=%d, n_eigvecs=%d, k_graph=%d)", fold_idx + 1, len(splits), len(train_idx), len(val_idx), n_eigvecs, k_graph)

    cols = {f"{column_prefix}_e{j}": out[:, j] for j in range(n_eigvecs)}
    return pl.DataFrame(cols)
