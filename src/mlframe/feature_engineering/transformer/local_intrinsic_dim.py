"""Local intrinsic dimension via PCA spectrum on local kNN neighborhoods.

Iter 74 mechanism. Geometric agent's #1 ranked.

For each row, take K=30 nearest neighbors in standardized X-space; compute PCA eigenvalues of the
neighbor-deviation matrix; emit spectrum-derived geometric invariants:
- participation_ratio = (sum λ)² / sum(λ²)        — effective # of components in spectrum
- top1_ratio = λ_1 / sum(λ)                       — fraction of variance in top eigenvalue (anisotropy)
- top2_ratio = λ_2 / λ_1                          — secondary direction's strength
- spectrum_entropy = -sum(p log p) for p = λ/sum(λ) — Shannon entropy of normalized spectrum
- effective_dim = exp(spectrum_entropy)            — exponential of entropy (intuitive scalar)

5 features per row, NO baseline, NO y. Pure input-manifold geometry.

Why structurally novel vs 72 existing:
- Iter 72 used kNN distance for density (scalar p̂(x))
- Iter 74 uses kNN spectrum (full eigendistribution) → captures MANIFOLD SHAPE at each row
- Rows in thin sheets / curves have low effective_dim; rows in isotropic clouds have high.

Iter 72 already proved input-geometry signal works on abalone (+1.15pp LGB R² record). Iter 74 extracts
a different invariant — shape rather than scale — from the same neighborhood.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def compute_local_intrinsic_dim_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "regression",
    k_neighbors: int = 30,
    standardize: bool = True,
    column_prefix: str = "lid",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Local intrinsic dimension features via PCA spectrum on kNN neighborhood.

    Output: 5 features per row.
    """
    from sklearn.neighbors import NearestNeighbors

    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()
    n_features = 5

    def _process(Xt: np.ndarray, Xq: np.ndarray) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        k_eff = min(k_neighbors, Xt_s.shape[0])
        nn = NearestNeighbors(n_neighbors=k_eff, n_jobs=-1).fit(Xt_s)
        _, idx = nn.kneighbors(Xq_s)  # (n_q, k_eff)
        neighbor_X = Xt_s[idx]  # (n_q, k_eff, d)
        # Center neighbor matrix on query row.
        deviations = neighbor_X - Xq_s[:, None, :]  # (n_q, k_eff, d)
        # Per-row covariance stack (n_q, d, d) built in one batched matmul (BLAS gemm
        # batched path), then a single batched eigvalsh over the leading axis -- both run
        # in C with no Python per-row frame. The spectrum math is vectorized over rows.
        # Equivalent to the prior `for q in range(n_q)` loop (same LAPACK eigvalsh per
        # slice); float32 identity within ~1e-6 abs / ~1e-7 rel. ~7.3x @ d=8, ~1.3x @ d=50.
        n_q, _, d = deviations.shape
        cov = np.matmul(deviations.transpose(0, 2, 1), deviations) / np.float32(k_eff)  # (n_q, d, d)
        lambdas = np.linalg.eigvalsh(cov)  # (n_q, d) ascending
        lambdas = np.clip(lambdas, 0.0, None) + 1e-9
        sum_l = lambdas.sum(axis=1)                # (n_q,)
        sum_l_sq = (lambdas ** 2).sum(axis=1)      # (n_q,)
        out = np.empty((n_q, n_features), dtype=np.float32)
        out[:, 0] = (sum_l * sum_l) / sum_l_sq     # participation_ratio
        top1 = lambdas[:, -1]
        top2 = lambdas[:, -2] if d >= 2 else np.full(n_q, 1e-9, dtype=lambdas.dtype)
        out[:, 1] = top1 / sum_l                   # top1_ratio
        out[:, 2] = top2 / top1                    # top2_ratio
        p = lambdas / sum_l[:, None]
        spectrum_entropy = -np.sum(p * np.log(p + 1e-9), axis=1)
        out[:, 3] = spectrum_entropy
        out[:, 4] = np.exp(spectrum_entropy)       # effective_dim
        return out

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        cols: dict[str, np.ndarray] = {}
        cols[f"{column_prefix}_participation_ratio"] = feats[:, 0].astype(dtype, copy=False)
        cols[f"{column_prefix}_top1_ratio"] = feats[:, 1].astype(dtype, copy=False)
        cols[f"{column_prefix}_top2_ratio"] = feats[:, 2].astype(dtype, copy=False)
        cols[f"{column_prefix}_spectrum_entropy"] = feats[:, 3].astype(dtype, copy=False)
        cols[f"{column_prefix}_effective_dim"] = feats[:, 4].astype(dtype, copy=False)
        return cols

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, n_features), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx])
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("local_intrinsic_dim: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
