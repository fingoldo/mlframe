"""Class-conditional Mahalanobis distance features for binary classification.

Iter 25 mechanism. Targets the mammography-CB ceiling. CatBoost cannot invert covariance internally (its symmetric oblivious trees + ordered TS encoding both operate
per-column; Mahalanobis distance requires the inverse covariance ``Σ⁻¹`` which is a global quadratic-form feature CB structurally cannot compute).

For each row, expose three Mahalanobis-style features:
1. ``m_pos`` = ``(x - μ_+)ᵀ Σ_+⁻¹ (x - μ_+)`` — Mahalanobis distance to the positive-class centroid with positive-class covariance.
2. ``m_neg`` = ``(x - μ_-)ᵀ Σ_-⁻¹ (x - μ_-)`` — same for negative class.
3. ``m_gap`` = ``m_neg - m_pos`` — signed gap. Bayes-rule classifier under multivariate Gaussian class-conditionals: positive iff m_gap > log(p_neg / p_pos) + log det ratio.

Covariance shrinkage (Ledoit-Wolf): for n=4000 mammography with ~52 positives, the empirical positive-class covariance is rank-deficient — Σ_+ has at most 51 non-trivial
eigenvalues for 6 features (well-conditioned in this case) but for higher-dim or smaller-N data the Ledoit-Wolf optimal shrinkage prevents singular inversion.

Leakage discipline: per-class μ and Σ refit per fold on train-fold subsets only.

Why CB struggles without this: CB's TS gives ``E[y | feature_j]`` per column, but a quadratic form ``(x - μ)ᵀ Σ⁻¹ (x - μ)`` requires:
- Cross-feature interactions (``Σ⁻¹`` is full-matrix, not diagonal)
- Global covariance pooling (CB sees only per-feature statistics)
- Per-class conditioning at the matrix level (not per-column)

The signed gap ``m_neg - m_pos`` is the LDA log-likelihood-ratio under Gaussian class-conditionals — Bayes-optimal under that assumption. Even when the Gaussian assumption is
violated, it's a strong shrunk-quadratic baseline that CB symmetric oblivious trees cannot derive from per-feature splits.

Cost: one Ledoit-Wolf fit per class per fold (O(N·d²)) + per-row quadratic form (O(N·d²)). At d=6-10 (mammography/diabetes scale), microseconds per row.

References:
- Ledoit & Wolf 2004 — optimal linear-shrinkage covariance.
- McLachlan 1992 — LDA / QDA likelihood ratio.
- Class-conditional Mahalanobis features for tabular ML (less commonly used; this is a frozen FE adaptation).
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _shrunk_covariance(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ledoit-Wolf shrunk covariance + its inverse. Returns (mean, inv_cov)."""
    from sklearn.covariance import LedoitWolf
    if X.shape[0] < 2:
        d = X.shape[1]
        return X.mean(axis=0) if X.shape[0] > 0 else np.zeros(d, dtype=np.float32), np.eye(d, dtype=np.float32)
    lw = LedoitWolf().fit(X)
    mean = X.mean(axis=0).astype(np.float32)
    # pinv tolerates near-singular Ledoit-Wolf covariance without raising; a
    # merely ill-conditioned (not exactly singular) cov would slip past an
    # inv()+LinAlgError guard and yield exploded / non-finite distances.
    inv_cov = np.linalg.pinv(lw.covariance_).astype(np.float32)
    return mean, inv_cov


def _mahalanobis(X: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    """Per-row Mahalanobis distance squared.

    Computed as ``((diff @ inv_cov) * diff).sum(axis=1)`` rather than the naive
    3-operand ``einsum("ij,jk,ik->i", ...)``: the matmul routes the O(n*d^2) work
    through optimized BLAS GEMM (3.7-11.6x faster across n in {4k,20k,100k} x
    d in {6,15,30}; bench_mahalanobis_quadform.py) and accumulates in higher
    precision (max rel err vs fp64 truth 3.2e-7 vs einsum's 2.4e-6), so the
    result is FP-reduction-order equivalent (selection-safe), not a regression.
    """
    diff = X - mean
    return (np.matmul(diff, inv_cov) * diff).sum(axis=1).astype(np.float32)


def compute_class_mahalanobis_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    standardize: bool = True,
    column_prefix: str = "mahcc",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Class-conditional Mahalanobis features for binary classification.

    Output: 3 columns per row — ``{prefix}_m_pos``, ``{prefix}_m_neg``, ``{prefix}_m_gap``.

    Mode A: per-class μ and Σ refit per fold on train-fold rows.
    Mode B: per-class μ and Σ fit once on full X_train.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        if pos_mask.sum() < 2 or neg_mask.sum() < 2:
            n_q = Xq_s.shape[0]
            return np.zeros(n_q, dtype=np.float32), np.zeros(n_q, dtype=np.float32), np.zeros(n_q, dtype=np.float32)
        mu_pos, inv_pos = _shrunk_covariance(Xt_s[pos_mask])
        mu_neg, inv_neg = _shrunk_covariance(Xt_s[neg_mask])
        m_pos = _mahalanobis(Xq_s, mu_pos, inv_pos)
        m_neg = _mahalanobis(Xq_s, mu_neg, inv_neg)
        m_gap = m_neg - m_pos
        return m_pos, m_neg, m_gap

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        m_pos, m_neg, m_gap = _process(X_train_f, Xq, y_train_f)
        return pl.DataFrame({
            f"{column_prefix}_m_pos": m_pos.astype(dtype, copy=False),
            f"{column_prefix}_m_neg": m_neg.astype(dtype, copy=False),
            f"{column_prefix}_m_gap": m_gap.astype(dtype, copy=False),
        })

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out_pos: np.ndarray = np.zeros(n_train, dtype=dtype)
    out_neg: np.ndarray = np.zeros(n_train, dtype=dtype)
    out_gap: np.ndarray = np.zeros(n_train, dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_tr = X_train_f[train_idx]
        X_va = X_train_f[val_idx]
        y_tr = y_train_f[train_idx]
        m_pos, m_neg, m_gap = _process(X_tr, X_va, y_tr)
        out_pos[val_idx] = m_pos.astype(dtype, copy=False)
        out_neg[val_idx] = m_neg.astype(dtype, copy=False)
        out_gap[val_idx] = m_gap.astype(dtype, copy=False)
        logger.info("class_mahalanobis: fold %d/%d done (n_pos=%d, n_neg=%d)", fold_idx + 1, len(splits), int((y_tr > 0.5).sum()), int((y_tr <= 0.5).sum()))

    return pl.DataFrame({
        f"{column_prefix}_m_pos": out_pos,
        f"{column_prefix}_m_neg": out_neg,
        f"{column_prefix}_m_gap": out_gap,
    })
