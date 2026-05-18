"""Fisher LDA axis projection features: project queries onto class-discriminative direction.

Iter 37 mechanism. For binary classification, Fisher's LDA direction is ``w = Σ⁻¹ (μ_pos − μ_neg)`` where ``Σ`` is the pooled within-class covariance (Ledoit-Wolf
shrunk). The projection ``y = w^T x`` is the optimal linear discriminant under Gaussian class-conditionals with equal covariance.

Mechanism (binary):
1. Compute class-conditional means μ_pos, μ_neg.
2. Compute pooled within-class covariance Σ via Ledoit-Wolf shrinkage (numerically stable inversion).
3. Compute Fisher direction w = Σ⁻¹ (μ_pos − μ_neg).
4. Compute decision-boundary threshold ``c = 0.5 × w^T (μ_pos + μ_neg)`` (midpoint between class means along w).
5. Per query: expose 3 features:
   - ``lda_raw``: w^T x — raw LDA score.
   - ``lda_signed``: w^T x − c — signed distance from decision boundary (positive = positive-side).
   - ``lda_magnitude``: |w^T x − c| — distance magnitude (confidence).

For regression: replace pos/neg classes with top-quintile / bottom-quintile y.

CB cannot internally compute Fisher LDA:
- Requires multi-feature centroids per class (per-feature TS doesn't aggregate at row level)
- Requires covariance inversion (CB has no matrix-inverse op)
- Requires pooled within-class statistics (not just per-feature splits)

The LDA projection is to multi-feature linear discriminant what iter 28 denrat is to KDE: a parametric, Gaussian-assumption Bayes-optimal feature for the boosting to split on.

Differs from iter 25 (`class_mahalanobis`) which gives QUADRATIC class-conditional Mahalanobis distances. LDA gives the LINEAR projection — different shape.

Leakage discipline: μ_pos, μ_neg, Σ computed per fold from train-fold rows only.

Cost: O(d²) Ledoit-Wolf + O(d³) inversion per fold + O(N · d) per-row projection. At d=6-10 (mammography/diabetes), microseconds per fold.

Reference: Fisher 1936 — LDA. Ledoit & Wolf 2004 — shrinkage covariance.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _fisher_lda(X_pos: np.ndarray, X_neg: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute Fisher LDA direction and decision threshold.

    Returns (w, c) where w is the LDA direction vector and c is the decision threshold.
    """
    from sklearn.covariance import LedoitWolf
    d = X_pos.shape[1]
    mu_pos = X_pos.mean(axis=0).astype(np.float32)
    mu_neg = X_neg.mean(axis=0).astype(np.float32)
    diff = (mu_pos - mu_neg).astype(np.float32)
    # Pool within-class covariance via Ledoit-Wolf on combined within-class scatter.
    n_p, n_n = X_pos.shape[0], X_neg.shape[0]
    Xp_c = X_pos - mu_pos
    Xn_c = X_neg - mu_neg
    pooled = np.concatenate([Xp_c, Xn_c], axis=0)
    if pooled.shape[0] < 2:
        return diff, float(diff @ (0.5 * (mu_pos + mu_neg)))
    lw = LedoitWolf().fit(pooled)
    cov = lw.covariance_ + np.eye(d) * 1e-4  # ridge for stability
    try:
        cov_inv = np.linalg.inv(cov).astype(np.float32)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov).astype(np.float32)
    w = (cov_inv @ diff).astype(np.float32)
    # Normalize w so its scale is comparable across folds.
    w_norm = np.linalg.norm(w) + 1e-9
    w = w / w_norm
    c = float(w @ (0.5 * (mu_pos + mu_neg)))
    return w, c


def compute_lda_projection_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    q_high: float = 0.8,
    standardize: bool = True,
    column_prefix: str = "lda",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Fisher LDA axis projection features.

    Output: 3 columns per row — ``lda_raw``, ``lda_signed``, ``lda_magnitude``.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _slice(X_sub: np.ndarray, y_sub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if task == "binary":
            pos = y_sub > 0.5
            return X_sub[pos], X_sub[~pos]
        y_hi = np.quantile(y_sub, q_high)
        y_lo = np.quantile(y_sub, 1.0 - q_high)
        return X_sub[y_sub >= y_hi], X_sub[y_sub <= y_lo]

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray) -> np.ndarray:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        X_pos, X_neg = _slice(Xt_s, y_t)
        if X_pos.shape[0] < 2 or X_neg.shape[0] < 2:
            return np.zeros((Xq_s.shape[0], 3), dtype=np.float32)
        w, c = _fisher_lda(X_pos, X_neg)
        raw = (Xq_s @ w).astype(np.float32)
        signed = raw - np.float32(c)
        magnitude = np.abs(signed)
        return np.column_stack([raw, signed, magnitude]).astype(np.float32)

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        names = ["raw", "signed", "magnitude"]
        return {f"{column_prefix}_{name}": feats[:, j].astype(dtype, copy=False) for j, name in enumerate(names)}

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, 3), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx])
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("lda_projection: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
