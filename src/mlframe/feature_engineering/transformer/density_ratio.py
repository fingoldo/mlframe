"""Class-conditional kernel density ratio features at multiple bandwidths.

Iter 28 mechanism. Targets CB AUC mammography ceiling AND offers Bayes-rule-aligned discrimination.

For each row, evaluate ``log(KDE_pos(x) / KDE_neg(x))`` at multiple bandwidths (h ∈ {0.5, 1.0, 2.0, 4.0}). The log-ratio is the LDA Bayes-optimal decision feature when
densities are correctly estimated, regardless of distributional form. CB internally approximates this via per-feature splits but cannot construct a multi-feature
density ratio.

Mechanism (binary):
1. Standardise X.
2. Slice train rows by class.
3. For each query, compute kernel density estimate using Gaussian kernel: ``KDE_c(x) = (1 / N_c) Σ_{i ∈ class c} exp(-||x - x_i||² / (2 h²))``.
4. Return log ratio: ``log(KDE_pos(x) + ε) - log(KDE_neg(x) + ε)``.
5. Repeat at multiple bandwidths to expose multi-scale density structure.

Output: ``len(bandwidths)`` features per row — typically 4.

For regression: replace positive/negative class slices with top/bottom y-quantile slices.

Bandwidth choice: ``h ∈ {0.5, 1.0, 2.0, 4.0}`` × Silverman's rule-of-thumb baseline. Multi-scale exposes both fine-grained (h=0.5) and global (h=4.0) class-density
structure.

Leakage discipline: per-class slices computed per fold from train-fold y only.

Cost: O(N_q · N_train · d) per bandwidth for full pairwise distances. Memory-heavy but tractable at N<10k. Vectorised via broadcasting; uses chunked computation
if memory becomes a concern.

Reference: Bayes decision rule via density ratio (Sugiyama et al. 2012); kNN-density estimation (Loftsgaarden & Quesenberry 1965).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_BANDWIDTHS = (0.5, 1.0, 2.0, 4.0)


def _silverman_h(X: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth: h = (4 σ⁵ / 3 N)^(1/5). Returns scalar averaged across features."""
    n, d = X.shape
    if n < 2:
        return 1.0
    sigma = X.std(axis=0).mean()
    h = sigma * (4.0 / (3.0 * n)) ** 0.2
    return float(max(h, 1e-3))


def _gaussian_kde_log(X_query: np.ndarray, X_train_subset: np.ndarray, h: float, chunk: int = 1000) -> np.ndarray:
    """Log-KDE: log( (1/N) Σ exp(-||x - x_i||² / (2 h²)) ).

    Chunked over query rows to keep memory bounded.
    """
    n_q = X_query.shape[0]
    n_t = X_train_subset.shape[0]
    if n_t < 1:
        return np.full(n_q, -30.0, dtype=np.float32)
    out = np.zeros(n_q, dtype=np.float32)
    h_sq = max(h * h, 1e-9)
    log_N = np.log(n_t)
    for start in range(0, n_q, chunk):
        end = min(start + chunk, n_q)
        Xq = X_query[start:end]
        # pairwise squared distances (Xq, X_train_subset)
        d2 = ((Xq[:, None, :] - X_train_subset[None, :, :]) ** 2).sum(axis=2)  # (chunk, n_t)
        logits = -d2 / (2.0 * h_sq)  # (chunk, n_t)
        # log-sum-exp for numerical stability
        m = logits.max(axis=1, keepdims=True)
        lse = m.ravel() + np.log(np.exp(logits - m).sum(axis=1) + 1e-30)
        out[start:end] = (lse - log_N).astype(np.float32)
    return out


def compute_density_ratio_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    bandwidth_multipliers: Tuple[float, ...] = _BANDWIDTHS,
    standardize: bool = True,
    q_low: float = 0.2,
    q_high: float = 0.8,
    column_prefix: str = "denrat",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Class-conditional KDE log-ratio features.

    Output: ``len(bandwidth_multipliers)`` columns per row, named ``{prefix}_b{i}`` for the i-th bandwidth.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if task not in ("binary", "regression"):
        raise ValueError(f"task must be 'binary' or 'regression'; got {task!r}.")

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _slice(X_sub: np.ndarray, y_sub: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if task == "binary":
            pos = y_sub > 0.5
            return X_sub[pos], X_sub[~pos]
        y_lo = np.quantile(y_sub, q_low)
        y_hi = np.quantile(y_sub, q_high)
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
        Xt_pos, Xt_neg = _slice(Xt_s, y_t)
        if Xt_pos.shape[0] < 2 or Xt_neg.shape[0] < 2:
            return np.zeros((Xq_s.shape[0], len(bandwidth_multipliers)), dtype=np.float32)
        h_base = _silverman_h(Xt_s)
        feats = np.zeros((Xq_s.shape[0], len(bandwidth_multipliers)), dtype=np.float32)
        for col_idx, mult in enumerate(bandwidth_multipliers):
            h = h_base * mult
            log_pos = _gaussian_kde_log(Xq_s, Xt_pos, h=h)
            log_neg = _gaussian_kde_log(Xq_s, Xt_neg, h=h)
            feats[:, col_idx] = (log_pos - log_neg).astype(np.float32)
        return feats

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        return {f"{column_prefix}_b{j}": feats[:, j].astype(dtype, copy=False) for j in range(feats.shape[1])}

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out = np.zeros((n_train, len(bandwidth_multipliers)), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx])
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("density_ratio: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
