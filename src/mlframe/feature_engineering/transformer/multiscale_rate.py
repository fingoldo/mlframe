"""Multi-scale local class-rate / quantile-rate features: positive-class fraction in kNN at multiple neighborhood sizes.

Iter 31 mechanism. For binary: fraction of positives in top-k neighbors at k ∈ {4, 8, 16, 32, 64, 128}. For regression: fraction of top-quintile-y neighbors at the same scales.

Mathematical intuition: under rare positive class, the density gradient ``p(y=1 | x, k)`` for varying k captures the LOCAL CURVATURE of the class-conditional density at x.
A point near a tight positive cluster has high rate at small k that drops at large k; a point at the cluster boundary has the opposite shape; a point inside the negative
manifold has uniformly low rate. CB's TS encoding gives global ``E[y | feature_j]`` per column; it cannot compute per-row multi-k rates as a profile.

This differs from iter 24 (`compute_local_lift_features`) which exposes only a SINGLE k normalized by global mean. Iter 31 exposes the RAW rate at MULTIPLE k scales —
the boosting can split on any one or compute differences across scales itself.

Output: ``len(k_scales)`` features per row (typically 6).

For regression: replace positive-class indicator with "is y in top quintile" — output the fraction of top-y neighbors at each k.

Leakage discipline: per-fold kNN refit on train-fold rows; train-fold-y used for class/quantile labels.

Cost: one kNN search per fold at the largest k_scale (since smaller k subsets are nested prefixes of the larger). O(N·log N + N·k_max) per fold.

References:
- Multi-scale kNN density estimation (Loftsgaarden & Quesenberry 1965 at fixed k; multi-k is a natural extension).
- "Density gradient" features in spatial statistics; not typically used as a frozen feature engineering primitive for tabular ML.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional, Tuple

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)

_K_SCALES_DEFAULT = (4, 8, 16, 32, 64, 128)


def _classify_or_quantile_indicator(y: np.ndarray, task: str, q_high: float = 0.8) -> np.ndarray:
    """For binary: y itself (0/1). For regression: indicator y >= q_high quantile of y."""
    if task == "binary":
        return (y > 0.5).astype(np.float32)
    threshold = np.quantile(y, q_high)
    return np.asarray((y >= threshold).astype(np.float32))


def compute_multiscale_rate_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    k_scales: Tuple[int, ...] = _K_SCALES_DEFAULT,
    standardize: bool = True,
    q_high: float = 0.8,
    column_prefix: str = "msrate",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Multi-scale local positive-rate (binary) or top-quintile-rate (regression) features.

    Output: ``len(k_scales)`` columns per row.

    Mode A: kNN refit per fold; rates computed from y_train[train_idx].
    Mode B: kNN fit once on full X_train.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if task not in ("binary", "regression"):
        raise ValueError(f"task must be 'binary' or 'regression'; got {task!r}.")
    if not k_scales or any(k < 2 for k in k_scales):
        raise ValueError(f"k_scales must be a non-empty sequence of ints >= 2; got {k_scales}.")

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
        from sklearn.neighbors import NearestNeighbors
        k_max = min(max(k_scales), Xt_s.shape[0])
        nn = NearestNeighbors(n_neighbors=k_max, algorithm="auto", n_jobs=-1).fit(Xt_s)
        _dists, ids = nn.kneighbors(Xq_s)
        ind_t = _classify_or_quantile_indicator(y_t, task=task, q_high=q_high)
        # Vectorised cumulative sum over kNN: cumsum_k[i, j] = number of positives in top-(j+1) NN of row i.
        ind_n = ind_t[ids]  # (n_q, k_max)
        cumsum = np.cumsum(ind_n, axis=1)
        out = np.zeros((Xq_s.shape[0], len(k_scales)), dtype=np.float32)
        for col_idx, k in enumerate(k_scales):
            eff_k = min(k, k_max)
            out[:, col_idx] = cumsum[:, eff_k - 1] / eff_k
        return out

    def _make_df(feats: np.ndarray) -> dict[str, np.ndarray]:
        return {f"{column_prefix}_k{k}": feats[:, j].astype(dtype, copy=False) for j, k in enumerate(k_scales)}

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    out: np.ndarray = np.zeros((n_train, len(k_scales)), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx])
        out[val_idx] = feats.astype(dtype, copy=False)
        logger.info("multiscale_rate: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out))
