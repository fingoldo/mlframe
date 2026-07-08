"""Quantile-regression neighbours: per-row weighted-quantile estimation of y from kNN.

All existing iters that aggregate target over neighbours (rows-attention, anchor, rf-proximity) return y_mean (and optionally y_std). None expose the SHAPE of the
local target distribution. Iter 20 returns weighted-quantile estimates of y at q ∈ {0.1, 0.25, 0.5, 0.75, 0.9} from the same kNN top-k.

Why this matters:
- For regression: skewed local target distributions are common (mammography would be too if it were regression). The mean is a poor summary; quantiles expose
  asymmetry.
- For BINARY classification: the rate y_q10 (10th percentile of neighbour labels) is essentially the "almost-all-negative" indicator, y_q90 the "almost-all-
  positive" indicator. For rare-positive classes (mammography 1.3%, electricity 4%), most neighbourhoods have y_mean ≈ 0; the q90 feature surfaces neighbourhoods
  with ≥10% positives that mean-aggregation washes out.

Mechanism:
1. Build kNN graph on standardised X (cosine top-k via sklearn NearestNeighbors).
2. Per row: weighted softmax over top-k distances → per-neighbour weight w_i.
3. Compute weighted quantile estimator: given (y_i, w_i) sorted by y, find the smallest y_i such that Σ_{j≤i} w_j ≥ q.
4. Return q-quantile per row for each q in quantile_grid.

Mode A: kNN refit per fold on X_train[train_idx]; val rows queried against fold bank.
Mode B: kNN fit once on full X_train; query rows queried.

Cost: dominated by kNN search (same as row-attention's stage 3). Quantile computation is O(N_q * k * log k) which is ms-scale at k=32.

Reference: weighted quantile estimation (Akinshin 2023), quantile regression forests (Meinshausen 2006).
"""
from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numba
import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


@numba.njit(parallel=True, cache=True)
def _weighted_quantiles_njit(y_neighbors: np.ndarray, weights: np.ndarray, qs: np.ndarray) -> np.ndarray:  # pragma: no cover
    """Per-row fused weighted-quantile kernel: one argsort + cumsum + per-q scan per row, no (n_rows, k) temporaries or n_qs full sweeps.

    fastmath=False so the float32 cumsum order and the first-cdf>=q tie semantics are bit-identical to the numpy reference (the argmax-based path).
    """
    n_rows, k = y_neighbors.shape
    n_qs = qs.shape[0]
    out = np.empty((n_rows, n_qs), dtype=np.float32)
    for i in numba.prange(n_rows):
        idx = np.argsort(y_neighbors[i])
        ys = np.empty(k, dtype=np.float32)
        cdf = np.empty(k, dtype=np.float32)
        cum = np.float32(0.0)
        for j in range(k):
            ys[j] = y_neighbors[i, idx[j]]
            cum += weights[i, idx[j]]
            cdf[j] = cum
        for qi in range(n_qs):
            q = qs[qi]
            sel = k - 1
            for j in range(k):
                if cdf[j] >= q:
                    sel = j
                    break
            out[i, qi] = ys[sel]
    return out


def _weighted_quantiles(y_neighbors: np.ndarray, weights: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Vectorised weighted quantile estimator.

    y_neighbors: (n_rows, k) - per-row top-k neighbour y values.
    weights: (n_rows, k) - per-row softmax weights summing to 1.
    qs: (n_qs,) - quantiles in [0, 1].

    Returns (n_rows, n_qs) array of estimated quantiles.
    """
    return _weighted_quantiles_njit(
        np.ascontiguousarray(y_neighbors, dtype=np.float32),
        np.ascontiguousarray(weights, dtype=np.float32),
        np.ascontiguousarray(qs, dtype=np.float32),
    )


def compute_quantile_neighbours(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    k: int = 32,
    quantile_grid: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
    softmax_temp: float = 1.0,
    standardize: bool = True,
    column_prefix: str = "qnn",
    dtype: type = np.float32,
) -> pl.DataFrame:
    """Quantile-regression neighbours features.

    Output: ``len(quantile_grid)`` columns, named ``{column_prefix}_q{int(q*100)}``.

    Mode A: kNN refit per fold; weighted quantile from y_train[train_idx].
    Mode B: kNN fit once on full X_train.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if not quantile_grid or any((q <= 0 or q >= 1) for q in quantile_grid):
        raise ValueError(f"quantile_grid must contain values strictly in (0, 1); got {quantile_grid}.")
    if k < 4:
        raise ValueError(f"k must be >= 4 for sensible quantile estimation; got {k}.")

    qs = np.asarray(quantile_grid, dtype=np.float32)
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
        nn = NearestNeighbors(n_neighbors=min(k, Xt_s.shape[0]), algorithm="auto", n_jobs=-1).fit(Xt_s)
        dists, ids = nn.kneighbors(Xq_s)  # (n_q, k)
        # Softmax weights from negative distances (closer => higher weight).
        logits = -dists / (softmax_temp + 1e-9)
        logits -= logits.max(axis=1, keepdims=True)
        w = np.exp(logits)
        w /= w.sum(axis=1, keepdims=True)
        y_neighbors = y_t[ids]  # (n_q, k)
        quantiles = _weighted_quantiles(y_neighbors, w, qs)
        return quantiles

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        quantiles = _process(X_train_f, Xq, y_train_f)
        cols: dict = {f"{column_prefix}_q{int(round(qs[j] * 100))}": quantiles[:, j].astype(dtype, copy=False) for j in range(qs.shape[0])}
        return pl.DataFrame(cols)

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    n_qs = qs.shape[0]
    out: np.ndarray = np.zeros((n_train, n_qs), dtype=dtype)
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_tr = X_train_f[train_idx]
        X_va = X_train_f[val_idx]
        y_tr = y_train_f[train_idx]
        quantiles = _process(X_tr, X_va, y_tr)
        out[val_idx] = quantiles.astype(dtype, copy=False)
        logger.info("quantile_neighbours: fold %d/%d done (n_train=%d, n_val=%d, k=%d, n_quantiles=%d)", fold_idx + 1, len(splits), len(train_idx), len(val_idx), k, n_qs)

    cols = {f"{column_prefix}_q{int(round(qs[j] * 100))}": out[:, j] for j in range(n_qs)}
    return pl.DataFrame(cols)
