"""Local KS-test + moment-shift attention: distributional-shift hypothesis-test features per row.

Iter 29 mechanism. Combines two information-theoretically orthogonal signals:
1. **KS statistic**: ``sup_t |F_local(t) - F_global(t)|`` where F_local is the empirical CDF of y in the row's top-k neighbourhood, F_global is the train-wide empirical CDF.
   The KS-distance is a *full-CDF* dissimilarity measure — captures distributional shift CB's mean-aggregating TS encoding cannot derive.
2. **Wasserstein-1 (earth-mover) statistic**: ``∫ |F_local(t) - F_global(t)| dt``, the L1 integrated CDF gap. More sensitive to large-magnitude shifts than the
   sup-based KS; complements it.
3. **Standardised mean-shift**: ``(mean(y_local) - mean(y_global)) / std(y_global)``. Local effect size relative to global noise. Already capturable via row-attention's
   ``y_mean - global_mean`` but normalised differently here.
4. **Standardised variance-shift**: ``log(var(y_local) / var(y_global))``. Tells if local neighbourhood has more or less y-variability than global.

For binary y: KS statistic on binary CDFs reduces to ``|mean(y_local) - mean(y_global)|`` (since CDF only has two steps). Wasserstein for binary ≡ KS (also 1-step
metric). So for binary task only the mean-shift and variance-shift are informative, but they're still useful as CB-blind features.

For regression: all four features are informative.

Output: 4 features per row (or 2 if binary).

Mechanism vs existing iterations:
- Iter 9 target-quantile: bucket centroid distances in X-space; mean-aggregating.
- Iter 20 quantile-neighbours: y-quantile estimates from kNN; quantile output.
- Iter 29: HYPOTHESIS-TEST STATISTICS — KS / Wasserstein / moment-shift. Tests whether local-y is *distributionally different* from global-y. Fundamentally different
  from estimating local-y statistics.

CB's TS encoding answers "what is E[y|feature_j]" per column. KS answers "is the local-y CDF shifted from global?". These are orthogonal — KS exposes regions where
the local distribution is unusual EVEN IF the mean is normal (e.g., bimodal local y vs unimodal global).

Leakage discipline: kNN refit per fold, F_global computed from y_train[train_idx] only.

Cost: kNN search O(N·d·log N) + per-row CDF computation O(k log k). Lightweight.

Reference: Kolmogorov 1933 / Smirnov 1948; Vallender 1974 (Wasserstein-1 as integrated absolute CDF distance).
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


def _ks_and_wasserstein(y_neighbors: np.ndarray, y_global_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each query row (= row of y_neighbors), compute KS and W1 against global y CDF.

    Vectorised: for each row, sort local-y, interpolate both CDFs on the union of sample points, compute sup and integrated abs diff.
    """
    n_q, k = y_neighbors.shape
    n_g = y_global_sorted.shape[0]
    ks_out = np.zeros(n_q, dtype=np.float32)
    w1_out = np.zeros(n_q, dtype=np.float32)
    # Pre-sort local rows.
    y_local_sorted = np.sort(y_neighbors, axis=1)
    cdf_local = (np.arange(k) + 1).astype(np.float32) / k  # CDF heights at sorted-local positions
    # For each query, evaluate global CDF at local sample points (right-side counts).
    # searchsorted on sorted global array gives the global rank at each local sample.
    for i in range(n_q):
        local = y_local_sorted[i]
        # Global CDF evaluated at each local sample.
        global_ranks = np.searchsorted(y_global_sorted, local, side="right").astype(np.float32) / n_g
        # KS: max over sample points of |local_cdf - global_cdf|.
        diff = np.abs(cdf_local - global_ranks)
        ks_out[i] = diff.max()
        # Wasserstein-1: integrate |CDF_local - CDF_global| over y. For piecewise CDFs,
        # this is sum over sample intervals of |local_height - global_height| * (sample_right - sample_left).
        # Use trapezoidal approximation on sorted local samples.
        widths = np.diff(local, prepend=local[0])
        w1_out[i] = (diff * widths).sum()
    return ks_out, w1_out


def compute_ks_shift_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_query: Optional[np.ndarray],
    splitter: Optional[Any] = None,
    *,
    seed: int,
    task: Literal["binary", "regression"] = "binary",
    k: int = 32,
    standardize: bool = True,
    column_prefix: str = "ksshift",
    dtype: np.dtype = np.float32,
) -> pl.DataFrame:
    """Local KS-test + moment-shift features.

    Output: for regression — 4 columns (ks, w1, mean_shift, log_var_ratio).
    For binary — 2 columns (mean_shift, log_var_ratio) since KS/Wasserstein collapse to mean-shift for {0,1}-valued y.
    """
    seed = require_seed(seed)
    validate_numeric_input(X_train, name="X_train", allow_fp16=False)
    if X_query is not None:
        validate_numeric_input(X_query, name="X_query", allow_fp16=False)
    if k < 4:
        raise ValueError(f"k must be >= 4; got {k}.")
    if task not in ("binary", "regression"):
        raise ValueError(f"task must be 'binary' or 'regression'; got {task!r}.")

    X_train_f = np.asarray(X_train, dtype=np.float32)
    y_train_f = np.asarray(y_train, dtype=np.float32).ravel()

    def _process(Xt: np.ndarray, Xq: np.ndarray, y_t: np.ndarray) -> dict[str, np.ndarray]:
        if standardize:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler().fit(Xt)
            Xt_s = scaler.transform(Xt).astype(np.float32)
            Xq_s = scaler.transform(Xq).astype(np.float32)
        else:
            Xt_s = Xt
            Xq_s = Xq
        from sklearn.neighbors import NearestNeighbors
        k_used = min(k, Xt_s.shape[0])
        nn = NearestNeighbors(n_neighbors=k_used, algorithm="auto", n_jobs=-1).fit(Xt_s)
        _dists, ids = nn.kneighbors(Xq_s)
        y_neighbors = y_t[ids]
        # Mean shift (standardised).
        global_mean = float(y_t.mean())
        global_std = float(y_t.std()) + 1e-9
        local_mean = y_neighbors.mean(axis=1)
        mean_shift = ((local_mean - global_mean) / global_std).astype(np.float32)
        # Variance shift (log ratio).
        global_var = float(y_t.var()) + 1e-9
        local_var = y_neighbors.var(axis=1) + 1e-9
        log_var_ratio = np.log(local_var / global_var).astype(np.float32)
        out: dict[str, np.ndarray] = {
            "mean_shift": mean_shift,
            "log_var_ratio": log_var_ratio,
        }
        if task == "regression":
            y_global_sorted = np.sort(y_t)
            ks, w1 = _ks_and_wasserstein(y_neighbors, y_global_sorted)
            out["ks"] = ks
            out["w1"] = w1
        return out

    def _make_df(feats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {f"{column_prefix}_{name}": arr.astype(dtype, copy=False) for name, arr in feats.items()}

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    feat_names = ["mean_shift", "log_var_ratio"] + (["ks", "w1"] if task == "regression" else [])
    out_arrays = {name: np.zeros(n_train, dtype=dtype) for name in feat_names}
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx])
        for name in feat_names:
            out_arrays[name][val_idx] = feats[name].astype(dtype, copy=False)
        logger.info("ks_shift: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out_arrays))
