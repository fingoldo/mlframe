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

import numba
import numpy as np
import polars as pl

from ._utils import require_seed, validate_numeric_input

logger = logging.getLogger(__name__)


@numba.njit(cache=True)
def _count_le(sorted_arr: np.ndarray, v: float) -> int:
    """Number of entries of the sorted array that are <= ``v``."""
    lo = 0
    hi = sorted_arr.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if sorted_arr[mid] <= v:
            lo = mid + 1
        else:
            hi = mid
    return lo


@numba.njit(cache=True)
def _count_lt(sorted_arr: np.ndarray, v: float) -> int:
    """Number of entries of the sorted array that are < ``v``."""
    lo = 0
    hi = sorted_arr.shape[0]
    while lo < hi:
        mid = (lo + hi) >> 1
        if sorted_arr[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return lo


@numba.njit(cache=True)
def _global_cdf_integral(g: np.ndarray, prefix: np.ndarray, p: float, q: float) -> float:
    """``integral_p^q F_g(t) dt`` of the global empirical CDF, from the sorted sample ``g`` and its prefix sums (``prefix[m]`` = sum of the first m values)."""
    n_g = g.shape[0]
    cp = _count_le(g, p)
    cq = _count_le(g, q)
    return float((cq * q - cp * p - (prefix[cq] - prefix[cp])) / n_g)


@numba.njit(cache=True, parallel=True)
def _ks_w1_kernel(y_local_sorted: np.ndarray, y_global_sorted: np.ndarray, prefix: np.ndarray, ks_out: np.ndarray, w1_out: np.ndarray) -> None:
    """Exact two-sample KS statistic and Wasserstein-1 distance of every row of sorted local samples against the sorted global sample, written in place.

    KS compares the local empirical CDF with the global one on both sides of every local sample (the sup of two step functions sits at a jump, and a jump is
    checked from its left and its right limit). W1 is the integral of |F_local - F_global| over the line: F_local is constant on each interval between local
    samples, so the integral is evaluated interval by interval from the global CDF's prefix sums, splitting at the point where F_global crosses the local level.
    Inputs are centred by the caller so the prefix sums do not lose precision.
    """
    n_q, k = y_local_sorted.shape
    n_g = y_global_sorted.shape[0]
    g_lo = y_global_sorted[0]
    g_hi = y_global_sorted[n_g - 1]
    for i in numba.prange(n_q):
        ks = 0.0
        for j in range(k):
            v = y_local_sorted[i, j]
            right = (j + 1) / k - _count_le(y_global_sorted, v) / n_g
            left = _count_lt(y_global_sorted, v) / n_g - j / k
            ks = max(ks, right, left)
        w1 = 0.0
        for j in range(k + 1):
            c = j / k
            # F_local is c on (e_j, e_{j+1}) with e_0 = -inf and e_{k+1} = +inf; F_g is 0 left of g_lo and 1 from g_hi on.
            a_open = j == 0
            b_open = j == k
            a = y_local_sorted[i, j - 1] if not a_open else g_lo
            bb = y_local_sorted[i, j] if not b_open else g_hi
            if not a_open and a < g_lo:
                w1 += c * (min(bb if not b_open else g_lo, g_lo) - a)  # left of every global sample F_g = 0
            if not b_open and bb > g_hi:
                w1 += (1.0 - c) * (bb - max(a if not a_open else g_hi, g_hi))  # right of every global sample F_g = 1
            lo_mid = max(a, g_lo)
            hi_mid = min(bb, g_hi)
            if hi_mid <= lo_mid:
                continue
            # F_g crosses the local level c at x_c = g[ceil(c * n_g) - 1]: F_g < c below it, >= c from it on.
            m_c = int(np.ceil(c * n_g))
            if m_c >= 1:
                x_c = y_global_sorted[m_c - 1]
                below_hi = min(hi_mid, x_c)
                above_lo = max(lo_mid, x_c)
            else:
                below_hi = lo_mid
                above_lo = lo_mid
            if below_hi > lo_mid:
                w1 += c * (below_hi - lo_mid) - _global_cdf_integral(y_global_sorted, prefix, lo_mid, below_hi)
            if hi_mid > above_lo:
                w1 += _global_cdf_integral(y_global_sorted, prefix, above_lo, hi_mid) - c * (hi_mid - above_lo)
        ks_out[i] = ks
        w1_out[i] = w1


def _ks_and_wasserstein(y_neighbors: np.ndarray, y_global_sorted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each query row (= row of y_neighbors), the exact KS sup-distance and Wasserstein-1 distance against the global y CDF.

    Local and global samples are centred on the global mean and handled in float64: W1 comes from prefix sums of the global sample, which cancel badly when the values
    sit far from zero.
    """
    n_q = y_neighbors.shape[0]
    g = np.asarray(y_global_sorted, dtype=np.float64)
    centre = float(g.mean())
    g = g - centre
    y_local_sorted = np.sort(np.asarray(y_neighbors, dtype=np.float64) - centre, axis=1)
    prefix = np.concatenate([[0.0], np.cumsum(g)])
    ks_out = np.zeros(n_q, dtype=np.float64)
    w1_out = np.zeros(n_q, dtype=np.float64)
    _ks_w1_kernel(np.ascontiguousarray(y_local_sorted), np.ascontiguousarray(g), prefix, ks_out, w1_out)
    return ks_out.astype(np.float32), w1_out.astype(np.float32)


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
    dtype: type = np.float32,
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
        """Core per-fold pipeline: scale, find each query row's k nearest train neighbours, then compute standardized mean-shift + log-variance-ratio of their y values vs the global distribution, plus (regression only) local KS/Wasserstein-1 distances against the global y CDF."""
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
        global_std = max(float(y_t.std()), 1e-9)
        local_mean = y_neighbors.mean(axis=1)
        mean_shift = ((local_mean - global_mean) / global_std).astype(np.float32)
        # Variance shift (log ratio).
        global_var = max(float(y_t.var()), 1e-9)
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
        """Prefix each ``_process`` output key with ``column_prefix`` and cast to the requested output ``dtype``."""
        return {f"{column_prefix}_{name}": arr.astype(dtype, copy=False) for name, arr in feats.items()}

    if X_query is not None:
        Xq = np.asarray(X_query, dtype=np.float32)
        feats = _process(X_train_f, Xq, y_train_f)
        return pl.DataFrame(_make_df(feats))

    if splitter is None:
        raise ValueError("Mode A (X_query=None) requires a splitter.")
    n_train = X_train_f.shape[0]
    feat_names = ["mean_shift", "log_var_ratio"] + (["ks", "w1"] if task == "regression" else [])
    out_arrays: dict[str, np.ndarray] = {name: np.full(n_train, np.nan, dtype=dtype) for name in feat_names}
    splits = list(splitter.split(X_train_f))
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        feats = _process(X_train_f[train_idx], X_train_f[val_idx], y_train_f[train_idx])
        for name in feat_names:
            out_arrays[name][val_idx] = feats[name].astype(dtype, copy=False)
        logger.info("ks_shift: fold %d/%d done", fold_idx + 1, len(splits))

    return pl.DataFrame(_make_df(out_arrays))
