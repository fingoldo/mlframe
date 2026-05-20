"""Improved orthogonal-polynomial pair Feature Engineering.

Supports four orthogonal polynomial families via the basis kwarg: Hermite, Legendre, Chebyshev, Laguerre.
Default basis is Chebyshev, picked empirically across 12 synthetic + UCI regimes -- it never finishes last,
has the highest minimum MI, and dominates real-world tabular data + threshold targets.
See _benchmarks/bench_polynomial_bases.py for the supporting table.

Idea: orthogonal polynomials form a complete basis on their natural domain, so any sufficiently smooth bivariate
function f(x_a, x_b) can be represented as Sum c_{a,i} c_{b,j} P_i(x_a) P_j(x_b) -- find coefficients via Optuna,
MI-against-target as the objective. Replaces hand-coded unary x binary transformations with a single learned
parametric family.

Key implementation choices vs naive Hermite-only:

1. Standardisation. hermval(raw_x, c) blows up numerically when |x| >> 1 (high-degree Hermite goes superlinear).
   Z-score inputs before evaluation so [-3, 3] covers ~99.7% of the support.
2. Right Hermite family. Numpy's polynomial.hermite is the physicist's H_n(x) (weight e^{-x^2}); for z-scored
   inputs (standard Normal) we want the probabilist's He_n(x) (weight e^{-x^2/2}) -- polynomial.hermite_e.hermeval.
3. Tight coefficient range [-2, 2] instead of [-10, 10]: higher-degree terms dominate quickly, large ranges
   make TPE wander.
4. Fixed degree per study: random length per trial breaks TPE's posterior. Degrees swept as an outer loop.
5. L2 regularisation: penalty -lambda * ||c||^2 on the MI objective keeps coefficients bounded.
6. Identity baseline: returns best_mi only when it strictly beats baseline MI((x_a, x_b), y).

Usage::

    from mlframe.feature_selection.filters.hermite_fe import optimise_hermite_pair, HermiteResult
    res = optimise_hermite_pair(x_a=col_a, x_b=col_b, y=target, n_trials=200, max_degree=4, n_jobs=1)
    if res.uplift > 1.05:
        engineered = res.transform(x_a, x_b)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numpy.polynomial.hermite_e import hermeval  # probabilist's Hermite
from numpy.polynomial.legendre import legval
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.laguerre import lagval
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    # No-op decorators so the file imports without numba.
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco
    def prange(n):
        return range(n)


# Fast plug-in MI estimator (numba-accelerated). The polynomial-pair FE objective evaluates MI thousands of
# times during Optuna search; sklearn's KSG was 45% of cProfile wall-time. The njit plug-in below is ~50-100x
# faster on n<=10000 because it skips joblib, sklearn validation, and the Cython kNN search.
#
# Why plug-in is OK as Optuna objective (not as final reported MI):
# * Optuna only needs a monotone proxy of "is this coefficient set better?" -- absolute MI value is irrelevant.
# * Plug-in over-estimates MI vs KSG (entropy bias), but the bias is nearly constant across coefficient sets
#   (same n, same n_bins), so the optimum coefficient set is the same.
# * Quantile binning is rank-stable -- same as KSG's underlying permutation invariance.
# A separate "mi_estimator='ksg'" path keeps sklearn KSG as the reference; both paths reach equivalent best
# coefficients on the 12-regime sweep.


@njit(cache=True, fastmath=True)
def _quantile_bin_njit(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin a 1-D continuous array into n_bins equi-frequency bins. Returns int32 bin indices in [0, n_bins)."""
    n = x.shape[0]
    sort_idx = np.argsort(x)
    out = np.empty(n, dtype=np.int32)
    pos = 0
    base = n // n_bins
    rem = n % n_bins
    for b in range(n_bins):
        size = base + (1 if b < rem else 0)
        for _ in range(size):
            out[sort_idx[pos]] = b
            pos += 1
    return out


@njit(cache=True, fastmath=True)
def _plugin_mi_classif_njit(x: np.ndarray, y: np.ndarray,
                              n_bins: int = 20) -> float:
    """Plug-in MI estimator for continuous x (1-D float64) and discrete y (1-D int64). Returns MI in nats.
    ~50x faster than sklearn for n<=10k, single-thread."""
    n = x.shape[0]
    n_classes = 0
    for i in range(n):
        if y[i] >= n_classes:
            n_classes = y[i] + 1

    x_binned = _quantile_bin_njit(x, n_bins)

    hist_xy = np.zeros((n_bins, n_classes), dtype=np.int64)
    hist_x = np.zeros(n_bins, dtype=np.int64)
    hist_y = np.zeros(n_classes, dtype=np.int64)
    for i in range(n):
        b = x_binned[i]
        c = y[i]
        hist_xy[b, c] += 1
        hist_x[b] += 1
        hist_y[c] += 1

    log_n = math.log(n)
    mi = 0.0
    for b in range(n_bins):
        if hist_x[b] == 0:
            continue
        log_hx = math.log(hist_x[b])
        for c in range(n_classes):
            n_xy = hist_xy[b, c]
            if n_xy == 0 or hist_y[c] == 0:
                continue
            mi += (n_xy / n) * (math.log(n_xy) + log_n - log_hx - math.log(hist_y[c]))
    if mi < 0.0:
        mi = 0.0
    return mi


@njit(cache=True, fastmath=True)
def _plugin_mi_regression_njit(x: np.ndarray, y: np.ndarray,
                                 n_bins: int = 20) -> float:
    """Plug-in MI for continuous x (1-D) and continuous y (1-D). Bin both into n_bins equi-frequency bins, then plug-in estimator."""
    n = x.shape[0]
    x_binned = _quantile_bin_njit(x, n_bins)
    y_binned = _quantile_bin_njit(y, n_bins)

    hist_xy = np.zeros((n_bins, n_bins), dtype=np.int64)
    hist_x = np.zeros(n_bins, dtype=np.int64)
    hist_y = np.zeros(n_bins, dtype=np.int64)
    for i in range(n):
        bx = x_binned[i]
        by = y_binned[i]
        hist_xy[bx, by] += 1
        hist_x[bx] += 1
        hist_y[by] += 1

    log_n = math.log(n)
    mi = 0.0
    for bx in range(n_bins):
        if hist_x[bx] == 0:
            continue
        log_hx = math.log(hist_x[bx])
        for by in range(n_bins):
            n_xy = hist_xy[bx, by]
            if n_xy == 0 or hist_y[by] == 0:
                continue
            mi += (n_xy / n) * (math.log(n_xy) + log_n - log_hx - math.log(hist_y[by]))
    if mi < 0.0:
        mi = 0.0
    return mi


@njit(cache=True, fastmath=True, parallel=True)
def _plugin_mi_classif_batch_njit(X_cols: np.ndarray, y: np.ndarray,
                                    n_bins: int = 20) -> np.ndarray:
    """Plug-in MI of each column of X_cols (continuous) with discrete y. Parallel over columns; for k~3 (one per binary func)
    parallelism is shallow but still saves ~2x over sequential."""
    k = X_cols.shape[1]
    out = np.zeros(k, dtype=np.float64)
    for j in prange(k):
        out[j] = _plugin_mi_classif_njit(X_cols[:, j].copy(), y, n_bins)
    return out


@njit(cache=True, fastmath=True)
def _plugin_mi_from_binned_njit(x_binned: np.ndarray, y: np.ndarray,
                                  n_bins: int) -> float:
    """Plug-in MI given pre-binned x. Skips the ``np.argsort`` step inside
    :func:`_plugin_mi_classif_njit`; the caller does the argsort + bin
    assignment in pure numpy (which is ~1.6x faster than numba-wrapped
    ``np.argsort`` at n=1500, measured 2026-05-20 — numba's argsort
    dispatch eats ~70us out of 92us total).

    This is the kernel that numba is genuinely good at: tight nested
    histogram loops, log/log_n math, plug-in MI summation. Bench at
    n=1500, n_classes=3: ~10us per call, vs ~128us for the full
    ``_plugin_mi_classif_njit``.
    """
    n = x_binned.shape[0]
    n_classes = 0
    for i in range(n):
        if y[i] >= n_classes:
            n_classes = y[i] + 1

    hist_xy = np.zeros((n_bins, n_classes), dtype=np.int64)
    hist_x = np.zeros(n_bins, dtype=np.int64)
    hist_y = np.zeros(n_classes, dtype=np.int64)
    for i in range(n):
        b = x_binned[i]
        c = y[i]
        hist_xy[b, c] += 1
        hist_x[b] += 1
        hist_y[c] += 1

    log_n = math.log(n)
    mi = 0.0
    for b in range(n_bins):
        if hist_x[b] == 0:
            continue
        log_hx = math.log(hist_x[b])
        for c in range(n_classes):
            n_xy = hist_xy[b, c]
            if n_xy == 0 or hist_y[c] == 0:
                continue
            mi += (n_xy / n) * (math.log(n_xy) + log_n - log_hx - math.log(hist_y[c]))
    if mi < 0.0:
        mi = 0.0
    return mi


def _quantile_bin_numpy(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Pure-numpy quantile binning. ~1.6x faster than the numba version
    at n=1500 because numpy's ``np.argsort`` dispatches to a SIMD-optimised
    C sort that numba's argsort wrapper does not match.

    Used by the hot CMA-ES inner loop in :func:`optimise_hermite_pair`
    via :func:`plugin_mi_classif_fast` / :func:`plugin_mi_classif_batch_fast`
    which split the argsort (numpy) from the histogram math (njit).
    """
    n = x.shape[0]
    sort_idx = np.argsort(x)
    out = np.empty(n, dtype=np.int32)
    base = n // n_bins
    rem = n % n_bins
    pos = 0
    for b in range(n_bins):
        size = base + (1 if b < rem else 0)
        out[sort_idx[pos:pos + size]] = b
        pos += size
    return out


def plugin_mi_classif_fast(x: np.ndarray, y: np.ndarray,
                            n_bins: int = 20) -> float:
    """Faster single-column plug-in MI: numpy argsort + njit histogram math.

    Measured 2026-05-20 at n=1500 (CMA-ES inner-loop scale):
    - ``_plugin_mi_classif_njit`` (all-in-numba):                    128us
    - ``plugin_mi_classif_fast`` (numpy argsort + njit histogram):  ~67us
    -> **~1.9x speedup** on the hottest path. Numerical result is
    bit-for-bit identical (same quantile-bin recipe, same plug-in MI
    formula).

    Usage scope: SINGLE-COLUMN (k=1) hot paths only. For batch (k>=5),
    the parallel ``_plugin_mi_classif_batch_njit`` (prange over columns)
    wins over the per-column-numpy-argsort loop here. Exposed publicly
    for ad-hoc callers that compute single-column MI in a tight loop
    (e.g. residualised baseline pairs); ``plugin_mi_classif_dispatch``
    does NOT auto-route here because it already passes batches through
    the prange path which beats this implementation at k>=5.
    """
    x_binned = _quantile_bin_numpy(x, n_bins)
    return float(_plugin_mi_from_binned_njit(
        x_binned, np.asarray(y, dtype=np.int64), n_bins,
    ))


def plugin_mi_classif_batch_fast(X_cols: np.ndarray, y: np.ndarray,
                                   n_bins: int = 20) -> np.ndarray:
    """Batch variant of :func:`plugin_mi_classif_fast`. Does argsort + bin
    assignment per column in pure numpy then dispatches the histogram
    math to the njit kernel. Wins over ``_plugin_mi_classif_batch_njit``
    when k is small (<= ~10) because the prange-overhead and per-thread
    argsort cost dominate at low column counts.
    """
    n, k = X_cols.shape
    y_i64 = np.asarray(y, dtype=np.int64)
    out = np.empty(k, dtype=np.float64)
    for j in range(k):
        x_binned = _quantile_bin_numpy(
            np.ascontiguousarray(X_cols[:, j]), n_bins,
        )
        out[j] = _plugin_mi_from_binned_njit(x_binned, y_i64, n_bins)
    return out


# CUDA (cupy) port of plug-in MI for the batch path. At n >= ~300k * k >= 20
# the H2D + argsort on GPU + scatter histogram beats prange on CPU even with
# transfer overhead amortised over k columns. Single-column CUDA is slower
# than the njit version below ~500k due to setup cost; the dispatcher below
# routes accordingly.
def _plugin_mi_classif_batch_cuda(X_cols: np.ndarray, y: np.ndarray,
                                  n_bins: int = 20) -> np.ndarray:
    """Cupy batch plug-in MI. Quantile-bins each column on GPU via
    ``cp.argsort`` -> rank-to-bin lookup, then computes joint histograms
    via a single ``cp.bincount`` across (col, bin, class) flat indices.
    Numerically equivalent to :func:`_plugin_mi_classif_batch_njit`
    up to fp64 round-off.
    """
    import cupy as cp
    X_gpu = cp.asarray(X_cols, dtype=cp.float64)
    y_gpu = cp.asarray(y, dtype=cp.int64)
    n, k = X_gpu.shape
    n_classes = int(cp.max(y_gpu).item()) + 1 if n > 0 else 0
    if n == 0 or k == 0 or n_classes == 0:
        return np.zeros(k, dtype=np.float64)

    # Per-column quantile binning: argsort -> rank -> bin lookup.
    # bin_for_rank[r] = floor(r / (n / n_bins)) with the remainder
    # absorbed by the first ``rem`` bins (matches njit version exactly).
    sort_idx = cp.argsort(X_gpu, axis=0)  # (n, k) int64
    base = n // n_bins
    rem = n - base * n_bins
    sizes = cp.full(n_bins, base, dtype=cp.int64)
    if rem > 0:
        sizes[:rem] += 1
    offsets = cp.empty(n_bins, dtype=cp.int64)
    offsets[0] = 0
    if n_bins > 1:
        offsets[1:] = cp.cumsum(sizes[:-1])
    ranks = cp.arange(n, dtype=cp.int64)
    bin_for_rank = cp.searchsorted(offsets, ranks, side="right") - 1  # (n,)

    # Scatter rank-to-row: X_binned[sort_idx[r, j], j] = bin_for_rank[r]
    X_binned = cp.empty((n, k), dtype=cp.int64)
    cols_idx = cp.broadcast_to(cp.arange(k, dtype=cp.int64)[None, :], (n, k))
    X_binned[sort_idx, cols_idx] = bin_for_rank[:, None]

    # Joint hist via single bincount on flat index (col, bin, class).
    j_idx = cp.broadcast_to(cp.arange(k, dtype=cp.int64)[None, :], (n, k))
    y_b = y_gpu[:, None]
    flat = (j_idx * n_bins + X_binned) * n_classes + y_b  # (n, k)
    hist_flat = cp.bincount(
        flat.ravel(), minlength=k * n_bins * n_classes,
    )
    hist_xyc = hist_flat.reshape(k, n_bins, n_classes).astype(cp.float64)
    hist_x = hist_xyc.sum(axis=2)  # (k, n_bins)
    hist_y = hist_xyc.sum(axis=1)  # (k, n_classes); same across cols but kept per-col for cleanliness

    # MI sum vectorised across cells. Cells with zero count contribute 0
    # (same as the njit ``if n_xy == 0: continue`` short-circuit).
    log_n = math.log(n)
    mask = hist_xyc > 0
    safe_xyc = cp.where(mask, hist_xyc, 1.0)
    safe_x = cp.where(hist_x > 0, hist_x, 1.0)
    safe_y = cp.where(hist_y > 0, hist_y, 1.0)
    term = (hist_xyc / n) * (
        cp.log(safe_xyc) + log_n
        - cp.log(safe_x)[:, :, None] - cp.log(safe_y)[:, None, :]
    )
    mi = cp.where(mask, term, 0.0).sum(axis=(1, 2))  # (k,)
    mi = cp.maximum(mi, 0.0)
    return cp.asnumpy(mi)


def _plugin_mi_classif_cuda(x: np.ndarray, y: np.ndarray,
                            n_bins: int = 20) -> float:
    """Single-column cupy wrapper around :func:`_plugin_mi_classif_batch_cuda`.
    Provided for API symmetry; the dispatcher routes here only when n is
    big enough to amortise H2D + GPU launch (default >= 1M)."""
    X_cols = np.ascontiguousarray(x).reshape(-1, 1)
    res = _plugin_mi_classif_batch_cuda(X_cols, y, n_bins)
    return float(res[0])


# MI dispatcher backend choice. The 2026-05-20 fix routes through the
# ``pyutilz.system.kernel_tuning_cache`` infrastructure (already used for
# joint_hist_batched) instead of hardcoded global thresholds. The KTC
# pipeline:
#   1. ``lookup_mi_classif_backend(n, k)`` -> consults the per-host JSON
#      cache (~/.pyutilz/kernel_tuning/<hw_fingerprint>.json) and returns
#      "njit" or "cuda".
#   2. On cache miss: auto-tune sweep (~10-30s once per host) measures
#      the (n_samples, k) grid and persists.
#   3. Fallback (no pyutilz / no cuda): hand-coded measurements per HW
#      fingerprint -- on GTX 1050 Ti cc 6.1 (2026-05-20 sweep):
#      single-col cuda from n>=75k, batch (k>=5) cuda from n>=10k.
# Env-var ``MLFRAME_MI_BACKEND`` (``njit`` / ``cuda``) still force-
# overrides regardless of cache.
import os as _mi_os


def plugin_mi_classif_dispatch(x: np.ndarray, y: np.ndarray,
                                n_bins: int = 20) -> float:
    """Single-column plug-in MI for continuous x and discrete y.

    Routes to :func:`_plugin_mi_classif_njit` (CPU) or
    :func:`_plugin_mi_classif_cuda` (GPU) via the kernel tuning cache
    (per-host measurement-backed). Override via ``MLFRAME_MI_BACKEND``
    env var (``njit`` | ``cuda``) to force a specific backend.
    """
    forced = _mi_os.environ.get("MLFRAME_MI_BACKEND", "")
    n = x.shape[0]
    if forced == "njit":
        return float(_plugin_mi_classif_njit(x, y, n_bins))
    if forced == "cuda":
        if _CUDA_AVAILABLE:
            return _plugin_mi_classif_cuda(x, y, n_bins)
        return float(_plugin_mi_classif_njit(x, y, n_bins))
    # Consult the kernel tuning cache. Fallback to njit when cuda
    # unavailable regardless of cache choice.
    if not _CUDA_AVAILABLE:
        return float(_plugin_mi_classif_njit(x, y, n_bins))
    try:
        from mlframe.feature_selection._benchmarks.kernel_tuning_cache.dispatch import (
            lookup_mi_classif_backend,
        )
        backend = lookup_mi_classif_backend(n, 1, run_auto_tune=False)
    except Exception:
        backend = "cuda" if n >= 75_000 else "njit"
    if backend == "cuda":
        return _plugin_mi_classif_cuda(x, y, n_bins)
    return float(_plugin_mi_classif_njit(x, y, n_bins))


def plugin_mi_classif_batch_dispatch(X_cols: np.ndarray, y: np.ndarray,
                                      n_bins: int = 20) -> np.ndarray:
    """Batch plug-in MI per column of ``X_cols`` against discrete ``y``.

    Routes to :func:`_plugin_mi_classif_batch_njit` (prange CPU) or
    :func:`_plugin_mi_classif_batch_cuda` (GPU) via the kernel tuning
    cache. Override via ``MLFRAME_MI_BACKEND`` env var.
    """
    forced = _mi_os.environ.get("MLFRAME_MI_BACKEND", "")
    n, k = X_cols.shape
    if forced == "njit":
        return _plugin_mi_classif_batch_njit(X_cols, y, n_bins)
    if forced == "cuda":
        if _CUDA_AVAILABLE:
            return _plugin_mi_classif_batch_cuda(X_cols, y, n_bins)
        return _plugin_mi_classif_batch_njit(X_cols, y, n_bins)
    if not _CUDA_AVAILABLE:
        return _plugin_mi_classif_batch_njit(X_cols, y, n_bins)
    try:
        from mlframe.feature_selection._benchmarks.kernel_tuning_cache.dispatch import (
            lookup_mi_classif_backend,
        )
        backend = lookup_mi_classif_backend(n, k, run_auto_tune=False)
    except Exception:
        backend = "cuda" if (k == 1 and n >= 75_000) or (k > 1 and n >= 10_000) else "njit"
    if backend == "cuda":
        return _plugin_mi_classif_batch_cuda(X_cols, y, n_bins)
    return _plugin_mi_classif_batch_njit(X_cols, y, n_bins)


@njit(cache=True, fastmath=True, parallel=True)
def _plugin_mi_regression_batch_njit(X_cols: np.ndarray, y: np.ndarray,
                                       n_bins: int = 20) -> np.ndarray:
    """Plug-in MI of each column of X_cols (continuous) with continuous y."""
    k = X_cols.shape[1]
    out = np.zeros(k, dtype=np.float64)
    for j in prange(k):
        out[j] = _plugin_mi_regression_njit(X_cols[:, j].copy(), y, n_bins)
    return out


# njit polynomial evaluators. numpy's polyval-family is C-optimized but carries Python dispatch overhead
# (~30-40us); for n~2000 with degree<=4 dispatch dominates. Empirical: njit hermeval ~12us vs numpy 46us (3.7x);
# njit legval ~10us vs numpy 64us (6.3x). Gap shrinks at n>=20k where numpy's vectorization wins.
#
# Recurrences (probabilist's variants where applicable):
# * Hermite_e (He_n): He_0=1, He_1=x, He_n = x*He_{n-1} - (n-1)*He_{n-2}
# * Legendre  (P_n) : P_0=1,  P_1=x,  P_n = ((2n-1)*x*P_{n-1} - (n-1)*P_{n-2}) / n
# * Chebyshev (T_n) : T_0=1,  T_1=x,  T_n = 2*x*T_{n-1} - T_{n-2}
# * Laguerre  (L_n) : L_0=1,  L_1=1-x, L_n = ((2n-1-x)*L_{n-1} - (n-1)*L_{n-2}) / n


@njit(cache=True, fastmath=True)
def _hermeval_njit(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros(n, dtype=np.float64)
    nc = c.shape[0]
    if nc == 0:
        return out
    for i in range(n):
        out[i] = c[0]
    if nc == 1:
        return out
    p_prev = np.ones(n, dtype=np.float64)
    p_curr = x.copy()
    for i in range(n):
        out[i] += c[1] * p_curr[i]
    for k in range(2, nc):
        p_next = np.empty(n, dtype=np.float64)
        ck = c[k]
        km1 = k - 1
        for i in range(n):
            p_next[i] = x[i] * p_curr[i] - km1 * p_prev[i]
            out[i] += ck * p_next[i]
        p_prev = p_curr
        p_curr = p_next
    return out


@njit(cache=True, fastmath=True)
def _legval_njit(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros(n, dtype=np.float64)
    nc = c.shape[0]
    if nc == 0:
        return out
    for i in range(n):
        out[i] = c[0]
    if nc == 1:
        return out
    p_prev = np.ones(n, dtype=np.float64)
    p_curr = x.copy()
    for i in range(n):
        out[i] += c[1] * p_curr[i]
    for k in range(2, nc):
        p_next = np.empty(n, dtype=np.float64)
        ck = c[k]
        inv_k = 1.0 / k
        two_km1 = 2 * k - 1
        km1 = k - 1
        for i in range(n):
            p_next[i] = (two_km1 * x[i] * p_curr[i] - km1 * p_prev[i]) * inv_k
            out[i] += ck * p_next[i]
        p_prev = p_curr
        p_curr = p_next
    return out


@njit(cache=True, fastmath=True)
def _chebval_njit(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros(n, dtype=np.float64)
    nc = c.shape[0]
    if nc == 0:
        return out
    for i in range(n):
        out[i] = c[0]
    if nc == 1:
        return out
    p_prev = np.ones(n, dtype=np.float64)
    p_curr = x.copy()
    for i in range(n):
        out[i] += c[1] * p_curr[i]
    for k in range(2, nc):
        p_next = np.empty(n, dtype=np.float64)
        ck = c[k]
        for i in range(n):
            p_next[i] = 2.0 * x[i] * p_curr[i] - p_prev[i]
            out[i] += ck * p_next[i]
        p_prev = p_curr
        p_curr = p_next
    return out


@njit(cache=True, fastmath=True)
def _lagval_njit(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    out = np.zeros(n, dtype=np.float64)
    nc = c.shape[0]
    if nc == 0:
        return out
    for i in range(n):
        out[i] = c[0]
    if nc == 1:
        return out
    p_prev = np.ones(n, dtype=np.float64)
    p_curr = np.empty(n, dtype=np.float64)
    for i in range(n):
        p_curr[i] = 1.0 - x[i]
        out[i] += c[1] * p_curr[i]
    for k in range(2, nc):
        p_next = np.empty(n, dtype=np.float64)
        ck = c[k]
        inv_k = 1.0 / k
        two_km1 = 2 * k - 1
        km1 = k - 1
        for i in range(n):
            p_next[i] = ((two_km1 - x[i]) * p_curr[i] - km1 * p_prev[i]) * inv_k
            out[i] += ck * p_next[i]
        p_prev = p_curr
        p_curr = p_next
    return out


# Parallel-prange variants of the polynomial evaluators. Per-element Horner recurrence runs in registers
# (no intermediate p_prev / p_curr arrays), so prange over array elements scales linearly with cores at the
# cost of recomputing the recurrence per element. Wins for n >= 50k where memory bandwidth + thread-spawn
# overhead is amortised.


@njit(cache=True, fastmath=True, parallel=True)
def _hermeval_njit_parallel(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    nc = c.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if nc == 0:
        return out
    if nc == 1:
        c0 = c[0]
        for i in prange(n):
            out[i] = c0
        return out
    for i in prange(n):
        xi = x[i]
        p_prev = 1.0
        p_curr = xi
        s = c[0] + c[1] * p_curr
        for k in range(2, nc):
            p_next = xi * p_curr - (k - 1) * p_prev
            s += c[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        out[i] = s
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _legval_njit_parallel(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    nc = c.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if nc == 0:
        return out
    if nc == 1:
        c0 = c[0]
        for i in prange(n):
            out[i] = c0
        return out
    for i in prange(n):
        xi = x[i]
        p_prev = 1.0
        p_curr = xi
        s = c[0] + c[1] * p_curr
        for k in range(2, nc):
            inv_k = 1.0 / k
            two_km1 = 2 * k - 1
            km1 = k - 1
            p_next = (two_km1 * xi * p_curr - km1 * p_prev) * inv_k
            s += c[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        out[i] = s
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _chebval_njit_parallel(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    nc = c.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if nc == 0:
        return out
    if nc == 1:
        c0 = c[0]
        for i in prange(n):
            out[i] = c0
        return out
    for i in prange(n):
        xi = x[i]
        p_prev = 1.0
        p_curr = xi
        s = c[0] + c[1] * p_curr
        for k in range(2, nc):
            p_next = 2.0 * xi * p_curr - p_prev
            s += c[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        out[i] = s
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _lagval_njit_parallel(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    nc = c.shape[0]
    out = np.zeros(n, dtype=np.float64)
    if nc == 0:
        return out
    if nc == 1:
        c0 = c[0]
        for i in prange(n):
            out[i] = c0
        return out
    for i in prange(n):
        xi = x[i]
        p_prev = 1.0
        p_curr = 1.0 - xi
        s = c[0] + c[1] * p_curr
        for k in range(2, nc):
            inv_k = 1.0 / k
            two_km1 = 2 * k - 1
            km1 = k - 1
            p_next = ((two_km1 - xi) * p_curr - km1 * p_prev) * inv_k
            s += c[k] * p_next
            p_prev = p_curr
            p_curr = p_next
        out[i] = s
    return out


# 2026-05-18 Basis-matrix builders for GEMV-style polynomial evaluation.
#
# Per-iteration Horner (above) recomputes the polynomial basis ``T_k(x[i])``
# from scratch for every (coef_a, coef_b) trial inside CMA-ES. That's
# wasteful when the SAME ``z_a`` / ``z_b`` arrays are evaluated against
# many different coef vectors: the basis values ``T_k(z[i])`` are coef-
# independent.
#
# These builders compute the full basis matrix ``B[i, k] = T_k(z[i])`` of
# shape ``(n, max_degree + 1)`` ONCE per (pair, basis). Then per-trial
# evaluation reduces to a BLAS GEMV ``h = B[:, :len(c)] @ c`` - typically
# 2-5x faster than custom njit Horner because BLAS is decades-tuned
# (SIMD, cache blocking).
#
# Memory cost: at n=200_000 and max_degree=8, B is 200k x 9 = 14 MB
# float64 - negligible. At the multi-fidelity 1500-sample inner-search
# path it's ~110 KB.


@njit(cache=True, fastmath=True, parallel=True)
def _build_basis_hermite(x: np.ndarray, max_degree: int) -> np.ndarray:
    """Build ``B[i, k] = He_k(x[i])`` for k=0..max_degree using the He recurrence."""
    n = x.shape[0]
    nc = max_degree + 1
    B = np.empty((n, nc), dtype=np.float64)
    for i in prange(n):
        xi = x[i]
        B[i, 0] = 1.0
        if nc > 1:
            B[i, 1] = xi
            for k in range(2, nc):
                B[i, k] = xi * B[i, k - 1] - (k - 1) * B[i, k - 2]
    return B


@njit(cache=True, fastmath=True, parallel=True)
def _build_basis_legendre(x: np.ndarray, max_degree: int) -> np.ndarray:
    """Build ``B[i, k] = P_k(x[i])`` via Bonnet's recurrence."""
    n = x.shape[0]
    nc = max_degree + 1
    B = np.empty((n, nc), dtype=np.float64)
    for i in prange(n):
        xi = x[i]
        B[i, 0] = 1.0
        if nc > 1:
            B[i, 1] = xi
            for k in range(2, nc):
                inv_k = 1.0 / k
                two_km1 = 2 * k - 1
                km1 = k - 1
                B[i, k] = (two_km1 * xi * B[i, k - 1] - km1 * B[i, k - 2]) * inv_k
    return B


@njit(cache=True, fastmath=True, parallel=True)
def _build_basis_chebyshev(x: np.ndarray, max_degree: int) -> np.ndarray:
    """Build ``B[i, k] = T_k(x[i])`` via T_{k+1} = 2x*T_k - T_{k-1}."""
    n = x.shape[0]
    nc = max_degree + 1
    B = np.empty((n, nc), dtype=np.float64)
    for i in prange(n):
        xi = x[i]
        two_xi = 2.0 * xi
        B[i, 0] = 1.0
        if nc > 1:
            B[i, 1] = xi
            for k in range(2, nc):
                B[i, k] = two_xi * B[i, k - 1] - B[i, k - 2]
    return B


@njit(cache=True, fastmath=True, parallel=True)
def _build_basis_laguerre(x: np.ndarray, max_degree: int) -> np.ndarray:
    """Build ``B[i, k] = L_k(x[i])`` via the Laguerre recurrence."""
    n = x.shape[0]
    nc = max_degree + 1
    B = np.empty((n, nc), dtype=np.float64)
    for i in prange(n):
        xi = x[i]
        B[i, 0] = 1.0
        if nc > 1:
            B[i, 1] = 1.0 - xi
            for k in range(2, nc):
                inv_k = 1.0 / k
                two_km1 = 2 * k - 1
                km1 = k - 1
                B[i, k] = ((two_km1 - xi) * B[i, k - 1] - km1 * B[i, k - 2]) * inv_k
    return B


_BASIS_BUILDERS = {
    "hermite": _build_basis_hermite,
    "legendre": _build_basis_legendre,
    "chebyshev": _build_basis_chebyshev,
    "laguerre": _build_basis_laguerre,
}


def build_basis_matrix(basis: str, z: np.ndarray, max_degree: int) -> np.ndarray:
    """Public dispatcher: returns ``B[i, k] = T_k(z[i])`` for the named basis.

    Raises ``KeyError`` if the basis is not one of the orthogonal-polynomial
    family supported here (hermite / legendre / chebyshev / laguerre).
    Factory-based bases (RBF / Sigmoid / Fourier / Pade) need per-feature
    eval closures and are NOT compatible with basis-matrix caching - the
    caller checks via ``basis_info.get('eval_njit_factory') is None``
    before precomputing.
    """
    builder = _BASIS_BUILDERS.get(basis)
    if builder is None:
        raise KeyError(
            f"build_basis_matrix: basis {basis!r} not in "
            f"{sorted(_BASIS_BUILDERS)}; factory-based bases must use "
            f"the per-call eval_func path."
        )
    z_c = np.ascontiguousarray(z, dtype=np.float64)
    return builder(z_c, int(max_degree))


# Optional CUDA RawKernel backend. One thread per output element with the recurrence kept in registers.
# Wins at n >= 500k once host->device transfer is amortised.

_CUDA_AVAILABLE = False
_CUDA_KERNELS: dict = {}

try:
    import cupy as _cp  # noqa: F401
    _CUDA_AVAILABLE = True
except ImportError:
    pass


def _ensure_cuda_kernels():
    """Lazy-compile CUDA RawKernels on first use."""
    global _CUDA_KERNELS
    if _CUDA_KERNELS or not _CUDA_AVAILABLE:
        return
    import cupy as cp
    _CUDA_KERNELS["hermite"] = cp.RawKernel(r"""
extern "C" __global__
void hermeval_kernel(const double* __restrict__ x,
                     const double* __restrict__ c,
                     int nc, int n,
                     double* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double xi = x[i];
    if (nc == 0) { out[i] = 0.0; return; }
    if (nc == 1) { out[i] = c[0]; return; }
    double p_prev = 1.0, p_curr = xi;
    double s = c[0] + c[1] * p_curr;
    for (int k = 2; k < nc; ++k) {
        double p_next = xi * p_curr - (double)(k - 1) * p_prev;
        s += c[k] * p_next;
        p_prev = p_curr; p_curr = p_next;
    }
    out[i] = s;
}
""", "hermeval_kernel")
    _CUDA_KERNELS["legendre"] = cp.RawKernel(r"""
extern "C" __global__
void legval_kernel(const double* __restrict__ x,
                    const double* __restrict__ c,
                    int nc, int n,
                    double* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double xi = x[i];
    if (nc == 0) { out[i] = 0.0; return; }
    if (nc == 1) { out[i] = c[0]; return; }
    double p_prev = 1.0, p_curr = xi;
    double s = c[0] + c[1] * p_curr;
    for (int k = 2; k < nc; ++k) {
        double inv_k = 1.0 / (double)k;
        double p_next = ((double)(2 * k - 1) * xi * p_curr - (double)(k - 1) * p_prev) * inv_k;
        s += c[k] * p_next;
        p_prev = p_curr; p_curr = p_next;
    }
    out[i] = s;
}
""", "legval_kernel")
    _CUDA_KERNELS["chebyshev"] = cp.RawKernel(r"""
extern "C" __global__
void chebval_kernel(const double* __restrict__ x,
                     const double* __restrict__ c,
                     int nc, int n,
                     double* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double xi = x[i];
    if (nc == 0) { out[i] = 0.0; return; }
    if (nc == 1) { out[i] = c[0]; return; }
    double p_prev = 1.0, p_curr = xi;
    double s = c[0] + c[1] * p_curr;
    for (int k = 2; k < nc; ++k) {
        double p_next = 2.0 * xi * p_curr - p_prev;
        s += c[k] * p_next;
        p_prev = p_curr; p_curr = p_next;
    }
    out[i] = s;
}
""", "chebval_kernel")
    _CUDA_KERNELS["laguerre"] = cp.RawKernel(r"""
extern "C" __global__
void lagval_kernel(const double* __restrict__ x,
                    const double* __restrict__ c,
                    int nc, int n,
                    double* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double xi = x[i];
    if (nc == 0) { out[i] = 0.0; return; }
    if (nc == 1) { out[i] = c[0]; return; }
    double p_prev = 1.0, p_curr = 1.0 - xi;
    double s = c[0] + c[1] * p_curr;
    for (int k = 2; k < nc; ++k) {
        double inv_k = 1.0 / (double)k;
        double p_next = (((double)(2 * k - 1) - xi) * p_curr - (double)(k - 1) * p_prev) * inv_k;
        s += c[k] * p_next;
        p_prev = p_curr; p_curr = p_next;
    }
    out[i] = s;
}
""", "lagval_kernel")


def _polyeval_cuda(basis: str, x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """CUDA RawKernel polynomial eval. Includes H2D + launch + D2H. Worth it only at n >= 500k (per bench_poly_eval_backends)."""
    import cupy as cp
    _ensure_cuda_kernels()
    x_gpu = cp.asarray(x, dtype=cp.float64)
    c_gpu = cp.asarray(c, dtype=cp.float64)
    n = x.shape[0]
    out_gpu = cp.empty(n, dtype=cp.float64)
    block = 256
    grid = (n + block - 1) // block
    _CUDA_KERNELS[basis](
        (grid,), (block,),
        (x_gpu, c_gpu, c_gpu.shape[0], n, out_gpu),
    )
    return cp.asnumpy(out_gpu)


# Size + hardware-aware dispatcher. Crossover points measured on this repo's reference hardware
# (Intel CPU, GTX 1050 Ti) via bench_poly_eval_backends.py (cpu numpy in/out; includes H2D for CUDA):
#   n < 50k:      njit (single-thread Horner)
#   50k <= n:     njit_par (prange) -- 1.5-2x over single-thread
#   500k <= n:    cuda_kernel if cupy available -- ~5x over njit_par
# Thresholds are conservative; on faster GPUs the CUDA crossover may be lower. Override via
# MLFRAME_POLYEVAL_BACKEND env var.

_NJIT_FUNCS = {
    "hermite": _hermeval_njit, "legendre": _legval_njit,
    "chebyshev": _chebval_njit, "laguerre": _lagval_njit,
}
_NJIT_PAR_FUNCS = {
    "hermite": _hermeval_njit_parallel, "legendre": _legval_njit_parallel,
    "chebyshev": _chebval_njit_parallel, "laguerre": _lagval_njit_parallel,
}

# Thresholds in array length n. Tunable via env var.
import os as _os
_PAR_THRESHOLD = int(_os.environ.get("MLFRAME_POLYEVAL_PAR_THRESHOLD", "50000"))
_CUDA_THRESHOLD = int(_os.environ.get("MLFRAME_POLYEVAL_CUDA_THRESHOLD", "500000"))


def polyeval_dispatch(basis: str, x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Size + hardware-aware polynomial evaluator. Routes to njit / njit_par / cuda backend based on len(x)
    and CUDA availability. Override the chosen backend via MLFRAME_POLYEVAL_BACKEND env var (njit | njit_par | cuda)."""
    forced = _os.environ.get("MLFRAME_POLYEVAL_BACKEND", "")
    n = x.shape[0]
    if forced == "njit" or n < _PAR_THRESHOLD:
        return _NJIT_FUNCS[basis](x, c)
    if (forced == "cuda" or
            (forced == "" and n >= _CUDA_THRESHOLD and _CUDA_AVAILABLE)):
        if _CUDA_AVAILABLE:
            return _polyeval_cuda(basis, x, c)
        # User asked for cuda but it isn't available -- silent fallback.
    if forced == "njit_par" or n >= _PAR_THRESHOLD:
        return _NJIT_PAR_FUNCS[basis](x, c)
    return _NJIT_FUNCS[basis](x, c)


# Polynomial basis registry. Each entry maps a name to (eval_func, preprocess_func, expected_input_distribution_doc).
# - hermite (probabilist's He_n): orthogonal under N(0, 1); preprocess = z-score.
# - legendre (P_n): orthogonal on [-1, 1] uniform weight; preprocess = min-max -> [-1, 1].
# - chebyshev (T_n): orthogonal on [-1, 1] under 1/sqrt(1-x^2), minimax / equiripple; preprocess = min-max -> [-1, 1].
# - laguerre (L_n): orthogonal on [0, +inf) under e^{-x}, best for positive exponentially-distributed data; preprocess = shift to >= 0.


def _preprocess_zscore(x):
    mean = float(np.mean(x))
    std = float(np.std(x) + 1e-12)
    return (x - mean) / std, dict(mean=mean, std=std)


def _preprocess_minmax_neg1_1(x):
    lo = float(np.min(x))
    hi = float(np.max(x))
    span = hi - lo + 1e-12
    return 2 * (x - lo) / span - 1, dict(lo=lo, hi=hi)


def _preprocess_shift_nonneg(x):
    lo = float(np.min(x))
    return x - lo + 1e-9, dict(lo=lo)


def _apply_zscore(x, params):
    return (x - params["mean"]) / max(params["std"], 1e-12)


def _apply_minmax(x, params):
    span = params["hi"] - params["lo"] + 1e-12
    return 2 * (x - params["lo"]) / span - 1


def _apply_shift(x, params):
    return x - params["lo"] + 1e-9


def _make_dispatch(name):
    """Bind the basis name into a closure matching the (x, c) -> ndarray signature of eval / eval_njit."""
    def _d(x, c):
        return polyeval_dispatch(name, x, c)
    _d.__name__ = f"_polyeval_{name}_dispatched"
    return _d


# Registry of polynomial + non-polynomial basis families. Each entry: eval (numpy), eval_njit (numba),
# eval_dispatch (size-aware router), fit/apply (preprocessing), coef_size_func, canonical_seeds_func, and
# optionally eval_njit_factory for data-dependent bases (RBF, Sigmoid). Merged from bases.EXTRA_BASES at
# import time. Module-private: external callers use optimise_hermite_pair / polyeval_dispatch.
_POLY_BASES = {
    "hermite": dict(eval=hermeval, eval_njit=_hermeval_njit,
                     eval_dispatch=None,  # populated below after dispatcher exists
                     fit=_preprocess_zscore, apply=_apply_zscore,
                     dist_note="standard Normal (z-score)"),
    "legendre": dict(eval=legval, eval_njit=_legval_njit,
                      eval_dispatch=None,
                      fit=_preprocess_minmax_neg1_1, apply=_apply_minmax,
                      dist_note="uniform on [-1, 1]"),
    "chebyshev": dict(eval=chebval, eval_njit=_chebval_njit,
                       eval_dispatch=None,
                       fit=_preprocess_minmax_neg1_1, apply=_apply_minmax,
                       dist_note="uniform on [-1, 1] with 1/sqrt(1-x^2) weight"),
    "laguerre": dict(eval=lagval, eval_njit=_lagval_njit,
                      eval_dispatch=None,
                      fit=_preprocess_shift_nonneg, apply=_apply_shift,
                      dist_note="positive on [0, +inf)"),
}
for _bn in _POLY_BASES:
    _POLY_BASES[_bn]["eval_dispatch"] = _make_dispatch(_bn)
    _POLY_BASES[_bn]["coef_size_func"] = lambda d: d + 1
    # Polynomial canonical seeds use _canonical_seeds(basis, degree) defined later; bind via late closure.
    _POLY_BASES[_bn]["canonical_seeds_func"] = None
    _POLY_BASES[_bn]["kind"] = "polynomial"


# Merge non-polynomial basis families (Fourier, RBF, Sigmoid, Pade) from bases.py. Each entry must supply
# at minimum fit/apply/coef_size_func/canonical_seeds_func and either eval_njit (data-independent) or
# eval_njit_factory(params) (data-dependent like RBF centres).
try:
    from .bases import EXTRA_BASES as _EXTRA_BASES
    for _bn, _entry in _EXTRA_BASES.items():
        _POLY_BASES[_bn] = dict(_entry)  # copy
        # Non-polynomial bases skip the size-aware CUDA dispatch (rarely n>50k); route through eval_njit.
        if "eval_njit" in _entry:
            _ev = _entry["eval_njit"]
            _POLY_BASES[_bn]["eval_dispatch"] = _ev
        elif "eval_njit_factory" in _entry:
            # Built lazily per call once params are known.
            _POLY_BASES[_bn]["eval_dispatch"] = None
        else:
            _POLY_BASES[_bn]["eval_dispatch"] = None
        _POLY_BASES[_bn].setdefault("kind", "non-polynomial")
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class HermiteResult:
    """Result of an optimisation pass for a single feature pair. Despite the legacy name, carries the result for any supported polynomial basis (``basis`` field).
    Default basis is ``"chebyshev"``; pass ``"hermite"`` for synthetic-Gaussian inputs or ``"laguerre"`` for skewed-positive distributions.
    """
    coef_a: np.ndarray
    coef_b: np.ndarray
    bin_func_name: str
    bin_func: Callable
    mi: float
    baseline_mi: float
    uplift: float
    degree_a: int
    degree_b: int
    basis: str = "chebyshev"
    # Preprocessing parameters (z-score mean/std, or min-max lo/hi, or shift lo, depending on basis).
    preprocess_a: dict = field(default_factory=dict)
    preprocess_b: dict = field(default_factory=dict)

    def transform(self, x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
        """Apply the learned polynomial-pair transformation: preprocess to basis domain, eval polynomial, combine via bin_func. njit eval is 3-6x faster than numpy at n<5000."""
        basis_info = _POLY_BASES[self.basis]
        z_a = np.ascontiguousarray(basis_info["apply"](x_a, self.preprocess_a), dtype=np.float64)
        z_b = np.ascontiguousarray(basis_info["apply"](x_b, self.preprocess_b), dtype=np.float64)
        # eval_dispatch picks njit / njit_par / cuda based on len(z_a) and CUDA availability.
        eval_dispatch = basis_info["eval_dispatch"]
        coef_a = np.ascontiguousarray(self.coef_a, dtype=np.float64)
        coef_b = np.ascontiguousarray(self.coef_b, dtype=np.float64)
        h_a = eval_dispatch(z_a, coef_a)
        h_b = eval_dispatch(z_b, coef_b)
        return self.bin_func(h_a, h_b)


def _safe_div(a, b):
    """Element-wise division with sign-stable epsilon; avoids the x_a / 0 blowup that prevents polynomials from capturing ratio targets."""
    eps = 1e-9
    return a / (b + np.sign(b) * eps + eps)


def _atan2(a, b):
    """arctan2(a, b) for angular interactions; captures targets where signal is the ANGLE of the (a, b) vector, not the magnitudes."""
    return np.arctan2(a, b)


def _log_abs_signed(a, b):
    """sign(a*b) * log(|a|+eps + |b|+eps): sign-aware log of multiplicative magnitude; handles heavy-tail multiplicative targets where polynomials lose precision."""
    eps = 1e-9
    return np.sign(a * b + eps) * (np.log(np.abs(a) + eps) + np.log(np.abs(b) + eps))


_DEFAULT_BIN_FUNCS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    # The optimizer picks the best binary func per trial via batch MI; ratios + angular + log-multiplicative
    # enable discovery of targets that pure {add, sub, mul} cannot represent.
    "div": _safe_div,
    "atan2": _atan2,
    "logabs": _log_abs_signed,
}


# Canonical-polynomial warm-start coefficients. Many real targets coincide with a canonical low-degree polynomial:
# * XOR (y = sign(x_a * x_b)) -> He_1(z_a) * He_1(z_b) = z_a * z_b, so c_a = c_b = [0, 1].
# * Saddle (y = sign(x_a^2 - x_b^2)) -> He_2(z_a) - He_2(z_b), where He_2(z) = z^2 - 1, so c_a = c_b = [-1, 0, 1].
# * Circle (y = sign(x_a^2 + x_b^2 - r^2)) -> He_2(z_a) + He_2(z_b).
# Seeding with these accelerates convergence by 1-2 generations on Gaussian-ish inputs. Canonical identities
# provided for each basis up to degree 4; returned list contains coefficient vectors of shape (degree + 1,).


def basis_route_by_moments(x: np.ndarray) -> str:
    """Pick the polynomial basis best-matching the distribution of x based on a moment fingerprint.

    Heuristics:
    * |skew| > 1.5 and one-sided support -> Laguerre (matches e^{-x} weight on [0, +inf)).
    * Bounded support (range / std < 4) -> Chebyshev (arc-sine weight + min-max preprocessing).
    * Near-Gaussian (|skew| < 0.5, |excess kurt| < 1) -> Hermite (weight N(0,1)).
    * Otherwise -> Chebyshev (empirical "never bad" default).

    Returns one of {hermite, legendre, chebyshev, laguerre}.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 30:
        return "chebyshev"
    mean = float(np.mean(x))
    std = float(np.std(x) + 1e-12)
    z = (x - mean) / std
    skew = float(np.mean(z ** 3))
    kurt_excess = float(np.mean(z ** 4)) - 3.0
    rng = float(np.max(x) - np.min(x))
    spread_ratio = rng / std
    one_sided = (np.min(x) >= 0) or (np.max(x) <= 0)
    # Heavy-tailed positive / one-sided -> Laguerre.
    if abs(skew) > 1.5 and (one_sided or skew > 0):
        return "laguerre"
    # Compact / bounded -> Chebyshev.
    if spread_ratio < 4.0:
        return "chebyshev"
    # Near-Gaussian -> Hermite.
    if abs(skew) < 0.5 and abs(kurt_excess) < 1.0:
        return "hermite"
    # Default fallback: Chebyshev (empirical winner of the rank-stability bench).
    return "chebyshev"


def _canonical_seeds(basis: str, degree: int) -> list:
    """Return canonical coefficient vectors (shape (degree+1,)) for warm-start, representing explicit low-degree polynomials."""
    seeds = []
    # Identity P_1: e_1 = [0, 1, 0, ..., 0]
    e1 = np.zeros(degree + 1, dtype=np.float64)
    if degree >= 1:
        e1[1] = 1.0
        seeds.append(e1)
    # Pure P_2 polynomial coefficient vector
    if degree >= 2:
        e2 = np.zeros(degree + 1, dtype=np.float64)
        e2[2] = 1.0
        seeds.append(e2)
        # Composite low-degree: P_0 + P_2 (captures mean + curvature)
        e02 = np.zeros(degree + 1, dtype=np.float64)
        e02[0] = -1.0
        e02[2] = 1.0
        seeds.append(e02)
    # Pure P_3
    if degree >= 3:
        e3 = np.zeros(degree + 1, dtype=np.float64)
        e3[3] = 1.0
        seeds.append(e3)
    return seeds


def detect_pair_symmetry(x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray, *,
                          discrete_target: bool = True,
                          mi_estimator: str = "plugin",
                          plugin_n_bins: int = 20) -> float:
    """Symmetry score in [0, 1] for (x_a, x_b) as predictors of y. Targets of form f(a,b)=f(b,a) score near 1; asymmetric like y=sign(a-2b) score lower.

    Combines (geometric mean of) two indicators:
    1. Marginal MI ratio min(MI(a,y), MI(b,y)) / max(...).
    2. Sub/Add MI ratio MI(|a-b|, y) / MI(a+b, y).

    Score >= 0.95: caller can constrain c_a = c_b to halve search dim. Score <= 0.7: clearly asymmetric -- per-feature basis routing matters more.
    """
    from .fe_baselines import _mi_1d
    x_a = np.asarray(x_a, dtype=np.float64)
    x_b = np.asarray(x_b, dtype=np.float64)
    # Marginal MI test
    mi_a = _mi_1d(x_a, y, discrete_target=discrete_target,
                   mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins)
    mi_b = _mi_1d(x_b, y, discrete_target=discrete_target,
                   mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins)
    big_m = max(mi_a, mi_b, 1e-12)
    small_m = min(mi_a, mi_b)
    marginal_score = small_m / big_m
    # Sub vs add test (high = symmetric)
    mi_add = _mi_1d(x_a + x_b, y, discrete_target=discrete_target,
                     mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins)
    mi_sub_abs = _mi_1d(np.abs(x_a - x_b), y, discrete_target=discrete_target,
                          mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins)
    big_d = max(mi_add, mi_sub_abs, 1e-12)
    small_d = min(mi_add, mi_sub_abs)
    diff_score = small_d / big_d if big_d > 1e-9 else 0.0
    # Geometric mean of both signals.
    return float(np.sqrt(marginal_score * diff_score))


def _l2_normalize_pair(coef_a: np.ndarray, coef_b: np.ndarray,
                        target_norm: float = 1.0) -> tuple:
    """Project (c_a, c_b) jointly to the L2 sphere (or other target_norm). Used in direction_only search to remove
    the scaling ridge that confuses TPE/CMA on XOR-like targets where MI is invariant to overall scaling
    (bf=mul) or equivariant (bf=add/sub)."""
    norm = float(np.sqrt(np.sum(coef_a ** 2) + np.sum(coef_b ** 2)))
    if norm < 1e-12:
        return coef_a, coef_b
    scale = target_norm / norm
    return coef_a * scale, coef_b * scale


def _eval_coef_pair(coef_a, coef_b, *, z_a, z_b, eval_func, bf_callables,
                     bf_names, y, y_njit, mi_estimator, plugin_n_bins,
                     n_neighbors, discrete_target, l2_penalty,
                     direction_only=False, eval_func_b=None,
                     B_a=None, B_b=None):
    """Shared inner objective: evaluate one (c_a, c_b) pair across all binary funcs; return best (regularised score, raw MI, bf idx).

    ``eval_func_b`` defaults to ``eval_func`` (single-eval). Factory bases like RBF need per-feature preprocess fns, so the caller passes a separate
    ``eval_func_b`` closure over ``preprocess_b`` to evaluate ``h_b``. Without this the b-side silently re-used preprocess_a, biasing RBF fits.

    2026-05-18 PERFORMANCE: when ``B_a`` / ``B_b`` (precomputed basis
    matrices of shape ``(n, max_degree + 1)``) are supplied, evaluation
    uses BLAS GEMV ``h = B[:, :len(c)] @ c`` instead of recomputing
    Horner per call. Builds via ``build_basis_matrix`` are done ONCE
    per pair before CMA-ES; per-trial cost drops from ~340us (njit Horner
    at n=1500) to ~30-80us (BLAS GEMV) - 4-10x speedup. Factory bases
    (RBF / Sigmoid) need per-feature preprocess closures and DON'T
    support basis-matrix caching; the caller leaves B_a / B_b at None
    for those.
    """
    if direction_only:
        coef_a, coef_b = _l2_normalize_pair(coef_a, coef_b, target_norm=1.0)
    # 2026-05-18 BASIS-MATRIX FASTPATH (kept but disabled by default):
    # ``B_a`` / ``B_b`` are optional precomputed basis matrices that allow
    # ``h = B[:, :len(c)] @ c`` (BLAS GEMV) instead of Horner. Numerical
    # identity to Horner verified to 1e-16 across all four orthogonal
    # bases. HOWEVER, measured at n=1M / subsample=200k production budget
    # (cProfile 2026-05-18): GEMV on the 1500-sample multi-fidelity
    # inner search shows ZERO measurable speedup vs the existing
    # @njit(parallel=True) Horner kernel - the JIT'd recurrence with
    # prange is already cache-friendly enough that BLAS dgemv has no
    # margin. The build cost (one ``build_basis_matrix`` call per pair
    # x per restart, ~5ms each) and the per-call ``ascontiguousarray``
    # slice copy roughly cancels the GEMV win on the 1500-sample inner
    # loop.
    #
    # Optimization left in place because it WOULD help when:
    # - ``multi_fidelity=False`` (CMA-ES on full data per call)
    # - n_full < 4000 (multi-fidelity disabled by size threshold)
    # - future popsize-batched evaluation (where many coefs share one
    #   z, GEMM ``H = B @ C.T`` would amortise the build cost)
    #
    # CALLERS PASSING B_a / B_b MUST size them to match z_a / z_b. The
    # refinement step in ``optimise_hermite_pair`` explicitly sets B_a
    # = B_b = None when switching to full-n z (line 1665) - omitting
    # that produced shape (1500,) vs (n,) errors / silent hermite=0
    # regression measured in development.
    if B_a is not None and B_b is not None:
        _Ba_slice = np.ascontiguousarray(B_a[:, :coef_a.shape[0]])
        _Bb_slice = np.ascontiguousarray(B_b[:, :coef_b.shape[0]])
        _ca = np.ascontiguousarray(coef_a, dtype=np.float64)
        _cb = np.ascontiguousarray(coef_b, dtype=np.float64)
        h_a = _Ba_slice @ _ca
        h_b = _Bb_slice @ _cb
    else:
        h_a = eval_func(z_a, coef_a)
        h_b = (eval_func_b if eval_func_b is not None else eval_func)(z_b, coef_b)
    if not (np.all(np.isfinite(h_a)) and np.all(np.isfinite(h_b))):
        return -np.inf, 0.0, -1
    cols = []
    valid_idx = []
    for k, bf in enumerate(bf_callables):
        try:
            combined = bf(h_a, h_b)
        except Exception:
            continue
        if np.all(np.isfinite(combined)):
            cols.append(combined)
            valid_idx.append(k)
    if not cols:
        return -np.inf, 0.0, -1
    X_batch = np.ascontiguousarray(np.column_stack(cols), dtype=np.float64)
    if mi_estimator == "plugin":
        if discrete_target:
            mi_arr = _plugin_mi_classif_batch_njit(X_batch, y_njit, plugin_n_bins)
        else:
            mi_arr = _plugin_mi_regression_batch_njit(X_batch, y_njit, plugin_n_bins)
    else:  # ksg
        if discrete_target:
            mi_arr = mutual_info_classif(X_batch, y, n_neighbors=n_neighbors,
                                           random_state=42, discrete_features=False)
        else:
            mi_arr = mutual_info_regression(X_batch, y, n_neighbors=n_neighbors,
                                             random_state=42, discrete_features=False)
    penalty = 0.0 if direction_only else l2_penalty * (
        float(np.sum(coef_a ** 2)) + float(np.sum(coef_b ** 2))
    )
    best_score = -np.inf
    best_raw = 0.0
    best_idx = -1
    for j, k in enumerate(valid_idx):
        raw = float(mi_arr[j])
        s = raw - penalty
        if s > best_score:
            best_score = s
            best_raw = raw
            best_idx = k
    return best_score, best_raw, best_idx


def _select_diverse_topm(history: list, top_m: int,
                            min_l2_distance: float = 0.3) -> list:
    """Greedy diverse-top-M selection from (score, raw_mi, bf_idx, coef_a, coef_b) tuples; keeps entries whose joint (L2-normalized) coef vector is >= min_l2_distance from prior kept.

    Module-private; coefficient vectors of differing lengths are zero-padded to a common axis for cross-degree comparison.
    """
    if not history:
        return []
    sorted_h = sorted(history, key=lambda r: -r[0])
    # Pad lengths to the max coef vector for cross-degree comparison.
    max_a = max(e[3].shape[0] for e in sorted_h)
    max_b = max(e[4].shape[0] for e in sorted_h)

    def _padded_vec(coef_a, coef_b):
        v = np.zeros(max_a + max_b, dtype=np.float64)
        v[: coef_a.shape[0]] = coef_a
        v[max_a : max_a + coef_b.shape[0]] = coef_b
        return v

    kept = [sorted_h[0]]
    kept_dirs = [
        _padded_vec(sorted_h[0][3], sorted_h[0][4])
        / (np.linalg.norm(_padded_vec(sorted_h[0][3], sorted_h[0][4])) + 1e-12)
    ]
    for entry in sorted_h[1:]:
        if len(kept) >= top_m:
            break
        cand_vec = _padded_vec(entry[3], entry[4])
        cn = np.linalg.norm(cand_vec) + 1e-12
        cand_dir = cand_vec / cn
        is_diverse = True
        for k_dir in kept_dirs:
            cos_sim = float(abs(np.dot(cand_dir, k_dir)))
            cos_sim = min(cos_sim, 1.0)  # numerical safety
            l2_dist = np.sqrt(max(2 * (1 - cos_sim), 0.0))
            if l2_dist < min_l2_distance:
                is_diverse = False
                break
        if is_diverse:
            kept.append(entry)
            kept_dirs.append(cand_dir)
    return kept


def _run_cma_search(*, ca_size, cb_size, coef_range, n_trials, seed,
                     direction_only, warm_start_seeds, eval_kwargs,
                     popsize=None, eval_pair_fn=None,
                     track_history=False,
                     early_stop_no_improve_gens: int | None = None):
    """CMA-ES inner loop. Returns (best_coef_a, best_coef_b, best_bf_idx, best_raw_mi, n_evals). When track_history=True, also returns the full evaluation list.

    CMA minimizes; we negate the MI score. Default popsize=max(8, min(20, n_trials // 8)) -- smaller than CMA's default to allow more generations on tight budgets.

    ``early_stop_no_improve_gens`` (2026-05-20 NEW-D): break out of the
    CMA loop when ``best_score`` has not improved for this many
    consecutive GENERATIONS (not trials). Set to None to disable.
    Useful when the warm-start seeds + early CMA generations have
    already found the optimum and the remaining budget is wasted
    exploring around it. Default None (no plateau early-stop).
    """
    import cma
    dim = ca_size + cb_size
    if popsize is None:
        popsize = max(8, min(20, n_trials // 8))
    sigma0 = (coef_range[1] - coef_range[0]) / 4.0  # ~1.0 for [-2, 2]

    # Pre-evaluate canonical warm-start seeds: cheap (single MI eval each), frequently coincide with the global
    # optimum (e.g. He_1(x_a) * He_1(x_b) = x_a * x_b is exactly XOR). Best seed becomes CMA's x0 so CMA never
    # does worse than the warm-start.
    best_score = -np.inf
    best_raw = 0.0
    best_idx = -1
    best_coefs = None
    n_evals = 0
    history = [] if track_history else None
    if warm_start_seeds:
        for ws in warm_start_seeds:
            ws = np.asarray(ws, dtype=np.float64)
            coef_a = ws[:ca_size]
            coef_b = ws[ca_size:]
            score, raw_mi, bf_idx = (eval_pair_fn or _eval_coef_pair)(
                coef_a, coef_b, direction_only=direction_only, **eval_kwargs,
            )
            n_evals += 1
            if track_history and bf_idx >= 0 and np.isfinite(score):
                history.append((float(score), float(raw_mi), int(bf_idx),
                                  coef_a.copy(), coef_b.copy()))
            if score > best_score:
                best_score = score
                best_raw = raw_mi
                best_idx = bf_idx
                if direction_only:
                    nc_a, nc_b = _l2_normalize_pair(coef_a, coef_b)
                    best_coefs = (nc_a.copy(), nc_b.copy())
                else:
                    best_coefs = (coef_a.copy(), coef_b.copy())
        # Use the best canonical seed as CMA's starting point.
        x0 = (np.concatenate([best_coefs[0], best_coefs[1]])
              if best_coefs is not None else np.zeros(dim, dtype=np.float64))
        # Tighter sigma when we already have a good seed -- exploit
        # rather than explore.
        sigma0 = sigma0 * 0.5
    else:
        x0 = np.zeros(dim, dtype=np.float64)

    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        {
            "popsize": popsize,
            "bounds": [[coef_range[0]] * dim, [coef_range[1]] * dim],
            "verbose": -9,
            "verb_disp": 0,
            "verb_log": 0,
            "seed": seed if seed > 0 else 1,
            "tolfun": 1e-6,
            "tolx": 1e-6,
        },
    )
    # Inject remaining canonical seeds into the first generation -- CMA
    # 4.x lets us replace ask()'s random samples directly.
    inject_arrays = [np.asarray(s, dtype=np.float64)
                      for s in (warm_start_seeds or [])]

    first_gen = True
    # 2026-05-20 NEW-D: plateau early-stop state. Tracks how many
    # consecutive CMA generations passed without improving best_score.
    _plateau_gens = 0
    _last_gen_best_score = best_score
    while not es.stop() and n_evals < n_trials:
        try:
            if first_gen and inject_arrays:
                solutions = es.ask()
                # Replace the last len(inject) random samples with seeds.
                k = min(len(inject_arrays), len(solutions))
                for j in range(k):
                    solutions[-(j + 1)] = inject_arrays[j]
                first_gen = False
            else:
                solutions = es.ask()
        except Exception:
            break
        scores = []
        for sol in solutions:
            if n_evals >= n_trials:
                break
            coef_a = sol[:ca_size]
            coef_b = sol[ca_size:]
            score, raw_mi, bf_idx = (eval_pair_fn or _eval_coef_pair)(
                coef_a, coef_b, direction_only=direction_only, **eval_kwargs,
            )
            n_evals += 1
            if track_history and bf_idx >= 0 and np.isfinite(score):
                history.append((float(score), float(raw_mi), int(bf_idx),
                                  coef_a.copy(), coef_b.copy()))
            if score > best_score:
                best_score = score
                best_raw = raw_mi
                best_idx = bf_idx
                # Store post-projection coefs if direction_only mode.
                if direction_only:
                    nc_a, nc_b = _l2_normalize_pair(coef_a, coef_b)
                    best_coefs = (nc_a.copy(), nc_b.copy())
                else:
                    best_coefs = (coef_a.copy(), coef_b.copy())
            scores.append(-score if np.isfinite(score) else 1e6)
        if len(scores) < len(solutions):
            scores.extend([1e6] * (len(solutions) - len(scores)))
        es.tell(solutions, scores)
        # Plateau early-stop check (after es.tell so the generation is
        # complete). Compare end-of-generation best_score to start-of-
        # generation; if no improvement, increment plateau counter.
        if early_stop_no_improve_gens and early_stop_no_improve_gens > 0:
            if best_score > _last_gen_best_score:
                _plateau_gens = 0
                _last_gen_best_score = best_score
            else:
                _plateau_gens += 1
                if _plateau_gens >= int(early_stop_no_improve_gens):
                    break
    if best_coefs is None:
        return None
    if track_history:
        return (best_coefs[0], best_coefs[1], best_idx, best_raw, n_evals,
                history)
    return (best_coefs[0], best_coefs[1], best_idx, best_raw, n_evals)


def _baseline_mi_pair(x_a, x_b, y, *, discrete_target: bool,
                        n_neighbors: int = 3, mi_estimator: str = "plugin",
                        plugin_n_bins: int = 20) -> float:
    """Identity baseline: MI of (x_a, x_b) vs target. Plug-in is 1-D-x by design so we use max(MI(x_a, y), MI(x_b, y)) (lower bound on joint MI); KSG path uses sklearn's multi-D estimator."""
    if mi_estimator == "plugin":
        # Plug-in is 1-D-x by design; use max(MI(x_a, y), MI(x_b, y)) as a lower bound on the true joint MI.
        # Conservative gate (under-estimates baseline so engineered features clear it more easily); for the
        # final uplift number the bias is consistent (same estimator on both sides of the ratio).
        x_a_arr = np.asarray(x_a, dtype=np.float64)
        x_b_arr = np.asarray(x_b, dtype=np.float64)
        if discrete_target:
            y_arr = np.asarray(y, dtype=np.int64)
            mi_a = _plugin_mi_classif_njit(x_a_arr, y_arr, plugin_n_bins)
            mi_b = _plugin_mi_classif_njit(x_b_arr, y_arr, plugin_n_bins)
        else:
            y_arr = np.asarray(y, dtype=np.float64)
            mi_a = _plugin_mi_regression_njit(x_a_arr, y_arr, plugin_n_bins)
            mi_b = _plugin_mi_regression_njit(x_b_arr, y_arr, plugin_n_bins)
        return float(max(mi_a, mi_b))
    Xn = np.column_stack([x_a, x_b])
    if discrete_target:
        return float(mutual_info_classif(Xn, y, n_neighbors=n_neighbors,
                                          random_state=42, discrete_features=False).max())
    return float(mutual_info_regression(Xn, y, n_neighbors=n_neighbors,
                                         random_state=42, discrete_features=False).max())


def _ksg_mi_1d(x: np.ndarray, y: np.ndarray, *, discrete_target: bool,
               n_neighbors: int = 3) -> float:
    """KSG MI of 1-D x with target -- used as the optimisation objective."""
    if discrete_target:
        return float(mutual_info_classif(x.reshape(-1, 1), y,
                                          n_neighbors=n_neighbors, random_state=42,
                                          discrete_features=False)[0])
    return float(mutual_info_regression(x.reshape(-1, 1), y,
                                         n_neighbors=n_neighbors, random_state=42,
                                         discrete_features=False)[0])


def optimise_hermite_pair(
    x_a: np.ndarray,
    x_b: np.ndarray,
    y: np.ndarray,
    *,
    discrete_target: bool = True,
    bin_funcs: dict = None,
    max_degree: int = 4,
    min_degree: int = 2,
    n_trials: int = 200,
    coef_range: tuple = (-2.0, 2.0),
    l2_penalty: float = 0.05,
    n_neighbors: int | None = None,
    seed: int = 42,
    sweep_degrees: bool = True,
    baseline_uplift_threshold: float = 1.01,
    early_stop_no_improve: int = 50,
    basis: str = "chebyshev",
    mi_estimator: str = "plugin",
    plugin_n_bins: int = 20,
    optimizer: str = "cma",
    warm_start: bool = True,
    direction_only: bool = False,
    multi_fidelity: bool = True,
    use_trivial_baseline: bool = True,
    precomputed_trivial_baseline: float | None = None,
    precomputed_trivial_name: str | None = None,
) -> HermiteResult | None:
    """Find polynomial coefficients c_a, c_b that maximise MI(bin_func(P(x_a, c_a), P(x_b, c_b)), y) over the requested
    Optuna/CMA budget. Standardises inputs, regularises coefficients, and only returns a result when the engineered MI
    strictly beats the identity baseline by baseline_uplift_threshold.

    Knob tuning notes
    -----------------
    * basis="chebyshev" (default) wins empirically across 12 regimes (synthetic + UCI California Housing + UCI Diabetes +
      bounded / heavy-tailed) -- never finishes last, highest minimum MI. Pass basis="hermite" for synthetic Gaussian inputs
      or basis="laguerre" for skewed-positive. See _benchmarks/bench_polynomial_bases.py.
    * l2_penalty=0.05 is good for XOR-like targets where optimum |c| is small. For radial/saddle (|c| ~ 2-3) drop to 0.01.
    * n_neighbors (KSG): None auto-picks 3 for n>=5000, 5 for n in [1000,5000), 7 for n<1000.
    * max_degree=4 covers most smooth targets. For high-frequency targets raise to 6-8 (n_trials proportionally).
    * early_stop_no_improve: stop a study early if no improvement in the last N trials.
    * mi_estimator="plugin" (default) uses an njit plug-in estimator on quantile-binned values -- ~50-100x faster than
      sklearn's KSG, rank-equivalent for optimization (constant entropy bias). Pass "ksg" for sklearn's KSG.
    * plugin_n_bins=20 (default): ~sqrt(n) rule-of-thumb; larger bins reduce bias, raise variance.

    Returns HermiteResult or None if the search failed to beat the baseline.
    """
    if mi_estimator not in ("plugin", "ksg"):
        raise ValueError(
            f"unknown mi_estimator={mi_estimator!r}; expected 'plugin' or 'ksg'"
        )
    if optimizer not in ("optuna", "cma"):
        raise ValueError(
            f"unknown optimizer={optimizer!r}; expected 'optuna' or 'cma'"
        )
    # Auto-pick n_neighbors based on n.
    n = len(y)
    if n_neighbors is None:
        if n >= 5000:
            n_neighbors = 3
        elif n >= 1000:
            n_neighbors = 5
        else:
            n_neighbors = 7
    try:
        import optuna
        from optuna.samplers import TPESampler
        # TPESampler(multivariate=True) emits ExperimentalWarning per study init; flag has been "experimental"
        # since 2020 and is the recommended setting for correlated params -- suppress the noise.
        import warnings as _w
        try:
            from optuna.exceptions import ExperimentalWarning
            _w.filterwarnings("ignore", category=ExperimentalWarning)
        except ImportError:
            pass
    except ImportError as e:
        raise ImportError(
            "optimise_hermite_pair requires the optional optuna package. "
            "Install via pip install optuna."
        ) from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    bin_funcs = bin_funcs or _DEFAULT_BIN_FUNCS

    if basis not in _POLY_BASES:
        raise ValueError(f"unknown basis {basis!r}; expected one of {list(_POLY_BASES)}")
    basis_info = _POLY_BASES[basis]

    # Preprocess inputs to the basis's natural domain.
    z_a, preprocess_a = basis_info["fit"](x_a)
    z_b, preprocess_b = basis_info["fit"](x_b)
    z_a = np.ascontiguousarray(z_a, dtype=np.float64)
    z_b = np.ascontiguousarray(z_b, dtype=np.float64)
    # Hoist size-aware dispatch out of the hot trial loop: pick the backend ONCE per call (n is fixed across trials).
    # Saves ~4us/call closure overhead, ~5ms over 1000+ trials.
    n_eval = z_a.shape[0]
    factory_top = basis_info.get("eval_njit_factory")
    if factory_top is not None:
        # Non-polynomial basis with data-dependent eval (RBF/Sigmoid). Factory is invoked below per-feature.
        eval_func = None
    elif basis in _NJIT_FUNCS:
        # Polynomial basis -- size-aware ladder applies.
        if n_eval < _PAR_THRESHOLD:
            eval_func = basis_info["eval_njit"]
        elif n_eval >= _CUDA_THRESHOLD and _CUDA_AVAILABLE:
            eval_func = basis_info["eval_dispatch"]  # cuda path
        else:
            eval_func = _NJIT_PAR_FUNCS[basis]
    else:
        # Other non-polynomial basis with simple eval_njit (Fourier, Pade).
        eval_func = basis_info["eval_njit"]

    baseline = _baseline_mi_pair(z_a, z_b, y, discrete_target=discrete_target,
                                  n_neighbors=n_neighbors,
                                  mi_estimator=mi_estimator,
                                  plugin_n_bins=plugin_n_bins)
    logger.debug(f"baseline MI(pair, y) = {baseline:.4f}")

    # Stronger gate than the identity max(MI(x_a, y), MI(x_b, y)): try trivial pair-feature transforms
    # (mul, ratio, sum_sq, atan2, ...) and use BEST trivial MI as baseline. Often a simple mul(x_a, x_b)
    # captures most of the signal a polynomial would (verified on XOR / circle / saddle / UCI).
    #
    # 2026-05-20 NEW-A: callers running multiple ``fe_smart_polynom_iters``
    # restarts per pair can pre-compute the trivial baseline once and feed
    # it in via ``precomputed_trivial_baseline`` (+ ``precomputed_trivial_name``);
    # this elides ~5x duplicated 50-150ms ``best_trivial_pair`` calls per
    # pair on the n=200k production config.
    trivial_baseline_name = precomputed_trivial_name
    if (use_trivial_baseline
            and precomputed_trivial_baseline is None):
        try:
            from .fe_baselines import best_trivial_pair
            trivial = best_trivial_pair(
                np.asarray(x_a, dtype=np.float64),
                np.asarray(x_b, dtype=np.float64), y,
                discrete_target=discrete_target,
                mi_estimator=mi_estimator,
                plugin_n_bins=plugin_n_bins,
                n_neighbors=n_neighbors,
            )
            if trivial is not None:
                trivial_baseline_name, _, trivial_mi = trivial
                if trivial_mi > baseline:
                    logger.debug(
                        f"trivial baseline {trivial_baseline_name!r} "
                        f"raises baseline from {baseline:.4f} to {trivial_mi:.4f}"
                    )
                    baseline = trivial_mi
        except Exception as e:
            logger.debug(f"trivial baseline check failed: {e}")
    elif precomputed_trivial_baseline is not None:
        # Caller supplied the precomputed value -- use it directly.
        if precomputed_trivial_baseline > baseline:
            logger.debug(
                f"trivial baseline {precomputed_trivial_name!r} "
                f"raises baseline from {baseline:.4f} to "
                f"{precomputed_trivial_baseline:.4f} (precomputed)"
            )
            baseline = float(precomputed_trivial_baseline)

    # Pre-cast y once for the njit fast path.
    if mi_estimator == "plugin":
        y_njit = (np.asarray(y, dtype=np.int64) if discrete_target
                  else np.asarray(y, dtype=np.float64))
    else:
        y_njit = None  # KSG path does not need it

    best: HermiteResult | None = None

    degree_grid = list(range(min_degree, max_degree + 1)) if sweep_degrees else [max_degree]

    bf_names_global = list(bin_funcs.keys())
    bf_callables_global = [bin_funcs[n] for n in bf_names_global]

    # Multi-fidelity subsample ladder: for large n, fit coefficients on a small subsample (saves O(n) MI work)
    # and refine on full data at the end. With 2*(d+1) <= 8 coefficients, 1500 samples is enough to estimate stably.
    n_full = z_a.shape[0]
    if multi_fidelity and n_full >= 4000:
        rng_mf = np.random.default_rng(seed if seed > 0 else 0)
        sub_idx = rng_mf.choice(n_full, size=1500, replace=False)
        z_a_search = np.ascontiguousarray(z_a[sub_idx], dtype=np.float64)
        z_b_search = np.ascontiguousarray(z_b[sub_idx], dtype=np.float64)
        y_search = (y_njit[sub_idx] if y_njit is not None else None)
        y_search_any = y[sub_idx] if isinstance(y, np.ndarray) else np.asarray(y)[sub_idx]
    else:
        z_a_search = z_a
        z_b_search = z_b
        y_search = y_njit
        y_search_any = y

    # Coef-size lookup: polynomial bases use degree + 1; non-poly bases (Fourier 2K, RBF up to 9, Pade 2p+1) override.
    coef_size_func = basis_info.get("coef_size_func", lambda d: d + 1)
    canonical_seeds_func = basis_info.get("canonical_seeds_func")

    # Factory-based bases (RBF, Sigmoid) eval depends on train-fold-fitted centres / thresholds. Build per-basis
    # eval once preprocess params are known. TODO: separate eval for x_a and x_b when factory-based.
    factory = basis_info.get("eval_njit_factory")
    if factory is not None:
        eval_func = factory(preprocess_a)
        eval_func_b = factory(preprocess_b)
    else:
        eval_func_b = eval_func

    # 2026-05-18 PERFORMANCE: precompute basis matrices once per pair for
    # BLAS GEMV fastpath. Initial 2026-05-18 measurement (different
    # hardware) found zero speedup at multi_fidelity=True scale and
    # gated the basis-matrix path OFF for that case. Re-measured
    # 2026-05-20 on current hardware (numba 0.59, MKL BLAS) at the same
    # 1500-element inner CMA-ES scale showed BLAS GEMV is **1.13-1.19x
    # faster than ``@njit(parallel=True)`` Horner** — slice-copy
    # overhead and recurrence pipelining no longer cancel. Gate flipped
    # to build B matrices for ALL polynomial bases (including under
    # multi_fidelity=True). The refinement step at the bottom of this
    # function still drops B_a / B_b before evaluating on full z (see
    # ``full_kwargs["B_a"] = None`` line below) so the
    # subsample-sized matrices never leak into the full-n evaluation.
    B_a_search = None
    B_b_search = None
    _multi_fidelity_active = bool(multi_fidelity and n_full >= 4000)
    if factory is None and basis in _BASIS_BUILDERS:
        try:
            B_a_search = build_basis_matrix(basis, z_a_search, max_degree)
            B_b_search = build_basis_matrix(basis, z_b_search, max_degree)
        except Exception as _bm_err:
            logger.debug(f"build_basis_matrix failed for {basis!r}: {_bm_err}")
            B_a_search = None
            B_b_search = None

    for degree in degree_grid:
        ca_size = coef_size_func(degree)
        cb_size = coef_size_func(degree)

        # Shared kwargs for both Optuna and CMA paths. When eval_func differs per feature (factory-based bases
        # like RBF), wrap _eval_coef_pair to use both eval_func and eval_func_b.
        if factory is not None:
            def _eval_dual(coef_a, coef_b, **kw):
                from numpy import column_stack, ascontiguousarray, all as npall, isfinite
                z_a_loc = kw["z_a"]
                z_b_loc = kw["z_b"]
                bf_call = kw["bf_callables"]
                if kw.get("direction_only"):
                    coef_a, coef_b = _l2_normalize_pair(coef_a, coef_b, 1.0)
                h_a = eval_func(z_a_loc, coef_a)
                h_b = eval_func_b(z_b_loc, coef_b)
                if not (npall(isfinite(h_a)) and npall(isfinite(h_b))):
                    return -np.inf, 0.0, -1
                cols = []
                valid_idx = []
                for k, bf in enumerate(bf_call):
                    try:
                        combined = bf(h_a, h_b)
                    except Exception:
                        continue
                    if npall(isfinite(combined)):
                        cols.append(combined)
                        valid_idx.append(k)
                if not cols:
                    return -np.inf, 0.0, -1
                X_batch = ascontiguousarray(column_stack(cols), dtype=np.float64)
                if kw["mi_estimator"] == "plugin":
                    if kw["discrete_target"]:
                        mi_arr = _plugin_mi_classif_batch_njit(X_batch, kw["y_njit"], kw["plugin_n_bins"])
                    else:
                        mi_arr = _plugin_mi_regression_batch_njit(X_batch, kw["y_njit"], kw["plugin_n_bins"])
                else:
                    if kw["discrete_target"]:
                        mi_arr = mutual_info_classif(X_batch, kw["y"], n_neighbors=kw["n_neighbors"], random_state=42, discrete_features=False)
                    else:
                        mi_arr = mutual_info_regression(X_batch, kw["y"], n_neighbors=kw["n_neighbors"], random_state=42, discrete_features=False)
                penalty = 0.0 if kw.get("direction_only") else kw["l2_penalty"] * (float(np.sum(coef_a**2)) + float(np.sum(coef_b**2)))
                best_score = -np.inf
                best_raw = 0.0
                best_idx = -1
                for j, k in enumerate(valid_idx):
                    raw = float(mi_arr[j])
                    s = raw - penalty
                    if s > best_score:
                        best_score = s
                        best_raw = raw
                        best_idx = k
                return best_score, best_raw, best_idx
            eval_pair_fn = _eval_dual
        else:
            eval_pair_fn = _eval_coef_pair

        eval_kwargs = dict(
            z_a=z_a_search, z_b=z_b_search,
            eval_func=eval_func,
            bf_callables=bf_callables_global, bf_names=bf_names_global,
            y=y_search_any, y_njit=y_search,
            mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins,
            n_neighbors=n_neighbors, discrete_target=discrete_target,
            l2_penalty=l2_penalty,
            # Precomputed basis matrices for BLAS GEMV fastpath (None when
            # factory-based basis or polynomial basis not in registry).
            B_a=B_a_search, B_b=B_b_search,
        )

        # Canonical warm-start: low-degree polynomial identities matching common targets (XOR, saddle, radial).
        # Replicate across both feature slots, then concatenate.
        warm_seeds = []
        if warm_start:
            if canonical_seeds_func is not None:
                # Non-polynomial basis ships its own canonical seeds via the registry.
                seeds_per_feature = canonical_seeds_func(degree)
            else:
                seeds_per_feature = _canonical_seeds(basis, degree)
            # Pair every seed with every other seed for c_b (limited to keep init pop small).
            for s_a in seeds_per_feature:
                for s_b in seeds_per_feature:
                    warm_seeds.append(np.concatenate([s_a, s_b]))
            # One symmetric pair (c_a = -c_b) captures antisymmetric targets like saddle.
            if seeds_per_feature:
                s = seeds_per_feature[0]
                warm_seeds.append(np.concatenate([s, -s]))

        coef_a_best = None
        coef_b_best = None
        bf_idx_best = -1
        raw_mi_best = -np.inf

        if optimizer == "cma":
            # 2026-05-20 NEW-D: translate the Optuna-trial-based
            # ``early_stop_no_improve`` knob into a CMA-generation count.
            # CMA popsize defaults to max(8, min(20, n_trials // 8)), so
            # ``early_stop_gens = max(2, early_stop_no_improve // popsize)``
            # gives a comparable "give up after X plateau trials" budget.
            # The +1 floor keeps the bound meaningful even for tiny
            # popsizes.
            _early_stop_gens = None
            if early_stop_no_improve and early_stop_no_improve < n_trials:
                _eff_popsize = max(8, min(20, n_trials // 8))
                _early_stop_gens = max(
                    2, int(early_stop_no_improve) // _eff_popsize + 1,
                )
            try:
                cma_result = _run_cma_search(
                    ca_size=ca_size, cb_size=cb_size,
                    coef_range=coef_range, n_trials=n_trials, seed=seed,
                    direction_only=direction_only,
                    warm_start_seeds=warm_seeds,
                    eval_kwargs=eval_kwargs,
                    eval_pair_fn=eval_pair_fn,
                    early_stop_no_improve_gens=_early_stop_gens,
                )
            except Exception as e:
                logger.warning("CMA-ES failed at degree %d (%s); "
                                "falling back to Optuna", degree, e)
                cma_result = None
            if cma_result is None:
                continue
            coef_a_best, coef_b_best, bf_idx_best, raw_mi_best, _ = cma_result
        else:  # optuna
            def _optuna_obj(trial, _degree=degree, _ca_size=ca_size, _cb_size=cb_size,
                            _eval_pair_fn=eval_pair_fn, _eval_kwargs=eval_kwargs):
                coef_a = np.array([
                    trial.suggest_float(f"a_{i}", *coef_range)
                    for i in range(_ca_size)
                ], dtype=np.float64)
                coef_b = np.array([
                    trial.suggest_float(f"b_{i}", *coef_range)
                    for i in range(_cb_size)
                ], dtype=np.float64)
                score, raw_mi, bf_idx = (_eval_pair_fn or _eval_coef_pair)(
                    coef_a, coef_b, direction_only=direction_only,
                    **_eval_kwargs,
                )
                if bf_idx >= 0:
                    trial.set_user_attr("bf_idx", bf_idx)
                    trial.set_user_attr("raw_mi", raw_mi)
                return score
            sampler = TPESampler(multivariate=True, seed=seed)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            # Inject canonical warm-start seeds as enqueued trials.
            if warm_seeds:
                for ws in warm_seeds[:min(8, len(warm_seeds))]:
                    params = {f"a_{i}": float(ws[i]) for i in range(ca_size)}
                    params.update({f"b_{i}": float(ws[ca_size + i])
                                    for i in range(cb_size)})
                    try:
                        study.enqueue_trial(params)
                    except Exception:
                        pass
            if early_stop_no_improve and early_stop_no_improve < n_trials:
                stop_state = {"best": -np.inf, "since_improve": 0}
                def _early_stop_cb(s, trial, _stop_state=stop_state):
                    cur_best = s.best_value if s.best_trial is not None else -np.inf
                    if cur_best > _stop_state["best"]:
                        _stop_state["best"] = cur_best
                        _stop_state["since_improve"] = 0
                    else:
                        _stop_state["since_improve"] += 1
                    if _stop_state["since_improve"] >= early_stop_no_improve:
                        s.stop()
                study.optimize(_optuna_obj, n_trials=n_trials,
                               callbacks=[_early_stop_cb],
                               show_progress_bar=False)
            else:
                study.optimize(_optuna_obj, n_trials=n_trials,
                               show_progress_bar=False)
            try:
                bf_idx_best = study.best_trial.user_attrs.get("bf_idx", -1)
                raw_mi_best = study.best_trial.user_attrs.get("raw_mi", -np.inf)
                coef_a_best = np.array(
                    [study.best_params[f"a_{i}"] for i in range(ca_size)],
                    dtype=np.float64)
                coef_b_best = np.array(
                    [study.best_params[f"b_{i}"] for i in range(cb_size)],
                    dtype=np.float64)
            except (ValueError, KeyError):
                continue

        if (coef_a_best is None or bf_idx_best < 0
                or raw_mi_best <= 0 or not np.isfinite(raw_mi_best)):
            continue

        # Multi-fidelity refinement: re-evaluate the best coef set on the FULL data for an honest gating MI.
        if multi_fidelity and n_full >= 4000:
            full_kwargs = dict(eval_kwargs)
            full_kwargs.update(z_a=z_a, z_b=z_b, y=y, y_njit=y_njit)
            # CRITICAL: B_a / B_b were precomputed on the 1500-element
            # SUBSAMPLE (z_a_search / z_b_search). Refinement runs on the
            # FULL z_a / z_b (typically 100k-1M elements). We MUST drop
            # the basis matrices here so _eval_coef_pair falls back to
            # the Horner eval_func path on full data. Without this drop,
            # h_a from ``B[:, :len(c)] @ c`` would be 1500-sized while
            # other code expects the full n - produces shape-mismatch
            # OR (silently worse) re-uses subsample-sized h_a but
            # subsample-sized MI -> CMA-ES misjudges which coef is best.
            # Discovered 2026-05-18 via in-flight VERIFY assertion.
            full_kwargs["B_a"] = None
            full_kwargs["B_b"] = None
            _, raw_mi_full, bf_idx_full = _eval_coef_pair(
                coef_a_best, coef_b_best, direction_only=direction_only,
                **full_kwargs,
            )
            if bf_idx_full >= 0 and raw_mi_full > 0:
                raw_mi_best = raw_mi_full
                bf_idx_best = bf_idx_full

        bf_name = bf_names_global[bf_idx_best]
        cand = HermiteResult(
            coef_a=coef_a_best, coef_b=coef_b_best,
            bin_func_name=bf_name, bin_func=bin_funcs[bf_name],
            mi=raw_mi_best, baseline_mi=baseline,
            uplift=raw_mi_best / max(baseline, 1e-12),
            degree_a=degree, degree_b=degree,
            basis=basis,
            preprocess_a=preprocess_a,
            preprocess_b=preprocess_b,
        )
        if best is None or cand.mi > best.mi:
            best = cand
        logger.debug(
            f"degree={degree}: best MI={raw_mi_best:.4f} (baseline {baseline:.4f}, "
            f"uplift {cand.uplift:.2f}x), bf={bf_name}"
        )

    if best is None or best.mi <= baseline * baseline_uplift_threshold:
        # Failed to beat baseline by enough -- don't recommend an engineered feature.
        return None
    return best


def optimise_pair_multimode(
    x_a: np.ndarray,
    x_b: np.ndarray,
    y: np.ndarray,
    *,
    top_m: int = 3,
    min_l2_distance: float = 0.3,
    discrete_target: bool = True,
    bin_funcs: dict = None,
    max_degree: int = 4,
    min_degree: int = 2,
    n_trials: int = 200,
    coef_range: tuple = (-2.0, 2.0),
    l2_penalty: float = 0.05,
    n_neighbors: int | None = None,
    seed: int = 42,
    sweep_degrees: bool = True,
    baseline_uplift_threshold: float = 1.01,
    basis: str = "chebyshev",
    mi_estimator: str = "plugin",
    plugin_n_bins: int = 20,
    warm_start: bool = True,
    direction_only: bool = False,
) -> list:
    """Multi-mode pair-FE: return up to top_m distinct HermiteResult objects, greedily filtered to maintain
    pair-wise L2 distance >= min_l2_distance after L2-normalisation (direction-only comparison).

    A single 2D f(x_a, x_b) can have multiple rank-1 separable approximations of similar MI; emitting all of
    them lets the downstream model exploit the multi-modal structure. Verified on Friedman1-style targets
    where MI is split across 2-3 modes; emitting all 3 raises downstream R^2 by 1-2% over single-mode FE.

    Returns list[HermiteResult] sorted by MI descending; empty if no mode beats baseline.
    """
    # Forced CMA-ES because diverse top-M needs a bag of evaluations, which CMA's population gives naturally;
    # Optuna's TPE samples are less diverse early-on (coupled by the multivariate prior).
    if bin_funcs is None:
        bin_funcs = _DEFAULT_BIN_FUNCS

    if basis not in _POLY_BASES:
        raise ValueError(f"unknown basis {basis!r}; expected one of {list(_POLY_BASES)}")
    basis_info = _POLY_BASES[basis]

    z_a, preprocess_a = basis_info["fit"](x_a)
    z_b, preprocess_b = basis_info["fit"](x_b)
    z_a = np.ascontiguousarray(z_a, dtype=np.float64)
    z_b = np.ascontiguousarray(z_b, dtype=np.float64)

    # Pick eval_func via the size-aware ladder (mirrors optimise_hermite_pair). For factory bases (RBF/Sigmoid) eval_func_b is a separate closure over
    # preprocess_b so the b-side feature evaluates with its OWN centres/thresholds; for njit polynomial bases both are the same callable.
    factory_top = basis_info.get("eval_njit_factory")
    if factory_top is not None:
        eval_func = factory_top(preprocess_a)
        eval_func_b = factory_top(preprocess_b)
    elif basis in _NJIT_FUNCS:
        n_eval = z_a.shape[0]
        if n_eval < _PAR_THRESHOLD:
            eval_func = basis_info["eval_njit"]
        elif n_eval >= _CUDA_THRESHOLD and _CUDA_AVAILABLE:
            eval_func = basis_info["eval_dispatch"]
        else:
            eval_func = _NJIT_PAR_FUNCS[basis]
        eval_func_b = eval_func
    else:
        eval_func = basis_info["eval_njit"]
        eval_func_b = eval_func

    n = len(y)
    if n_neighbors is None:
        if n >= 5000:
            n_neighbors = 3
        elif n >= 1000:
            n_neighbors = 5
        else:
            n_neighbors = 7

    coef_size_func = basis_info.get("coef_size_func", lambda d: d + 1)
    canonical_seeds_func = basis_info.get("canonical_seeds_func")

    if mi_estimator == "plugin":
        y_njit = (np.asarray(y, dtype=np.int64) if discrete_target
                  else np.asarray(y, dtype=np.float64))
    else:
        y_njit = None

    bf_names_global = list(bin_funcs.keys())
    bf_callables_global = [bin_funcs[n] for n in bf_names_global]

    baseline = _baseline_mi_pair(z_a, z_b, y, discrete_target=discrete_target,
                                   n_neighbors=n_neighbors,
                                   mi_estimator=mi_estimator,
                                   plugin_n_bins=plugin_n_bins)

    # Aggregate history across degrees, then apply diverse top-M.
    full_history = []
    degree_grid = list(range(min_degree, max_degree + 1)) if sweep_degrees else [max_degree]
    for degree in degree_grid:
        ca_size = coef_size_func(degree)
        cb_size = coef_size_func(degree)
        # ``eval_func_b`` carries the per-feature preprocess fn for factory bases (RBF). For njit bases both eval_func / eval_func_b are the same callable
        # and ``_eval_coef_pair`` falls back transparently. Plumbing this through prevents the b-side from silently re-using preprocess_a on RBF/factory fits.
        eval_kwargs = dict(
            z_a=z_a, z_b=z_b, eval_func=eval_func, eval_func_b=eval_func_b,
            bf_callables=bf_callables_global, bf_names=bf_names_global,
            y=y, y_njit=y_njit,
            mi_estimator=mi_estimator, plugin_n_bins=plugin_n_bins,
            n_neighbors=n_neighbors, discrete_target=discrete_target,
            l2_penalty=l2_penalty,
        )
        warm_seeds = []
        if warm_start:
            seeds_per_feat = (canonical_seeds_func(degree)
                               if canonical_seeds_func else _canonical_seeds(basis, degree))
            for s_a in seeds_per_feat:
                for s_b in seeds_per_feat:
                    warm_seeds.append(np.concatenate([s_a, s_b]))
            if seeds_per_feat:
                s = seeds_per_feat[0]
                warm_seeds.append(np.concatenate([s, -s]))
        try:
            r = _run_cma_search(
                ca_size=ca_size, cb_size=cb_size, coef_range=coef_range,
                n_trials=n_trials, seed=seed,
                direction_only=direction_only, warm_start_seeds=warm_seeds,
                eval_kwargs=eval_kwargs, track_history=True,
            )
        except Exception as e:
            logger.warning("CMA-ES failed in multimode degree %d: %s", degree, e)
            continue
        if r is None:
            continue
        coef_a, coef_b, bf_idx, raw_mi, _n, history = r
        # Tag history entries with degree so we can rebuild HermiteResult.
        full_history.extend([(s, mi, idx, ca, cb, degree) for s, mi, idx, ca, cb in history])

    if not full_history:
        return []

    # Diverse top-M selection (post-process).
    diverse = _select_diverse_topm(
        [(s, mi, idx, ca, cb) for s, mi, idx, ca, cb, _d in full_history],
        top_m=top_m, min_l2_distance=min_l2_distance,
    )

    # Build degree lookup back into the diverse entries (diverse only carries 5-tuples).
    deg_lookup = {(tuple(ca), tuple(cb)): d for s, mi, idx, ca, cb, d in full_history}

    results = []
    for _score, raw_mi, bf_idx, coef_a, coef_b in diverse:
        if raw_mi <= baseline * baseline_uplift_threshold:
            continue
        bf_name = bf_names_global[bf_idx]
        deg = deg_lookup.get((tuple(coef_a), tuple(coef_b)), len(coef_a) - 1)
        results.append(HermiteResult(
            coef_a=coef_a, coef_b=coef_b,
            bin_func_name=bf_name, bin_func=bin_funcs[bf_name],
            mi=raw_mi, baseline_mi=baseline,
            uplift=raw_mi / max(baseline, 1e-12),
            degree_a=deg, degree_b=deg,
            basis=basis,
            preprocess_a=preprocess_a,
            preprocess_b=preprocess_b,
        ))
    results.sort(key=lambda r: -r.mi)
    return results
