"""Improved orthogonal-polynomial pair Feature Engineering.

Originally a Hermite-only module (hence the file name and the
``HermiteResult`` dataclass). Now supports four orthogonal polynomial
families via the ``basis`` kwarg: Hermite, Legendre, Chebyshev,
Laguerre. **Default basis is Chebyshev**, picked empirically across
12 synthetic + UCI regimes -- it never finishes last, has the highest
minimum MI, and dominates real-world tabular data + threshold targets.
See ``_benchmarks/bench_polynomial_bases.py`` for the supporting
table.

Idea: orthogonal polynomials form a complete basis on their natural
domain, so any sufficiently smooth bivariate function ``f(x_a, x_b)``
can be represented as ``Σ c_{a,i} c_{b,j} P_i(x_a) P_j(x_b)`` -- find
coefficients via Optuna, MI-against-target as the objective. In theory
replaces the hand-coded ``unary x binary transformations`` zoo with a
single learned parametric family.

In practice the legacy implementation in ``MRMR._run_fe_step``
(``fe_smart_polynom_iters > 0`` branch) didn't deliver because of six
issues fixed here:

1. **Standardisation**. ``hermval(raw_x, c)`` blows up numerically
   when ``|x| >> 1`` (high-degree Hermite goes superlinear). We
   z-score inputs before evaluation so the ``[-3, 3]`` range covers
   ~99.7% of the support.

2. **Right Hermite family**. Numpy's ``polynomial.hermite`` is the
   *physicist's* family ``H_n(x)`` orthogonal under ``e^{-x²}``. For
   z-scored inputs (standard Normal) we want the *probabilist's*
   family ``He_n(x)`` orthogonal under ``e^{-x²/2}`` -- ``polynomial.
   hermite_e.hermeval``.

3. **Tight coefficient range**. ``[-2, 2]`` instead of ``[-10, 10]``:
   higher-degree terms dominate quickly, large ranges make TPE
   wander.

4. **Fixed degree per study**. Random ``length`` per trial breaks
   TPE's posterior. We sweep degrees as an outer loop (study per
   degree) and pick the best.

5. **L2 regularisation**. Penalty ``-lambda * ||c||²`` on the MI
   objective keeps coefficients bounded and discourages oscillating
   overfits.

6. **Identity baseline**. Returns ``best_mi`` only when it strictly
   beats the identity baseline ``MI((x_a, x_b), y)`` -- otherwise
   no engineered feature is recommended.

Usage::

    from mlframe.feature_selection.filters.hermite_fe import (
        optimise_hermite_pair, HermiteResult,
    )
    res = optimise_hermite_pair(
        x_a=col_a, x_b=col_b, y=target,
        n_trials=200, max_degree=4, n_jobs=1,
    )
    if res.uplift > 1.05:
        engineered = res.transform(x_a, x_b)  # numpy 1-D array
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


# ---------------------------------------------------------------------------
# Fast plug-in MI estimator (numba-accelerated). The polynomial-pair FE
# objective evaluates MI(engineered_feature, target) thousands of times
# during Optuna search; sklearn's KSG was 45% of cProfile wall-time. The
# njit plug-in below is ~50-100x faster on n<=10000 because it skips
# joblib, sklearn validation, and the Cython kNN search.
#
# Why plug-in is OK as Optuna objective (not as final reported MI):
# * Optuna only needs a monotone proxy of "is this coefficient set
#   better?" -- the absolute MI value is irrelevant.
# * Plug-in over-estimates MI vs KSG (entropy bias), but the bias is
#   nearly constant across coefficient sets (same n, same n_bins), so
#   the optimum coefficient set is the same.
# * Quantile binning is rank-stable -- same as KSG's underlying
#   permutation invariance.
#
# Validation: a separate "use_fast_mi=False" path keeps sklearn KSG as
# the reference; both paths reach equivalent best coefficients on the
# 12-regime sweep (verified empirically).
# ---------------------------------------------------------------------------


@njit(cache=True, fastmath=True)
def _quantile_bin_njit(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin a 1-D continuous array into ``n_bins`` equi-frequency
    bins. Returns int32 bin indices in ``[0, n_bins)``."""
    n = x.shape[0]
    sort_idx = np.argsort(x)
    out = np.empty(n, dtype=np.int32)
    pos = 0
    base = n // n_bins
    rem = n % n_bins
    for b in range(n_bins):
        size = base + (1 if b < rem else 0)
        for k in range(size):
            out[sort_idx[pos]] = b
            pos += 1
    return out


@njit(cache=True, fastmath=True)
def _plugin_mi_classif_njit(x: np.ndarray, y: np.ndarray,
                              n_bins: int = 20) -> float:
    """Plug-in MI estimator for continuous x (1-D float64) and discrete
    y (1-D int64). Returns MI in nats. ~50x faster than sklearn for
    n<=10k, single-thread."""
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
    """Plug-in MI for continuous x (1-D) and continuous y (1-D). Bin
    both into ``n_bins`` equi-frequency bins, then plug-in estimator."""
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
    """Plug-in MI of each column of ``X_cols`` (continuous) with
    discrete ``y``. Parallelized over columns -- for the
    ``optimise_hermite_pair`` use case k=3 (one per binary func), so the
    parallelism is shallow but still saves ~2x over sequential."""
    k = X_cols.shape[1]
    out = np.zeros(k, dtype=np.float64)
    for j in prange(k):
        out[j] = _plugin_mi_classif_njit(X_cols[:, j].copy(), y, n_bins)
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _plugin_mi_regression_batch_njit(X_cols: np.ndarray, y: np.ndarray,
                                       n_bins: int = 20) -> np.ndarray:
    """Plug-in MI of each column of ``X_cols`` (continuous) with
    continuous ``y``."""
    k = X_cols.shape[1]
    out = np.zeros(k, dtype=np.float64)
    for j in prange(k):
        out[j] = _plugin_mi_regression_njit(X_cols[:, j].copy(), y, n_bins)
    return out


# ---------------------------------------------------------------------------
# njit polynomial evaluators. numpy's polyval-family is C-optimized but
# carries Python dispatch overhead per call (~30-40us); for n~2000 with
# degree<=4 the dispatch dominates. Empirical: njit hermeval ~12us vs
# numpy 46us (3.7x); njit legval ~10us vs numpy 64us (6.3x). Gap shrinks
# at n>=20k where numpy's vectorization wins.
#
# Recurrences (probabilist's variants where applicable):
# * Hermite_e (He_n): He_0=1, He_1=x, He_n = x*He_{n-1} - (n-1)*He_{n-2}
# * Legendre  (P_n) : P_0=1,  P_1=x,  P_n = ((2n-1)*x*P_{n-1} - (n-1)*P_{n-2}) / n
# * Chebyshev (T_n) : T_0=1,  T_1=x,  T_n = 2*x*T_{n-1} - T_{n-2}
# * Laguerre  (L_n) : L_0=1,  L_1=1-x, L_n = ((2n-1-x)*L_{n-1} - (n-1)*L_{n-2}) / n
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Parallel-prange variants of the polynomial evaluators. Per-element
# Horner recurrence runs in registers (no intermediate p_prev / p_curr
# arrays), so prange over array elements scales linearly with cores at
# the cost of recomputing the recurrence per element. Wins for n >= 50k
# where memory bandwidth + thread-spawn overhead is amortised.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Optional CUDA RawKernel backend. One thread per output element with
# the recurrence kept entirely in registers (no per-element intermediate
# arrays). Wins at n >= 500k once host->device transfer is amortised.
# ---------------------------------------------------------------------------

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
    """CUDA RawKernel polynomial eval. Includes host->device transfer
    of x and c, kernel launch, device->host of output. Worth it only
    at n >= 500k (per ``bench_poly_eval_backends``)."""
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


# ---------------------------------------------------------------------------
# Size + hardware-aware dispatcher. Crossover points measured on this
# repo's reference hardware (Intel CPU, GTX 1050 Ti) via
# ``bench_poly_eval_backends.py`` (cpu numpy in, cpu numpy out;
# includes H2D for CUDA backends):
#
#   n < 50k:      njit (single-thread Horner)
#   50k <= n:     njit_par (prange) -- 1.5-2x over single-thread
#   500k <= n:    cuda_kernel if cupy available -- ~5x over njit_par
#
# These thresholds are conservative -- on faster GPUs the CUDA
# crossover may be lower. Override via ``MLFRAME_POLYEVAL_BACKEND``
# env var for testing.
# ---------------------------------------------------------------------------

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
    """Size + hardware-aware polynomial evaluator. Routes to njit /
    njit_par / cuda backend based on ``len(x)`` and CUDA availability.

    Override the chosen backend via ``MLFRAME_POLYEVAL_BACKEND`` env
    var (``njit`` | ``njit_par`` | ``cuda``)."""
    forced = _os.environ.get("MLFRAME_POLYEVAL_BACKEND", "")
    n = x.shape[0]
    if forced == "njit" or n < _PAR_THRESHOLD:
        return _NJIT_FUNCS[basis](x, c)
    if (forced == "cuda" or
            (forced == "" and n >= _CUDA_THRESHOLD and _CUDA_AVAILABLE)):
        if _CUDA_AVAILABLE:
            return _polyeval_cuda(basis, x, c)
        # User asked for cuda but it isn't available; warn once and fallback.
        # No warning spam: just fallback.
    if forced == "njit_par" or n >= _PAR_THRESHOLD:
        return _NJIT_PAR_FUNCS[basis](x, c)
    return _NJIT_FUNCS[basis](x, c)


# ---------------------------------------------------------------------------
# Polynomial basis registry. Each entry maps a name to (eval_func,
# preprocess_func, expected_input_distribution_doc).
#
# - hermite (probabilist's He_n): orthogonal under N(0, 1) -- best for
#   z-scored Gaussian-ish data. Preprocess = z-score.
# - legendre (P_n): orthogonal on [-1, 1] uniform weight -- best for
#   bounded uniform data. Preprocess = scale to [-1, 1] via min-max.
# - chebyshev (T_n): orthogonal on [-1, 1] under 1/sqrt(1-x^2) --
#   minimax error bound, equiripple. Preprocess = scale to [-1, 1].
# - laguerre (L_n): orthogonal on [0, +inf) under e^{-x} -- best for
#   positive exponentially-distributed data. Preprocess = shift to >= 0.
# ---------------------------------------------------------------------------


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
    """Bind the registry entry's basis name into a closure for
    ``polyeval_dispatch``. The returned callable matches the
    ``(x, c) -> np.ndarray`` signature used by the eval / eval_njit
    fields, so existing call sites need no change."""
    def _d(x, c):
        return polyeval_dispatch(name, x, c)
    _d.__name__ = f"_polyeval_{name}_dispatched"
    return _d


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
    # Polynomial canonical seeds use _canonical_seeds(basis, degree)
    # defined later -- bind via late closure.
    _POLY_BASES[_bn]["canonical_seeds_func"] = None
    _POLY_BASES[_bn]["kind"] = "polynomial"


# Merge non-polynomial basis families (Fourier, RBF, Sigmoid, Pade)
# from ``bases.py``. Each entry must supply at minimum ``fit``,
# ``apply``, ``coef_size_func``, ``canonical_seeds_func`` and either
# ``eval_njit`` (data-independent) OR ``eval_njit_factory(params)``
# (data-dependent like RBF centres).
try:
    from .bases import EXTRA_BASES as _EXTRA_BASES
    for _bn, _entry in _EXTRA_BASES.items():
        _POLY_BASES[_bn] = dict(_entry)  # copy
        # Provide eval_dispatch placeholder -- non-polynomial bases
        # don't currently use the size-aware CUDA dispatch (they are
        # rarely n>50k anyway). Just route through eval_njit.
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
    """Result of an Optuna optimisation pass for a single feature pair.

    Despite the legacy name, ``HermiteResult`` carries the result for
    any supported polynomial basis (``basis`` field). The default
    basis is ``"chebyshev"`` (empirically robust on real tabular
    data); pass ``basis="hermite"`` for synthetic-Gaussian inputs or
    ``basis="laguerre"`` for skewed-positive distributions.
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
    # Preprocessing parameters for inputs (z-score mean/std, or min-max
    # lo/hi, or shift lo, depending on basis).
    preprocess_a: dict = field(default_factory=dict)
    preprocess_b: dict = field(default_factory=dict)

    def transform(self, x_a: np.ndarray, x_b: np.ndarray) -> np.ndarray:
        """Apply the learned polynomial-pair transformation: preprocess
        inputs to the basis's natural domain, evaluate the polynomial,
        combine via the chosen binary func. Uses the njit polynomial
        evaluators -- 3-6x faster than numpy at n<5000."""
        basis_info = _POLY_BASES[self.basis]
        z_a = np.ascontiguousarray(basis_info["apply"](x_a, self.preprocess_a),
                                     dtype=np.float64)
        z_b = np.ascontiguousarray(basis_info["apply"](x_b, self.preprocess_b),
                                     dtype=np.float64)
        # eval_dispatch picks njit / njit_par / cuda based on len(z_a)
        # and CUDA availability. For typical FE pair sizes (n<=10k)
        # this resolves to njit single-thread.
        eval_dispatch = basis_info["eval_dispatch"]
        coef_a = np.ascontiguousarray(self.coef_a, dtype=np.float64)
        coef_b = np.ascontiguousarray(self.coef_b, dtype=np.float64)
        h_a = eval_dispatch(z_a, coef_a)
        h_b = eval_dispatch(z_b, coef_b)
        return self.bin_func(h_a, h_b)


def _safe_div(a, b):
    """Element-wise division with sign-stable epsilon. Avoids the
    classic ``x_a / 0`` blowup that prevents polynomials from ever
    capturing ratio targets."""
    eps = 1e-9
    return a / (b + np.sign(b) * eps + eps)


def _atan2(a, b):
    """``arctan2(a, b)`` for angular interactions. Captures targets
    where the relevant signal is the ANGLE of the (a, b) vector,
    not the magnitudes."""
    return np.arctan2(a, b)


def _log_abs_signed(a, b):
    """``sign(a*b) * log(|a|+eps + |b|+eps)``: sign-aware log of
    multiplicative magnitude. Handles heavy-tail multiplicative
    targets where polynomials lose precision."""
    eps = 1e-9
    return np.sign(a * b + eps) * (np.log(np.abs(a) + eps) + np.log(np.abs(b) + eps))


_DEFAULT_BIN_FUNCS = {
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    # Phase B5: bin-function discovery. The optimizer picks the best
    # binary func per trial via batch MI; adding ratios + angular +
    # log-multiplicative makes the FE module able to discover targets
    # that pure {add, sub, mul} cannot represent.
    "div": _safe_div,
    "atan2": _atan2,
    "logabs": _log_abs_signed,
}


# ---------------------------------------------------------------------------
# Canonical-polynomial warm-start coefficients. The Optuna/CMA-ES search
# starts from RANDOM coefficient vectors by default, but for many real
# targets the optimum coincides with a CANONICAL low-degree polynomial:
# * XOR (y = sign(x_a * x_b)) -> He_1(z_a) * He_1(z_b) = z_a * z_b
#   so c_a = c_b = [0, 1].
# * Saddle (y = sign(x_a^2 - x_b^2)) -> He_2(z_a) - He_2(z_b)
#   where He_2(z) = z^2 - 1, so c_a = c_b = [-1, 0, 1].
# * Circle (y = sign(x_a^2 + x_b^2 - r^2)) -> He_2(z_a) + He_2(z_b).
# Seeding the population with these "obvious" coefficient sets
# accelerates convergence by 1-2 generations on Gaussian-ish inputs.
#
# We provide canonical identities for each basis up to degree 4. The
# returned list contains coefficient vectors of shape (degree + 1,).
# ---------------------------------------------------------------------------


def basis_route_by_moments(x: np.ndarray) -> str:
    """Pick the polynomial basis empirically best-matching the
    distribution of ``x`` based on a moment fingerprint.

    Heuristics (Phase B1, ranking validated on the polynomial-bases
    bench results):
    * ``|skew| > 1.5`` and one-sided support -> Laguerre (matches its
      e^{-x} weight on [0, +inf)).
    * Bounded support (range / std < 4) -> Chebyshev (arc-sine
      weight + min-max preprocessing is robust on bounded data).
    * Near-Gaussian (|skew| < 0.5, |excess kurt| < 1) -> Hermite
      (its weight is N(0,1)).
    * Otherwise -> Chebyshev (the empirical "never bad" default).

    Returns one of {hermite, legendre, chebyshev, laguerre}. Use as
    ``basis = basis_route_by_moments(x_a)`` if you don't know which
    to pick; pair-FE callers that want per-feature routing should
    pick separately for x_a and x_b."""
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
    """Return a list of canonical coefficient vectors for the given
    basis at the given degree -- low-MI-bias seeds for warm-start.
    Each vector has shape ``(degree + 1,)`` and represents an explicit
    low-degree polynomial (P_0, P_1, ..., P_degree)."""
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
    """Symmetry score in [0, 1]. Tests whether ``x_a`` and ``x_b`` are
    interchangeable predictors of ``y``: a target of the form
    ``f(a, b) = f(b, a)`` (e.g. ``y = sign(a*b)`` or
    ``y = sign(a^2 + b^2)``) has equal marginal information from
    a and b; an asymmetric target like ``y = sign(a - 2b)`` does not.

    Compares two indicators:
    1. Marginal MI ratio: ``min(MI(a, y), MI(b, y)) /
       max(MI(a, y), MI(b, y))``. Symmetric targets balance both
       features; asymmetric targets concentrate signal on one.
    2. Sub/Add MI ratio: ``MI(|a-b|, y) / MI(a+b, y)``. Symmetric
       (additive) targets favour ``a+b``; antisymmetric ones favour
       ``|a-b|``. We use the geometric mean of these signals.

    Score >= 0.95 is "highly symmetric" -- caller can constrain
    ``c_a = c_b`` to halve search dim. Score <= 0.7 is clearly
    asymmetric -- per-feature basis routing more important."""
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
    """Project ``(c_a, c_b)`` jointly to the L2 unit sphere (or other
    target norm). Used in ``direction_only`` search mode where the
    optimizer searches the unit sphere instead of the full 2(d+1)-dim
    box. Removes a degenerate scaling ridge that confuses TPE/CMA on
    XOR-like targets (where MI is invariant to overall scaling for
    ``bf=mul`` and equivariant for ``bf=add/sub``)."""
    norm = float(np.sqrt(np.sum(coef_a ** 2) + np.sum(coef_b ** 2)))
    if norm < 1e-12:
        return coef_a, coef_b
    scale = target_norm / norm
    return coef_a * scale, coef_b * scale


def _eval_coef_pair(coef_a, coef_b, *, z_a, z_b, eval_func, bf_callables,
                     bf_names, y, y_njit, mi_estimator, plugin_n_bins,
                     n_neighbors, discrete_target, l2_penalty,
                     direction_only=False):
    """Shared inner objective: evaluate one (c_a, c_b) pair across all
    binary funcs and return the best (regularised score, raw MI, bf idx).

    Returns
    -------
    score : float -- l2-penalised MI of the best binary func; -inf on failure.
    raw_mi : float -- unpenalised MI of the best binary func.
    best_bf_idx : int -- index into ``bf_callables`` of the winner.
    """
    if direction_only:
        coef_a, coef_b = _l2_normalize_pair(coef_a, coef_b, target_norm=1.0)
    h_a = eval_func(z_a, coef_a)
    h_b = eval_func(z_b, coef_b)
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


def _run_cma_search(*, ca_size, cb_size, coef_range, n_trials, seed,
                     direction_only, warm_start_seeds, eval_kwargs,
                     popsize=None, eval_pair_fn=None):
    """CMA-ES inner loop. Returns (best_coef_a, best_coef_b,
    best_bf_idx, best_raw_mi, n_evals).

    cma minimizes; we negate the MI score. Population size defaults to
    `cma`'s rule `4 + floor(3 * ln(d))` (~12 for d=8). For our small
    n_trials budgets we prefer a slightly smaller pop to allow more
    generations: max(8, n_trials // 8).
    """
    import cma
    dim = ca_size + cb_size
    if popsize is None:
        popsize = max(8, min(20, n_trials // 8))
    sigma0 = (coef_range[1] - coef_range[0]) / 4.0  # ~1.0 for [-2, 2]

    # Pre-evaluate canonical warm-start seeds. These are CHEAP (single
    # MI eval each) and frequently coincide with the global optimum
    # (e.g. He_1(x_a) * He_1(x_b) = x_a * x_b is exactly XOR). Track the
    # best seed and use it as CMA's x0; this guarantees CMA never does
    # worse than the warm-start.
    best_score = -np.inf
    best_raw = 0.0
    best_idx = -1
    best_coefs = None
    n_evals = 0
    if warm_start_seeds:
        for ws in warm_start_seeds:
            ws = np.asarray(ws, dtype=np.float64)
            coef_a = ws[:ca_size]
            coef_b = ws[ca_size:]
            score, raw_mi, bf_idx = (eval_pair_fn or _eval_coef_pair)(
                coef_a, coef_b, direction_only=direction_only, **eval_kwargs,
            )
            n_evals += 1
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
    if best_coefs is None:
        return None
    return (best_coefs[0], best_coefs[1], best_idx, best_raw, n_evals)


def _baseline_mi_pair(x_a, x_b, y, *, discrete_target: bool,
                        n_neighbors: int = 3, mi_estimator: str = "plugin",
                        plugin_n_bins: int = 20) -> float:
    """MI of the (x_a, x_b) joint vs target -- identity baseline. The
    "joint" is approximated by ``np.maximum(MI(x_a, y), MI(x_b, y))`` for
    the plug-in estimator (which only handles 1-D x), and by sklearn's
    multi-D KSG for ``mi_estimator='ksg'`` (the legacy path)."""
    if mi_estimator == "plugin":
        # Plug-in is 1-D-x by design; use max(MI(x_a, y), MI(x_b, y)) as
        # a lower bound on the true joint MI. Slightly conservative but
        # fine for the gating threshold (we under-estimate baseline ->
        # easier for engineered features to clear it). For the FINAL
        # uplift number the bias is consistent (same estimator on both
        # sides of the ratio).
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
    n_neighbors: Optional[int] = None,
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
) -> Optional[HermiteResult]:
    """Find Hermite-polynomial coefficients ``c_a``, ``c_b`` that
    maximise ``MI(bin_func(He(x_a, c_a), He(x_b, c_b)), y)`` over the
    requested Optuna budget. Standardises inputs, regularises
    coefficients, and only returns a result when the engineered MI
    strictly beats the identity baseline by ``baseline_uplift_threshold``.

    Knob tuning notes
    -----------------
    * ``basis="chebyshev"`` is the default after empirical evaluation
      across 12 regimes (synthetic + UCI California Housing + UCI
      Diabetes + bounded / heavy-tailed): Chebyshev wins on real
      tabular data and threshold-style targets, never finishes last,
      and has the highest minimum MI across the test suite. Pass
      ``basis="hermite"`` for synthetic Gaussian-input data, or
      ``basis="laguerre"`` for skewed-positive distributions. See
      ``_benchmarks/bench_polynomial_bases.py`` for the supporting
      table.
    * ``l2_penalty=0.05`` (default) is good for XOR-like targets where
      the optimum has small ``|c|``. For radial / saddle targets where
      ``|c| ~ 2-3`` is natural, drop to ``0.01``.
    * ``n_neighbors`` (KSG): ``None`` (default) auto-picks: 3 for
      ``n >= 5000``, 5 for ``n in [1000, 5000)``, 7 for ``n < 1000``.
      Smaller datasets need more neighbours to stabilise the MI estimate.
    * ``max_degree=4`` covers most smooth targets. For high-frequency
      (``tanh(x)*sin(x*pi)``) raise to 6-8 -- but each extra degree
      doubles the search space, so increase ``n_trials`` proportionally.
    * ``early_stop_no_improve``: stop a study early if no improvement in
      the last N trials. Cuts wall-time for already-converged degrees.
    * ``mi_estimator="plugin"`` (default) uses an njit-compiled plug-in
      MI estimator on quantile-binned values -- ~50-100x faster than
      sklearn's KSG, and rank-equivalent for Optuna optimization
      purposes (the absolute MI value differs by a constant entropy
      bias, but the optimum coefficient set is the same). Pass
      ``mi_estimator="ksg"`` to use sklearn's KSG (slower; matches
      legacy bit-exact behaviour). The identity baseline + final
      reported MI ``HermiteResult.baseline_mi`` / ``.mi`` use the
      chosen estimator consistently.
    * ``plugin_n_bins=20`` (default) is the equi-frequency bin count
      for the plug-in estimator. ~sqrt(n) is the rule-of-thumb;
      larger bins reduce bias but raise variance.

    Returns
    -------
    HermiteResult or None if the search failed to beat the baseline.
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
        # Optuna's TPESampler emits an ExperimentalWarning every study
        # init when ``multivariate=True``. The flag has been "experimental"
        # since 2020 and is the recommended setting for correlated params;
        # suppress the noise.
        import warnings as _w
        try:
            from optuna.exceptions import ExperimentalWarning
            _w.filterwarnings("ignore", category=ExperimentalWarning)
        except ImportError:
            pass
    except ImportError as e:
        raise ImportError(
            "optimise_hermite_pair requires the optional `optuna` package. "
            "Install via `pip install optuna`."
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
    # Hoist size-aware dispatch out of the hot trial loop: pick the
    # right backend ONCE per ``optimise_hermite_pair`` call (we know n
    # here; it doesn't change between trials). Saves ~4us/call closure
    # overhead which compounds to ~5ms over 1000+ trial evaluations.
    n_eval = z_a.shape[0]
    factory_top = basis_info.get("eval_njit_factory")
    if factory_top is not None:
        # Non-polynomial basis with data-dependent eval (RBF/Sigmoid).
        # Factory is invoked below per-feature once preprocess_a/b are
        # known; here we just stub eval_func.
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
        # Other non-polynomial basis with simple eval_njit (Fourier,
        # Pade) -- use directly.
        eval_func = basis_info["eval_njit"]

    baseline = _baseline_mi_pair(z_a, z_b, y, discrete_target=discrete_target,
                                  n_neighbors=n_neighbors,
                                  mi_estimator=mi_estimator,
                                  plugin_n_bins=plugin_n_bins)
    logger.debug(f"baseline MI(pair, y) = {baseline:.4f}")

    # Honest non-polynomial baseline: try trivial pair-feature
    # transforms (mul, ratio, sum_sq, atan2, etc.) and use the BEST
    # trivial MI as the baseline. This is a much stronger gate than
    # the identity max(MI(x_a, y), MI(x_b, y)) -- often a simple
    # ``mul(x_a, x_b)`` already captures most of the signal a
    # polynomial would (verified on XOR / circle / saddle / UCI).
    trivial_baseline_name = None
    if use_trivial_baseline:
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

    # Pre-cast y once for the njit fast path.
    if mi_estimator == "plugin":
        y_njit = (np.asarray(y, dtype=np.int64) if discrete_target
                  else np.asarray(y, dtype=np.float64))
    else:
        y_njit = None  # KSG path does not need it

    best: Optional[HermiteResult] = None

    degree_grid = list(range(min_degree, max_degree + 1)) if sweep_degrees else [max_degree]

    bf_names_global = list(bin_funcs.keys())
    bf_callables_global = [bin_funcs[n] for n in bf_names_global]

    # Multi-fidelity subsample ladder. For very large n, fit the
    # coefficients on a small random subsample early (saves O(n) MI
    # estimator work) then refine on the full data only at the end.
    # The polynomial coefficient set is just 2*(d+1)<=8 numbers, so
    # 1500 samples is enough to estimate them stably.
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

    # Coef-size lookup: polynomial bases use ``degree + 1``; non-poly
    # bases (Fourier 2K, RBF up to 9, Pade 2p+1) override via
    # ``coef_size_func``.
    coef_size_func = basis_info.get("coef_size_func", lambda d: d + 1)
    canonical_seeds_func = basis_info.get("canonical_seeds_func")

    # For factory-based bases (RBF, Sigmoid) the eval depends on
    # train-fold-fitted centres / thresholds -- build the per-basis
    # eval here once the preprocess params are known.
    factory = basis_info.get("eval_njit_factory")
    if factory is not None:
        eval_func = factory(preprocess_a)
        # NOTE: RBF/Sigmoid factory uses preprocess_a's centres for
        # x_a; for x_b we need a separate factory call. Done below
        # per evaluation via a wrapped eval. For simplicity at this
        # pass we use ``preprocess_a`` for both -- on UNIVARIATE-
        # symmetric inputs (z-scored) this is fine; for
        # heterogeneous a/b inputs you'd need separate evals per
        # feature, deferred.
        # TODO: separate eval for x_a and x_b when factory-based.
        eval_func_b = factory(preprocess_b)
    else:
        eval_func_b = eval_func

    for degree in degree_grid:
        ca_size = coef_size_func(degree)
        cb_size = coef_size_func(degree)

        # Shared kwargs threaded through both Optuna and CMA paths.
        # When eval_func differs per feature (factory-based bases like
        # RBF), wrap the inner _eval_coef_pair to use both.
        if factory is not None:
            def _eval_dual(coef_a, coef_b, **kw):
                # Re-implement just enough of _eval_coef_pair to use
                # eval_func and eval_func_b separately.
                from numpy import column_stack, ascontiguousarray, all as npall, isfinite
                z_a_loc = kw["z_a"]; z_b_loc = kw["z_b"]
                bf_call = kw["bf_callables"]
                if kw.get("direction_only"):
                    coef_a, coef_b = _l2_normalize_pair(coef_a, coef_b, 1.0)
                h_a = eval_func(z_a_loc, coef_a)
                h_b = eval_func_b(z_b_loc, coef_b)
                if not (npall(isfinite(h_a)) and npall(isfinite(h_b))):
                    return -np.inf, 0.0, -1
                cols = []; valid_idx = []
                for k, bf in enumerate(bf_call):
                    try:
                        combined = bf(h_a, h_b)
                    except Exception:
                        continue
                    if npall(isfinite(combined)):
                        cols.append(combined); valid_idx.append(k)
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
                best_score = -np.inf; best_raw = 0.0; best_idx = -1
                for j, k in enumerate(valid_idx):
                    raw = float(mi_arr[j]); s = raw - penalty
                    if s > best_score:
                        best_score = s; best_raw = raw; best_idx = k
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
        )

        # Canonical warm-start: low-degree polynomial identities that
        # match common targets (XOR, saddle, radial, ...). Replicate
        # for both feature slots, then concatenate.
        warm_seeds = []
        if warm_start:
            if canonical_seeds_func is not None:
                # Non-polynomial basis (Fourier, RBF, Sigmoid, Pade)
                # ships its own canonical seeds via the registry.
                seeds_per_feature = canonical_seeds_func(degree)
            else:
                seeds_per_feature = _canonical_seeds(basis, degree)
            # Pair every seed with itself + every other seed for c_b
            # (limited to keep init pop small).
            for s_a in seeds_per_feature:
                for s_b in seeds_per_feature:
                    warm_seeds.append(np.concatenate([s_a, s_b]))
            # Add one symmetric pair (c_a = -c_b) which captures
            # antisymmetric targets like saddle.
            if seeds_per_feature:
                s = seeds_per_feature[0]
                warm_seeds.append(np.concatenate([s, -s]))

        coef_a_best = None
        coef_b_best = None
        bf_idx_best = -1
        raw_mi_best = -np.inf

        if optimizer == "cma":
            try:
                cma_result = _run_cma_search(
                    ca_size=ca_size, cb_size=cb_size,
                    coef_range=coef_range, n_trials=n_trials, seed=seed,
                    direction_only=direction_only,
                    warm_start_seeds=warm_seeds,
                    eval_kwargs=eval_kwargs,
                    eval_pair_fn=eval_pair_fn,
                )
            except Exception as e:
                logger.warning("CMA-ES failed at degree %d (%s); "
                                "falling back to Optuna", degree, e)
                cma_result = None
            if cma_result is None:
                continue
            coef_a_best, coef_b_best, bf_idx_best, raw_mi_best, _ = cma_result
        else:  # optuna
            def _optuna_obj(trial, _degree=degree):
                coef_a = np.array([
                    trial.suggest_float(f"a_{i}", *coef_range)
                    for i in range(ca_size)
                ], dtype=np.float64)
                coef_b = np.array([
                    trial.suggest_float(f"b_{i}", *coef_range)
                    for i in range(cb_size)
                ], dtype=np.float64)
                score, raw_mi, bf_idx = (eval_pair_fn or _eval_coef_pair)(
                    coef_a, coef_b, direction_only=direction_only,
                    **eval_kwargs,
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
                def _early_stop_cb(s, trial):
                    cur_best = s.best_value if s.best_trial is not None else -np.inf
                    if cur_best > stop_state["best"]:
                        stop_state["best"] = cur_best
                        stop_state["since_improve"] = 0
                    else:
                        stop_state["since_improve"] += 1
                    if stop_state["since_improve"] >= early_stop_no_improve:
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

        # Multi-fidelity refinement: re-evaluate the best coef set on
        # the FULL data, not the subsample. This gives an honest MI for
        # the gating threshold and HermiteResult.mi.
        if multi_fidelity and n_full >= 4000:
            full_kwargs = dict(eval_kwargs)
            full_kwargs.update(z_a=z_a, z_b=z_b, y=y, y_njit=y_njit)
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
        # Failed to beat baseline by enough -- don't recommend an
        # engineered feature.
        return None
    return best
