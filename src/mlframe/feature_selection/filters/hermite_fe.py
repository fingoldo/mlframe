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




def _polyeval_cuda(basis: str, x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """CUDA RawKernel polynomial eval. Includes H2D + launch + D2H. Worth it only at n >= 500k (per bench_poly_eval_backends)."""
    import cupy as cp
    # _ensure_cuda_kernels writes into the _hermite_fe_mi module's dict;
    # read from the same source to avoid a stale local-module dict that
    # would surface as KeyError(basis) after the lazy compile succeeded.
    from . import _hermite_fe_mi as _hfmi
    _ensure_cuda_kernels()
    x_gpu = cp.asarray(x, dtype=cp.float64)
    c_gpu = cp.asarray(c, dtype=cp.float64)
    n = x.shape[0]
    out_gpu = cp.empty(n, dtype=cp.float64)
    block = 256
    grid = (n + block - 1) // block
    _hfmi._CUDA_KERNELS[basis](
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


def _lookup_polyeval_thresholds(basis: str, n: int) -> tuple[int, int]:
    """Wave 23 P2 (2026-05-20): consult kernel_tuning_cache for HW-tuned
    (par_threshold, cuda_threshold) crossovers; fall back to the
    source-code defaults (which are env-var-overridable for tests)."""
    try:
        from pyutilz.system.kernel_tuning_cache import KernelTuningCache
        _cache = KernelTuningCache.load_or_create()
        _entry = _cache.lookup("polyeval", basis=basis, n_samples=n)
        _par = int(_entry["par_threshold"]) if _entry and "par_threshold" in _entry else _PAR_THRESHOLD
        _cuda = int(_entry["cuda_threshold"]) if _entry and "cuda_threshold" in _entry else _CUDA_THRESHOLD
        return _par, _cuda
    except Exception:
        return _PAR_THRESHOLD, _CUDA_THRESHOLD


# --- Param-Oracle CPU-backend migration (proof-of-concept) -----------------
# The njit-vs-njit_par CPU crossover is the FIRST kernel_tuning_cache decision
# migrated to the ParamOracle ("learning to optimize") path. It is gated OFF by
# default (MLFRAME_POLYEVAL_ORACLE unset/"0") so the legacy threshold path stays
# byte-identical. When enabled, a ParamOracle keyed on the array-size fingerprint
# picks njit vs njit_par from RECORDED wall-times instead of a hardcoded crossover.
#
# DEFERRED: the GPU (cuda) threshold migration is NOT done -- cupy is broken on
# this dev box so the CUDA crossover cannot be benched. The cuda branch below
# stays EXACTLY on kernel_tuning_cache (_lookup_polyeval_thresholds). Migrating it
# needs a working CUDA box to populate honest wall-times; see PHASE 5 of the
# migration note. The oracle here governs ONLY the {njit, njit_par} CPU choice.

_POLYEVAL_ORACLE_FN_NAME = "polyeval_cpu_backend"
_POLYEVAL_ORACLE_PARAM_SPACE = {"backend": ["njit", "njit_par"]}
_polyeval_oracle_singleton = None


def _polyeval_oracle_enabled() -> bool:
    return _os.environ.get("MLFRAME_POLYEVAL_ORACLE", "0").strip() not in ("", "0")


def _polyeval_size_fingerprint(n: int) -> dict:
    """Stat-only fingerprint for the CPU-backend choice: array length only.
    Buckets at half-decade resolution (handled downstream by the oracle), so
    n=200 and n=210 collapse to one region but n=200 and n=500k do not."""
    return {"n": int(n), "p": 1, "dtype_kind": "f"}


def get_polyeval_oracle():
    """Lazily build (once per process) the ParamOracle that governs the CPU
    njit/njit_par backend choice. Seeds cold-start observations from the
    existing ``polyeval`` kernel_tuning_cache regions (read-only bridge) so the
    migration inherits any HW-tuned history rather than starting blind."""
    global _polyeval_oracle_singleton
    if _polyeval_oracle_singleton is not None:
        return _polyeval_oracle_singleton
    from mlframe.utils._param_oracle import ParamOracle
    oracle = ParamOracle(
        "polyeval_cpu_backend.parquet",
        param_space=_POLYEVAL_ORACLE_PARAM_SPACE,
        minimize="elapsed_s",
        mode="inference",
        min_observations=1,
    )
    # Read-only import of any njit/njit_par history the kernel cache holds. The
    # legacy 'polyeval' KTC entry stores par_threshold/cuda_threshold, not a
    # per-size backend label, so this is usually a no-op today; the bridge is
    # exercised by Layer-103 tests with synthetic KTC data and is ready for the
    # day a per-size CPU sweep is recorded.
    try:
        oracle.read_ktc_regions(
            "polyeval_cpu_backend", param_field="backend",
            fixed_fp={"p": 1, "dtype_kind": "f"},
            fn_name=_POLYEVAL_ORACLE_FN_NAME,
        )
    except Exception:
        pass
    _polyeval_oracle_singleton = oracle
    return _polyeval_oracle_singleton


def benchmark_polyeval_cpu_backends(basis: str, sizes=(200, 500_000),
                                    repeats: int = 3, oracle=None) -> dict:
    """Sweep njit vs njit_par at the given array sizes, timing each, and record
    the wall-times into the CPU-backend ParamOracle. Populates the oracle so
    later ``inference`` calls recommend the empirically faster backend per size.

    Returns ``{(n, backend): median_elapsed_s}`` for inspection. CPU-only: never
    touches the cuda path (unbenchable here)."""
    import time as _time
    if oracle is None:
        oracle = get_polyeval_oracle()
    c = np.array([0.3, -0.7, 0.2, 0.5, -0.1], dtype=np.float64)
    funcs = {"njit": _NJIT_FUNCS[basis], "njit_par": _NJIT_PAR_FUNCS[basis]}
    results: dict = {}
    for n in sizes:
        x = np.linspace(-1.0, 1.0, int(n)).astype(np.float64)
        for backend, fn in funcs.items():
            fn(x, c)  # warm the numba compile so it doesn't pollute the timing
            times = []
            for _ in range(max(1, repeats)):
                t0 = _time.perf_counter()
                fn(x, c)
                times.append(_time.perf_counter() - t0)
            med = float(sorted(times)[len(times) // 2])
            fp = _polyeval_size_fingerprint(n)
            oracle.record(fp, {"backend": backend}, {"elapsed_s": med},
                          fn_name=_POLYEVAL_ORACLE_FN_NAME)
            results[(int(n), backend)] = med
    return results


def _polyeval_oracle_pick_cpu_backend(n: int) -> str:
    """Ask the oracle which CPU backend (njit | njit_par) is faster for size
    ``n``. Falls back to ``njit`` if the oracle has no usable recommendation."""
    oracle = get_polyeval_oracle()
    fp = _polyeval_size_fingerprint(n)
    combo = oracle.recommend(fp, fn_name=_POLYEVAL_ORACLE_FN_NAME)
    backend = combo.get("backend") if isinstance(combo, dict) else None
    return backend if backend in ("njit", "njit_par") else "njit"


def polyeval_dispatch(basis: str, x: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Size + hardware-aware polynomial evaluator. Routes to njit / njit_par / cuda backend based on len(x)
    and CUDA availability. Override the chosen backend via MLFRAME_POLYEVAL_BACKEND env var (njit | njit_par | cuda).

    Crossover thresholds consult ``kernel_tuning_cache`` first
    (HW-tuned) and fall back to the source-code defaults
    (env-var-overridable for tests) when no cache entry exists.

    CPU-backend migration (opt-in): when ``MLFRAME_POLYEVAL_ORACLE`` is truthy,
    the njit-vs-njit_par CPU choice is delegated to a ParamOracle that learns the
    crossover from recorded wall-times. The cuda path is unaffected and stays on
    kernel_tuning_cache (cupy unbenchable on the dev box -- migration DEFERRED).
    Default (flag unset) is byte-identical to the legacy threshold path."""
    forced = _os.environ.get("MLFRAME_POLYEVAL_BACKEND", "")
    n = x.shape[0]
    _par_threshold, _cuda_threshold = _lookup_polyeval_thresholds(basis, n)
    if forced == "njit":
        return _NJIT_FUNCS[basis](x, c)
    # CUDA path: untouched, still kernel_tuning_cache-driven (deferred migration).
    if (forced == "cuda" or
            (forced == "" and n >= _cuda_threshold and _CUDA_AVAILABLE)):
        if _CUDA_AVAILABLE:
            return _polyeval_cuda(basis, x, c)
        # User asked for cuda but it isn't available -- silent fallback.
    if forced == "njit_par":
        return _NJIT_PAR_FUNCS[basis](x, c)
    # CPU njit/njit_par crossover: oracle-driven when enabled, else the legacy
    # hardcoded/kernel_tuning_cache threshold.
    if forced == "" and _polyeval_oracle_enabled():
        if _polyeval_oracle_pick_cpu_backend(n) == "njit_par":
            return _NJIT_PAR_FUNCS[basis](x, c)
        return _NJIT_FUNCS[basis](x, c)
    if n < _par_threshold:
        return _NJIT_FUNCS[basis](x, c)
    return _NJIT_PAR_FUNCS[basis](x, c)


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
    # z**3 / z**4 via chained multiplication: numpy ** dispatches through
    # np.power's general path even for integer exponents (~3x slower than
    # z*z*z / z2*z2; same antipattern fixed in iter138 for
    # _target_distribution_analyzer + iter129 for regression_residual_audit).
    z2 = z * z
    skew = float(np.mean(z2 * z))
    kurt_excess = float(np.mean(z2 * z2)) - 3.0
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


# Default saturation constant for the scale-invariant coefficient penalty (see
# ``_l2_penalty_value``). When ``l2_penalty_saturation > 0`` the penalty is
# ``lambda * ||c||^2 / (||c||^2 + saturation)`` -- it rises from 0 toward a
# CONSTANT ``lambda`` ceiling as ``||c||^2`` grows past ``saturation``, so a
# genuinely high-MI / high-coefficient solution is never crushed (the failure
# mode the raw ``lambda * ||c||^2`` penalty caused on the F-POLY pre-distortion
# fixture, where the true Chebyshev coefficients have ``||c||^2 ~ 86`` and the
# raw penalty ~4.3 dwarfed the MI peak ~1.5). ``saturation`` sets the coef-norm
# scale at which the penalty reaches half ``lambda``; 1.0 means small-coef noise
# solutions (||c||^2 << 1, e.g. an atan2 plateau artifact) still pay almost the
# full ``lambda``, preserving noise rejection.
_L2_PENALTY_SATURATION_DEFAULT = 1.0


def _l2_penalty_value(coef_a: np.ndarray, coef_b: np.ndarray,
                       l2_penalty: float,
                       l2_penalty_saturation: float = _L2_PENALTY_SATURATION_DEFAULT) -> float:
    """Coefficient-magnitude penalty subtracted from the raw MI objective.

    Two regimes, selected by ``l2_penalty_saturation``:

    * ``l2_penalty_saturation > 0`` (the default / recommended path): a
      SCALE-SATURATING penalty ``lambda * s / (s + sat)`` where ``s = ||c_a||^2
      + ||c_b||^2``. As ``s`` grows the penalty saturates toward the constant
      ``lambda`` instead of growing without bound, so it regularises pure noise
      (tiny ``s`` -> tiny penalty difference between candidates, plus the
      constant ceiling discourages adding magnitude for no MI gain) WITHOUT
      punishing genuinely-high-MI high-coefficient solutions. This is what lets
      the separable Chebyshev reconstruction of ``(a**3-2a)(b**2-b)`` (||c||^2
      ~ 86, MI ~ 1.5) win over the deceptive small-||c|| atan2/div plateau.

    * ``l2_penalty_saturation <= 0``: the legacy RAW penalty ``lambda *
      ||c||^2``. Kept for byte-compatibility / opt-out; this is the formula that
      crushed large-coefficient solutions.

    ``l2_penalty <= 0`` returns 0.0 in both regimes (penalty disabled).
    """
    if l2_penalty <= 0.0:
        return 0.0
    s = float(np.sum(coef_a ** 2) + np.sum(coef_b ** 2))
    if l2_penalty_saturation and l2_penalty_saturation > 0.0:
        return l2_penalty * (s / (s + l2_penalty_saturation))
    return l2_penalty * s


def warm_start_als_seed(B_a: np.ndarray, B_b: np.ndarray, y: np.ndarray,
                         *, iters: int = 3) -> tuple:
    """Per-operand warm-start coefficients for the multiplicative pair model
    ``y ~ f(x_a) * g(x_b)`` via a rank-1 alternating-least-squares (ALS) sweep
    in the orthogonal-polynomial basis.

    ``B_a`` / ``B_b`` are precomputed basis matrices ``B[i, k] = T_k(z[i])`` of
    shape ``(n, degree + 1)`` (see :func:`build_basis_matrix`). Returns
    ``(coef_a, coef_b)`` -- each length ``degree + 1`` -- such that ``B_a @
    coef_a`` and ``B_b @ coef_b`` are the rank-1 separable factors best fitting
    the centred target.

    Why ALS and not two independent 1-D fits: for a centred product target the
    marginal ``E[y | x_b]`` is ~ ``g(x_b) * E[f(x_a)] ~ 0``, so an independent
    1-D least-squares fit of ``y`` on ``B_b`` recovers almost nothing on the
    b-side (measured corr 0.49 vs Q on the F-POLY fixture). One ALS sweep -- fit
    ``f`` given the current ``g`` by regressing ``y`` on ``B_a`` scaled
    column-wise by ``g``, then symmetrically -- recovers BOTH factors exactly
    (corr 1.0 each on F-POLY) in three cheap ``lstsq`` solves. This is the
    highest-leverage, near-free warm start: it lands the joint optimiser
    directly in the true (large-coefficient) basin that CMA-ES otherwise never
    finds from the canonical unit-magnitude seeds.

    The returned coefficient SCALE is arbitrary for a ``mul`` combination (MI is
    scale-invariant under ``mul``); the magnitude is split between the two
    factors by the ALS normalisation and is intentionally NOT projected -- the
    saturating penalty (:func:`_l2_penalty_value`) makes that scale harmless.

    Returns ``(None, None)`` if the target has no variance or ``lstsq`` fails.
    """
    yc = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    yc = yc - yc.mean()
    if float(np.std(yc)) < 1e-12:
        return None, None
    try:
        # Initialise g(b) from a plain 1-D least-squares fit on the b-basis.
        cb, *_ = np.linalg.lstsq(B_b, yc, rcond=None)
        g = B_b @ cb
        ca = None
        for _ in range(max(1, int(iters))):
            g_norm = g / (float(np.std(g)) + 1e-12)
            ca, *_ = np.linalg.lstsq(B_a * g_norm[:, None], yc, rcond=None)
            f = B_a @ ca
            f_norm = f / (float(np.std(f)) + 1e-12)
            cb, *_ = np.linalg.lstsq(B_b * f_norm[:, None], yc, rcond=None)
            g = B_b @ cb
        if ca is None or not (np.all(np.isfinite(ca)) and np.all(np.isfinite(cb))):
            return None, None
        return np.ascontiguousarray(ca, dtype=np.float64), np.ascontiguousarray(cb, dtype=np.float64)
    except (np.linalg.LinAlgError, ValueError):
        return None, None










def fit_operand_prewarp(
    x: np.ndarray,
    y: np.ndarray,
    *,
    basis: str = "chebyshev",
    max_degree: int = 4,
) -> dict | None:
    """Fit a per-operand 1-D pre-warp ``f(x)`` that linearises the operand's
    relationship to the (possibly non-monotone) target ``y`` via a single
    orthogonal-polynomial least-squares solve.

    This is the lightest sufficient pre-warp for the *unary/binary* pair search:
    where a single library unary (``sqr``, ``log``, ...) cannot express a
    within-operand polynomial such as ``a**3 - 2a``, an orthogonal-series fit of
    ``y ~ poly(x)`` can. It is deliberately the SAME 1-D machinery the
    orthogonal-poly path warm-starts from (:func:`warm_start_als_seed` is its
    rank-1 ALS sibling); exposing it here lets BOTH paths share one
    implementation rather than duplicating the basis fit.

    The fit consumes ``y`` (it is supervised, like the MI scoring), but the
    returned spec is a CLOSED-FORM function of ``x`` alone -- the stored
    ``coef`` + basis ``preprocess`` params reproduce ``f(x)`` deterministically
    at transform() time with NO ``y`` reference (leak-safe replay).

    Returns a dict ``{basis, degree, coef, preprocess}`` consumable by
    :func:`apply_operand_prewarp`, or ``None`` if the target / operand has no
    usable variance or the solve fails.
    """
    bi = _POLY_BASES.get(basis)
    if bi is None or bi.get("kind") != "polynomial":
        # Pre-warp only defined for the orthogonal-polynomial families (closed-
        # form basis matrix + apply params); non-polynomial bases need per-call
        # eval closures that are not replay-portable here.
        return None
    xf = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    yf = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    if xf.size == 0 or float(np.std(xf)) < 1e-12 or float(np.std(yf)) < 1e-12:
        return None
    deg = max(1, int(max_degree))
    z, params = bi["fit"](xf)
    z = np.ascontiguousarray(z, dtype=np.float64)
    try:
        B = build_basis_matrix(basis, z, deg)
        yc = yf - yf.mean()
        coef, *_ = np.linalg.lstsq(B, yc, rcond=None)
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(coef)):
        return None
    return {
        "basis": str(basis),
        "degree": int(deg),
        "coef": np.ascontiguousarray(coef, dtype=np.float64),
        "preprocess": dict(params),
    }


def fit_pair_prewarp_als(
    x_a: np.ndarray,
    x_b: np.ndarray,
    y: np.ndarray,
    *,
    basis: str = "chebyshev",
    max_degree: int = 4,
) -> tuple:
    """Jointly fit a per-operand pre-warp for BOTH sides of a pair via the rank-1
    ALS sweep (:func:`warm_start_als_seed`), returning ``(spec_a, spec_b)`` each
    consumable by :func:`apply_operand_prewarp`.

    Why joint ALS and not two independent 1-D fits (:func:`fit_operand_prewarp`):
    for a centred product target ``y ~ P(a) * Q(b)`` the marginal ``E[y | b] ~
    Q(b) * E[P(a)] ~ 0``, so an INDEPENDENT 1-D fit of ``y`` on the b-basis
    recovers almost nothing on the b-side (measured corr ~0.1 on the F-POLY
    fixture). The ALS sweep alternates -- fit ``f`` given the current ``g``, then
    ``g`` given ``f`` -- and recovers BOTH factors (corr ~1.0 each). This is the
    SAME mechanism the orthogonal-poly path warm-starts the joint CMA optimiser
    with; reusing it here gives the elementary unary/binary search a genuine
    per-operand non-monotone pre-warp for product-structured pairs.

    Returns ``(None, None)`` on no-variance / solve failure or a non-polynomial
    basis (the closed-form replay needs the polynomial basis-matrix path).
    """
    bi = _POLY_BASES.get(basis)
    if bi is None or bi.get("kind") != "polynomial":
        return None, None
    xa = np.ascontiguousarray(np.asarray(x_a, dtype=np.float64))
    xb = np.ascontiguousarray(np.asarray(x_b, dtype=np.float64))
    yf = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
    if (xa.size == 0 or float(np.std(xa)) < 1e-12 or float(np.std(xb)) < 1e-12
            or float(np.std(yf)) < 1e-12):
        return None, None
    deg = max(1, int(max_degree))
    za, pa = bi["fit"](xa)
    zb, pb = bi["fit"](xb)
    za = np.ascontiguousarray(za, dtype=np.float64)
    zb = np.ascontiguousarray(zb, dtype=np.float64)
    try:
        Ba = build_basis_matrix(basis, za, deg)
        Bb = build_basis_matrix(basis, zb, deg)
        coef_a, coef_b = warm_start_als_seed(Ba, Bb, yf, iters=3)
    except (np.linalg.LinAlgError, ValueError):
        return None, None
    if coef_a is None or coef_b is None:
        return None, None
    spec_a = {"basis": str(basis), "degree": int(deg),
              "coef": np.ascontiguousarray(coef_a, dtype=np.float64), "preprocess": dict(pa)}
    spec_b = {"basis": str(basis), "degree": int(deg),
              "coef": np.ascontiguousarray(coef_b, dtype=np.float64), "preprocess": dict(pb)}
    return spec_a, spec_b


def apply_operand_prewarp(x: np.ndarray, spec: dict) -> np.ndarray:
    """Replay a per-operand pre-warp ``f(x)`` from a spec produced by
    :func:`fit_operand_prewarp`. Closed-form in ``x`` (uses the stored basis
    ``preprocess`` params + ``coef``); no ``y`` reference, so transform()-time
    replay is bit-identical to fit time given the same ``x``."""
    basis = str(spec["basis"])
    bi = _POLY_BASES[basis]
    xf = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    z = np.ascontiguousarray(bi["apply"](xf, dict(spec["preprocess"])), dtype=np.float64)
    coef = np.ascontiguousarray(spec["coef"], dtype=np.float64)
    out = bi["eval_dispatch"](z, coef)
    return np.asarray(out, dtype=np.float64).reshape(-1)


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


# ----------------------------------------------------------------------
# Sibling-module re-exports. Big optimisation + MI clusters live in
# ``_hermite_fe_optimise.py`` and ``_hermite_fe_mi.py`` so this file
# stays below the 1k-LOC monolith threshold.
# ----------------------------------------------------------------------
from ._hermite_fe_optimise import (  # noqa: E402,F401
    _baseline_mi_pair, _eval_coef_pair, _run_cma_search, _select_diverse_topm, detect_pair_symmetry, optimise_hermite_pair, optimise_pair_multimode,
)
from ._hermite_fe_mi import (  # noqa: E402,F401
    _ensure_cuda_kernels, _plugin_mi_classif_batch_cuda, _plugin_mi_classif_njit, _plugin_mi_from_binned_njit, _plugin_mi_regression_njit, plugin_mi_classif_batch_dispatch, plugin_mi_classif_dispatch, plugin_mi_classif_fast,
)
