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
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.polynomial.hermite_e import hermeval  # probabilist's Hermite
from numpy.polynomial.legendre import legval
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.laguerre import lagval

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
# ``pyutilz.performance.kernel_tuning.cache`` infrastructure (already used for
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
    from .. import _hermite_fe_mi as _hfmi
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

import os as _os
from ._hermite_oracle import (  # noqa: E402,F401
    _CUDA_THRESHOLD,
    _PAR_THRESHOLD,
    _POLYEVAL_ORACLE_FN_NAME,
    _POLYEVAL_ORACLE_PARAM_SPACE,
    _lookup_polyeval_thresholds,
    _polyeval_oracle_enabled,
    _polyeval_oracle_pick_cpu_backend,
    _polyeval_size_fingerprint,
    benchmark_polyeval_cpu_backends,
    get_polyeval_oracle,
)


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


# ---------------------------------------------------------------------------
# Outlier-robust axis normalisation (gated; legacy-bit-identical on clean cols)
# ---------------------------------------------------------------------------
#
# The basis preprocessors below fit their normalisation scale (std for z-score, min/max span for min-max, min for shift)
# from RAW per-column statistics. On a heavy-tailed / outlier-contaminated column (e.g. 1-5% of values at +/-1000) the
# raw std / span blows up ~1000x, collapsing 99% of the data into a sliver near the axis centre. The engineered He_n / P_n
# transform then (a) carries an OUTLIER-INFLATED plug-in MI that can hijack selection, and (b) is SHIFT-FRAGILE -- a new
# extreme value in production moves the axis and changes every row's engineered value.
#
# Fix: estimate the scale from an INNER-QUANTILE / MAD range that excludes the contaminating tail, then CLAMP the mapped
# axis so the few clipped extreme rows land at the basis-domain edge instead of stretching the scale for everyone. The
# robust path is GATED on a cheap per-column heavy-tail detector and is byte-identical to the legacy path on clean columns
# (the gate stays OFF), so the wide byte-stability FE suite is untouched; it engages only where the raw scale is provably
# corrupted. Default ON (the fastest-correct default); set MLFRAME_ROBUST_AXIS=0 (or pass legacy params) to replay legacy.
from ._hermite_robust import (  # noqa: E402,F401
    _ROBUST_AXIS_GAP,
    _ROBUST_AXIS_K,
    _ROBUST_AXIS_MAX_FRAC,
    _ROBUST_AXIS_OUTER_K,
    _detect_heavy_tail,
    _huber_irls_lstsq,
    _ols_lstsq,
    _robust_axis_enabled,
    _robust_lo_hi,
    _robust_scale,
    _robust_warp_fit_enabled,
    fit_basis_coef_robust,
)


def _preprocess_zscore(x):
    if _robust_axis_enabled() and _detect_heavy_tail(x):
        xf = x[np.isfinite(x)]
        # Robust centre/scale from the inner-quantile core; map outliers but CLAMP to the Hermite working domain so a
        # +/-1000 spike lands at the edge rather than producing a huge He_n value that inflates MI and breaks shift-stability.
        center = float(np.median(xf))
        lo, hi = _robust_lo_hi(x)
        std = float((hi - lo) / 6.0)  # inner-quantile range ~ 6 sigma for a Gaussian core; matches z-score scale.
        std = std if std > 1e-12 else (float(np.std(xf)) + 1e-12)
        clip = 6.0  # +/-6 robust sigma covers the trimmed core; clipped extremes pin to the working-domain edge.
        z = np.clip((x - center) / std, -clip, clip)
        return z, dict(mean=center, std=std, clip=clip)
    mean = float(np.mean(x))
    std = float(np.std(x) + 1e-12)
    return (x - mean) / std, dict(mean=mean, std=std)


def _preprocess_minmax_neg1_1(x):
    if _robust_axis_enabled() and _detect_heavy_tail(x):
        # Min-max onto [-1, 1] from the inner-quantile bounds; clamp so clipped outliers pin to +/-1 (the basis domain edge)
        # instead of compressing the core toward 0. clip is implied (the [-1, 1] clamp), recorded so replay matches.
        lo, hi = _robust_lo_hi(x)
        span = hi - lo + 1e-12
        z = np.clip(2 * (x - lo) / span - 1, -1.0, 1.0)
        return z, dict(lo=lo, hi=hi, clip=1.0)
    lo = float(np.min(x))
    hi = float(np.max(x))
    span = hi - lo + 1e-12
    return 2 * (x - lo) / span - 1, dict(lo=lo, hi=hi)


def _preprocess_shift_nonneg(x):
    if _robust_axis_enabled() and _detect_heavy_tail(x):
        # Shift the inner-quantile lower bound to ~0 and clamp the upper tail to the inner-quantile range so a positive
        # spike does not push the Laguerre argument far out where L_n explodes. Upper clamp recorded for replay.
        lo, hi = _robust_lo_hi(x)
        upper = float(hi - lo)
        z = np.clip(x - lo + 1e-9, 0.0, upper + 1e-9)
        return z, dict(lo=lo, clip=upper)
    lo = float(np.min(x))
    return x - lo + 1e-9, dict(lo=lo)


def _apply_zscore(x, params):
    z = (x - params["mean"]) / max(params["std"], 1e-12)
    clip = params.get("clip")
    if clip is not None:
        z = np.clip(z, -float(clip), float(clip))
    return z


def _apply_minmax(x, params):
    span = params["hi"] - params["lo"] + 1e-12
    z = 2 * (x - params["lo"]) / span - 1
    clip = params.get("clip")
    if clip is not None:
        z = np.clip(z, -float(clip), float(clip))
    return z


def _apply_shift(x, params):
    z = x - params["lo"] + 1e-9
    clip = params.get("clip")
    if clip is not None:
        z = np.clip(z, 0.0, float(clip) + 1e-9)
    return z


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
    from ..bases import EXTRA_BASES as _EXTRA_BASES
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
    # Push the denominator at least eps away from zero in its own sign direction (eps when b==0). The prior
    # ``b + sign(b)*eps + eps`` cancelled to exactly ``b`` for negative b (no protection -> divide-by-zero blowup
    # on small negative denominators); this guarantees |denom| >= eps for every b, sign-stable on both branches.
    b = np.asarray(b, dtype=np.float64)
    denom = np.where(b >= 0, b + eps, b - eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / denom


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


# ----------------------------------------------------------------------
# Sibling-module re-exports. Big optimisation + MI clusters live in
# ``_hermite_fe_optimise.py`` and ``_hermite_fe_mi.py`` so this file
# stays below the 1k-LOC monolith threshold.
# ----------------------------------------------------------------------
from ._hermite_prewarp import (  # noqa: E402,F401
    _L2_PENALTY_SATURATION_DEFAULT,
    _canonical_seeds,
    _ksg_mi_1d,
    _l2_normalize_pair,
    _l2_penalty_value,
    apply_operand_prewarp,
    fit_operand_prewarp,
    fit_pair_prewarp_als,
    warm_start_als_seed,
)
from .._hermite_fe_optimise import (  # noqa: E402,F401
    _baseline_mi_pair, _eval_coef_pair, _run_cma_search, _select_diverse_topm, detect_pair_symmetry, optimise_hermite_pair, optimise_pair_multimode,
)
from .._hermite_fe_mi import (  # noqa: E402,F401
    _ensure_cuda_kernels, _plugin_mi_classif_batch_cuda, _plugin_mi_classif_njit, _plugin_mi_from_binned_njit, _plugin_mi_regression_njit, plugin_mi_classif_batch_dispatch, plugin_mi_classif_dispatch, plugin_mi_classif_fast,
)
