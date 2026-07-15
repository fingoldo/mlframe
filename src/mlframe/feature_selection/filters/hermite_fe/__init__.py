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
from typing import Any, Callable

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
        """No-numba fallback: return the function unchanged (bare-decorator form) or a pass-through decorator (parametrized form)."""
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def deco(fn):
            """Pass-through decorator used when ``njit`` is called with arguments but numba is unavailable."""
            return fn
        return deco
    def prange(n):
        """No-numba fallback: plain ``range``."""
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
    """Quantile-bin a 1-D continuous array into n_bins equi-frequency bins. Returns int32 bin indices in [0, n_bins).

    bench-attempt-rejected: kth-order-statistic edges via numba np.partition + searchsorted (the CPU analogue
    of the GPU radix-edge binner) measured 2.6x SLOWER than this argsort form (99401: 17.8ms vs 46.5ms) --
    numba's np.partition re-copies the array per edge, and no multi-kth variant exists. argsort stays optimal
    (third measured rejection at this site; see also the two in _plugin_mi_classif_batch_njit)."""
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
def _plugin_mi_classif_batch_njit(X_cols: np.ndarray, y: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Plug-in MI of each column of X_cols (continuous) with discrete y. Parallel over columns; for k~3 (one per binary func)
    parallelism is shallow but still saves ~2x over sequential."""
    # The per-column ``.copy()`` materialises a CONTIGUOUS column before numba's argsort (inside
    # ``_plugin_mi_classif_njit`` -> ``_quantile_bin_njit``). It looks like a hoistable per-thread alloc, but it is the
    # fast layout: numba's ``np.argsort`` on a contiguous buffer beats argsort on a strided ``X_cols[:, j]`` view.
    # bench-attempt-rejected (2026-06-20): two bit-identical reshapes both LOSE at the canonical 30k screen scale --
    #   (a) one-shot ``np.ascontiguousarray(X_cols.T)`` + contiguous-row argsort: 0.84-0.89x (the serial full-matrix
    #       transpose costs more than the k strided copies it removes);
    #   (b) drop the copy, pass ``X_cols[:, j]`` strided into argsort: 0.71-0.89x (strided argsort slower).
    #   Both reach ~1.02-1.06x only at n=100k, below the noise floor. The numpy-argsort split
    #   (``plugin_mi_classif_batch_fast``) is faster at tiny k but is NOT bit-identical here (numpy quicksort vs numba
    #   argsort break ties differently -> ~1e-5 MI drift -> selection-risky), so it is ruled out for the canonical fit.
    # bench-attempt-rejected (2026-06-29): this function is ALREADY parallel over the embarrassingly-parallel
    #   axis (columns, race-free: each ``out[j]`` is written only by its own iteration). Synchronized micro-bench
    #   (8 threads) shows the inner per-column ``_quantile_bin_njit`` argsort is ~97% of single-column time and is
    #   memory-bandwidth-bound with NO internal parallelism, so column-parallelism is the only lever and it is
    #   already taken: the heavy F2 calls are k=527 (4.6s @4thr -> 3.4s @8thr = 1.35x, scales with threads). Two
    #   dead ends measured: (a) swapping numba argsort for numpy's faster SIMD argsort breaks bit-identity on TIES
    #   (card=5 @1M -> 283k bin diffs -> partition shift -> selection-risky); (b) a k==1 fast-path skipping prange
    #   saves only ~6ms/call @1M (4%, negative at 200k) -> ~70ms over the 24 k=1 calls = sub-0.1% of a ~90s fit,
    #   not worth the branch. Left as-is: optimal safe form.
    k = X_cols.shape[1]
    out = np.zeros(k, dtype=np.float64)
    for j in prange(k):
        out[j] = _plugin_mi_classif_njit(X_cols[:, j].copy(), y, n_bins)
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _plugin_mi_classif_batch_rows_njit(X_rows: np.ndarray, y: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """ROW-major twin of ``_plugin_mi_classif_batch_njit``: takes (k, n) with each candidate column stored as a
    CONTIGUOUS row, so the per-column ``.copy()`` the (n, k) form needs before numba argsort disappears
    entirely. Bit-identical MI (same values, same contiguous argsort). Built for _eval_coef_pair_batch, which
    now fills its bf-combination batch row-major from the start (contiguous writes, no strided column stores).
    NOT a replacement for the (n, k) form: callers holding naturally column-major data keep the original
    (the measured-rejected transpose of an EXISTING (n, k) matrix is a different, losing lever -- see above)."""
    k = X_rows.shape[0]
    out = np.zeros(k, dtype=np.float64)
    for j in prange(k):
        out[j] = _plugin_mi_classif_njit(X_rows[j], y, n_bins)
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
        out[sort_idx[pos : pos + size]] = b
        pos += size
    return out


def plugin_mi_classif_batch_fast(X_cols: np.ndarray, y: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Batch variant of :func:`plugin_mi_classif_fast`. Does argsort + bin
    assignment per column in pure numpy then dispatches the histogram
    math to the njit kernel. Wins over ``_plugin_mi_classif_batch_njit``
    when k is small (<= ~10) because the prange-overhead and per-thread
    argsort cost dominate at low column counts.
    """
    _n, k = X_cols.shape
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


def _plugin_mi_classif_cuda(x: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
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
def _plugin_mi_regression_batch_njit(X_cols: np.ndarray, y: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Plug-in MI of each column of X_cols (continuous) with continuous y."""
    k = X_cols.shape[1]
    out = np.zeros(k, dtype=np.float64)
    for j in prange(k):
        out[j] = _plugin_mi_regression_njit(X_cols[:, j].copy(), y, n_bins)
    return out


# Basis-eval kernels carved to a sibling module; re-exported so the registries below and external importers resolve unchanged.
from ._hermite_basis_eval import (
    _hermeval_njit, _legval_njit, _chebval_njit, _lagval_njit,
    _hermeval_njit_parallel, _legval_njit_parallel, _chebval_njit_parallel, _lagval_njit_parallel,
    _build_basis_hermite, _build_basis_legendre, _build_basis_chebyshev, _build_basis_laguerre,
    _BASIS_BUILDERS, build_basis_matrix,
)

# Optional CUDA RawKernel backend. One thread per output element with the recurrence kept in registers.
# Wins at n >= 500k once host->device transfer is amortised.

_CUDA_AVAILABLE = False
_CUDA_KERNELS: dict = {}

try:
    import cupy as _cp
    _CUDA_AVAILABLE = True
except ImportError:
    pass


def _polyeval_cuda(basis: str, x: np.ndarray, c: np.ndarray, device: int | None = None) -> np.ndarray:
    """CUDA RawKernel polynomial eval on ``device`` (current device if None). Includes H2D + launch + D2H.
    Worth it only at n >= 500k (per bench_poly_eval_backends)."""
    import contextlib

    import cupy as cp
    # _ensure_cuda_kernels writes into the _hermite_fe_mi module's dict;
    # read from the same source to avoid a stale local-module dict that
    # would surface as KeyError(basis) after the lazy compile succeeded.
    from .. import _hermite_fe_mi as _hfmi
    _ctx = cp.cuda.Device(device) if device is not None else contextlib.nullcontext()
    with _ctx:
        _ensure_cuda_kernels()
        # x (the column being evaluated) is invariant across CMA-ES/Optuna trials for the SAME column;
        # resident-cache it so repeated trials on one column upload ONCE. c (the coefficients) genuinely
        # vary per trial -- that is the thing being optimized -- so it stays a raw per-call upload.
        from .._fe_resident_operands import resident_operand
        x_gpu = resident_operand(x, "hermite_polyeval_x", dtype=np.float64)
        c_gpu = cp.asarray(c, dtype=cp.float64)
        n = x.shape[0]
        out_gpu = cp.empty(n, dtype=cp.float64)
        block = 256
        grid = (n + block - 1) // block
        _hfmi._CUDA_KERNELS[basis](
            (grid,), (block,),
            (x_gpu, c_gpu, c_gpu.shape[0], n, out_gpu),
        )
        return np.asarray(cp.asnumpy(out_gpu))


def _polyeval_cuda_pick_devices(n: int) -> list:
    """Return CUDA device indices (most-free VRAM first) that can hold a polyeval of length ``n`` (~4x the column +
    cushion). Enables a per-column device CASCADE: try the roomiest GPU, then the next, then (empty list) the CPU. One
    memGetInfo per device; query failure -> empty (CPU). Mirrors the multi-device spirit of the _fe_gpu_batch executor
    at the cheap per-column granularity, where no residency-matrix concatenation is needed."""
    try:
        import cupy as cp

        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception:
        return []
    needed = int(n) * 8 * 4 + (64 << 20)
    # ABSOLUTE cushion guard (2026-07-05): the ``free >= needed`` test below is RELATIVE to this column's size
    # and passes even when the card is nearly full (the pool already ate it) -- letting the next launch fault on
    # a shared card. ADD an absolute free-VRAM floor (default >=1 GB free) per device so a near-full device is
    # dropped from the cascade -> empty list -> CPU. Pure ADD -- tightens, never loosens; permissive without it.
    _cushion: Any = None
    try:
        from .._fe_gpu_vram import fe_gpu_has_vram_cushion as _cushion
    except Exception:
        _cushion = None
    fits = []
    for d in range(ndev):
        try:
            with cp.cuda.Device(d):
                free, _total = cp.cuda.runtime.memGetInfo()
                _cush_ok = _cushion(needed) if _cushion is not None else True
            if free >= needed and _cush_ok:
                fits.append((free, d))
        except Exception as e:  # nosec B112 - swallow converted to debug-log, non-fatal by design  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
            logger.debug("suppressed in __init__.py:277: %s", e)
            continue
    fits.sort(reverse=True)  # most-free first
    return [d for _free, d in fits]


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
from ._hermite_oracle import (
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

_POLYEVAL_CUDA_FALLBACK_WARNED = False


def _warn_polyeval_cuda_fallback_once(exc: Exception) -> None:
    """Warn ONCE that the polyeval CUDA path failed and we fell back to CPU (the error recurs per column)."""
    global _POLYEVAL_CUDA_FALLBACK_WARNED
    if not _POLYEVAL_CUDA_FALLBACK_WARNED:
        _POLYEVAL_CUDA_FALLBACK_WARNED = True
        logger.warning(
            "polyeval_dispatch: CUDA backend failed (%s); falling back to the CPU polyeval path for this and "
            "subsequent columns. Usually the HOST is out of RAM so cupy cannot allocate the pinned H2D buffer -- "
            "set MLFRAME_POLYEVAL_BACKEND=njit_par to skip the GPU attempt entirely, or free host RAM.",
            type(exc).__name__,
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
        return np.asarray(_NJIT_FUNCS[basis](x, c))
    # CUDA path: kernel_tuning_cache-driven, with an OOM/driver-error auto-fallback to the CPU path. On a host that is
    # itself out of RAM (paging) cupy cannot allocate the pinned H2D staging buffer and raises cudaErrorMemoryAllocation;
    # without this guard the caller drops the ENTIRE engineered column ("...skipping") rather than computing it slower on
    # the CPU. Degrade, don't lose the feature. Warned once (the failure recurs per column and would flood the log).
    # HOST-IN / HOST-OUT: this dispatcher takes a host (numpy) column and returns a host column -- the winner-replay
    # (apply_recipe) and the host basis builder both APPEND the result to a pandas frame. Computing that on the GPU means
    # H2D + D2H over PCIe around a memory-bound Horner pass: two transfers of the column for a trivial arithmetic sweep,
    # so the CPU (njit, in place, no transfer) is competitive-or-faster -- and, per the resident-vs-host lesson that made
    # the plug-in MI dispatcher default to CPU, a solo GPU microbench (the source of the old n>=500k threshold) overstates
    # the win once the FE pipeline fires these from many joblib workers contending on one GPU. So the DEFAULT never uploads
    # the feature column to the GPU; the GPU polyeval is used ONLY when explicitly forced (MLFRAME_POLYEVAL_BACKEND=cuda),
    # e.g. for A/B or a genuinely device-resident caller. The STRICT resident builder keeps its device operands and does
    # NOT route through here (it uses _gpu_evaluate_basis_matrix), so that legitimately-resident path is unaffected.
    if forced == "cuda" and _CUDA_AVAILABLE:
        # Per-column device CASCADE: try the roomiest GPU, then the next, then CPU -- so on a multi-GPU box a busy/full
        # device 0 does not block the eval, and an OOM (incl. host pinned-mem exhaustion) degrades instead of dropping
        # the column. Empty candidate list (no device has room) -> straight to CPU.
        for _dev in _polyeval_cuda_pick_devices(n) or [None]:
            try:
                return _polyeval_cuda(basis, x, c, device=_dev)
            except Exception as _cuda_exc:  # OOM / driver error on this device -> try the next, then CPU  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
                _warn_polyeval_cuda_fallback_once(_cuda_exc)
        # every device failed -- fall through to the CPU njit / njit_par path.
    if forced == "njit_par":
        return np.asarray(_NJIT_PAR_FUNCS[basis](x, c))
    # CPU njit/njit_par crossover: oracle-driven when enabled, else the legacy
    # hardcoded/kernel_tuning_cache threshold.
    if forced == "" and _polyeval_oracle_enabled():
        if _polyeval_oracle_pick_cpu_backend(n) == "njit_par":
            return np.asarray(_NJIT_PAR_FUNCS[basis](x, c))
        return np.asarray(_NJIT_FUNCS[basis](x, c))
    if n < _par_threshold:
        return np.asarray(_NJIT_FUNCS[basis](x, c))
    return np.asarray(_NJIT_PAR_FUNCS[basis](x, c))


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
from ._hermite_robust import (
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
    """Standardize ``x`` to z-scores; heavy-tailed columns use robust median/inner-quantile-range center/scale with a +/-6-sigma clip instead of the raw mean/std."""
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


@njit(cache=True)
def _minmax_neg1_1_njit(x: np.ndarray):
    """Fused min/max + [-1, 1] rescale for the clean (non-heavy-tail) min-max preprocessor. One reduction pass + one
    output pass instead of numpy's np.min + np.max + elementwise (three full passes) on the 1M-row operand. Bit-identical
    to the numpy body: np.min/np.max are order-independent (loop reproduces them exactly, INCLUDING NaN propagation via
    the ``nan_seen`` poison so a NaN column yields the same all-NaN z numpy would), and the per-element expression keeps
    numpy's exact operator order ``2*(x-lo)/span - 1``."""
    n = x.size
    lo = np.inf
    hi = -np.inf
    nan_seen = False
    for i in range(n):
        v = x[i]
        if v != v:  # NaN
            nan_seen = True
        else:
            if v < lo:
                lo = v
            if v > hi:
                hi = v
    if nan_seen:
        lo = np.nan
        hi = np.nan
    span = hi - lo + 1e-12
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = 2.0 * (x[i] - lo) / span - 1.0
    return out, lo, hi


def _preprocess_minmax_neg1_1(x):
    """Rescale ``x`` onto ``[-1, 1]``; heavy-tailed columns use robust inner-quantile bounds with clamping to +/-1 instead of the raw min/max."""
    if _robust_axis_enabled() and _detect_heavy_tail(x):
        # Min-max onto [-1, 1] from the inner-quantile bounds; clamp so clipped outliers pin to +/-1 (the basis domain edge)
        # instead of compressing the core toward 0. clip is implied (the [-1, 1] clamp), recorded so replay matches.
        lo, hi = _robust_lo_hi(x)
        span = hi - lo + 1e-12
        z = np.clip(2 * (x - lo) / span - 1, -1.0, 1.0)
        return z, dict(lo=lo, hi=hi, clip=1.0)
    z, lo, hi = _minmax_neg1_1_njit(np.ascontiguousarray(x, dtype=np.float64))
    return z, dict(lo=float(lo), hi=float(hi))


def _preprocess_shift_nonneg(x):
    """Shift ``x`` to be non-negative (Laguerre domain); heavy-tailed columns clamp the upper tail to the robust inner-quantile range instead of the raw max, avoiding an exploding ``L_n`` argument."""
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
    """Replay a fitted ``_preprocess_zscore`` transform onto new data from its stored ``mean``/``std``/(optional)``clip`` params."""
    z = (x - params["mean"]) / max(params["std"], 1e-12)
    clip = params.get("clip")
    if clip is not None:
        z = np.clip(z, -float(clip), float(clip))
    return z


def _apply_minmax(x, params):
    """Replay a fitted ``_preprocess_minmax_neg1_1`` transform onto new data from its stored ``lo``/``hi``/(optional)``clip`` params."""
    span = params["hi"] - params["lo"] + 1e-12
    z = 2 * (x - params["lo"]) / span - 1
    clip = params.get("clip")
    if clip is not None:
        z = np.clip(z, -float(clip), float(clip))
    return z


def _apply_shift(x, params):
    """Replay a fitted ``_preprocess_shift_nonneg`` transform onto new data from its stored ``lo``/(optional)``clip`` params."""
    z = x - params["lo"] + 1e-9
    clip = params.get("clip")
    if clip is not None:
        z = np.clip(z, 0.0, float(clip) + 1e-9)
    return z


def _make_dispatch(name):
    """Bind the basis name into a closure matching the (x, c) -> ndarray signature of eval / eval_njit."""
    def _d(x, c):
        """Evaluate the ``name``-basis polynomial with coefficients ``c`` at ``x`` via the size-aware backend dispatcher."""
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
        return np.asarray(self.bin_func(h_a, h_b))


def _safe_div(a, b):
    """Element-wise division that is EXACT for every nonzero denominator and finite (never x_a/0 blowup) at exact zero,
    so polynomials can capture ratio targets without distorting them.

    SAFE-DIV CANONICAL RECIPE (y==0 -> eps, else exact divide). Mirrored bit-for-bit in two GPU spellings that must stay
    in lock-step with this one: ``_gpu_resident_fe._binary_apply`` (cupy ``xp.where(y==0.0, 1e-9, y)``) and the fused CUDA
    RawKernel ``_FUSED_GEN_SRC`` (``case 3: x / ((y==0.0) ? 1e-9 : y)``). Not DRY-unified because the three live in
    different runtimes (numpy host fn / xp-generic / CUDA-C source string); any change to the eps/branch here MUST be
    applied to both GPU sites or the CPU<->GPU op-parity test (equiv_rtol=1e-9) breaks."""
    eps = 1e-9
    # HEAVY-TAIL FIX (2026-06-13): substitute ``eps`` ONLY for an exactly-zero denominator; every nonzero ``b`` divides
    # exactly. The prior ``np.where(b >= 0, b + eps, b - eps)`` guaranteed ``|denom| >= eps`` but perturbed EVERY
    # denominator by ``eps`` -- negligible for ordinary ``b`` yet ~``eps/b`` relative error as ``b -> 0`` (a 0.1% error
    # at ``b=1e-6``), which on a heavy-tailed ratio target inflates a linear downstream's MAE on the small-``b`` tail.
    b = np.asarray(b, dtype=np.float64)
    denom = np.where(b == 0.0, eps, b)
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


@njit(cache=True)
def _moment_fingerprint_njit(x: np.ndarray):
    """Fused two-pass moment fingerprint (mean, std, skew, excess-kurtosis, min, max) for ``basis_route_by_moments``.

    Replaces ~10 separate numpy reductions + four 1M-element temporaries (z, z2, z2*z, z2*z2) with two cache-friendly
    loops and NO intermediate arrays -- the routing is called once per operand column per fit and was the dominant pure-
    host (numpy) hotspot of the STRICT F2 fit (~0.65s tottime). Numerics mirror the numpy body: ``mean = sum/n`` (same as
    ``np.mean``), ``std = sqrt(var) + 1e-12`` (same as ``np.std(x) + 1e-12``), ``skew = mean((x-mean)^3)/std^3`` and
    ``kurt_excess = mean((x-mean)^4)/std^4 - 3`` (identical to the ``z=(x-mean)/std``; ``mean(z^3)`` / ``mean(z^4)-3``
    formulation). Only the summation ORDER differs from numpy's pairwise reduction, so the moments match to ~1e-12
    relative and the routing verdict (a coarse thresholded pick over margins of order 1) is selection-identical.
    NaN handling is verdict-equivalent: any NaN poisons ``mean``/``std``/``skew`` -> every branch predicate is False ->
    the chebyshev default, exactly as the numpy path (NaN-poisoned comparisons) returns."""
    n = x.size
    s = 0.0
    xmin = np.inf
    xmax = -np.inf
    for i in range(n):
        v = x[i]
        s += v
        if v < xmin:
            xmin = v
        if v > xmax:
            xmax = v
    mean = s / n
    s2 = 0.0
    s3 = 0.0
    s4 = 0.0
    for i in range(n):
        d = x[i] - mean
        d2 = d * d
        s2 += d2
        s3 += d2 * d
        s4 += d2 * d2
    std = (s2 / n) ** 0.5 + 1e-12
    inv = 1.0 / std
    inv2 = inv * inv
    skew = (s3 / n) * (inv2 * inv)
    kurt_excess = (s4 / n) * (inv2 * inv2) - 3.0
    return mean, std, skew, kurt_excess, xmin, xmax


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
    # Single fused njit pass computes all six statistics without materialising the z / z2 / z3 / z4 temporaries the
    # legacy numpy body allocated (the ** vs chained-mul antipattern noted below is subsumed: the njit uses d*d / d2*d).
    x = np.ascontiguousarray(x)
    _mean, std, skew, kurt_excess, xmin, xmax = _moment_fingerprint_njit(x)
    rng = xmax - xmin
    spread_ratio = rng / std
    one_sided = (xmin >= 0) or (xmax <= 0)
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
from ._hermite_prewarp import (
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
from .._hermite_fe_optimise import (
    _baseline_mi_pair, _eval_coef_pair, _run_cma_search, _select_diverse_topm, detect_pair_symmetry, optimise_hermite_pair, optimise_pair_multimode, precompute_hermite_pair_basis,
)
from .._hermite_fe_mi import (
    _ensure_cuda_kernels, _plugin_mi_classif_batch_cuda, _plugin_mi_classif_batch_cuda_resident, _plugin_mi_classif_njit, _plugin_mi_from_binned_njit, _plugin_mi_regression_njit, plugin_mi_classif_batch_dispatch, plugin_mi_classif_dispatch, plugin_mi_classif_fast,
)
