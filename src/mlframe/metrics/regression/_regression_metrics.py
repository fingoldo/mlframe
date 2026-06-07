"""Regression metrics for ``mlframe.metrics.core`` — numba-accelerated drop-ins for sklearn.

Split out from ``core.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; every moved symbol is
re-exported from ``core`` so existing
``from mlframe.metrics.core import fast_mean_squared_error`` (and the
other moved names) imports continue to work.

What lives here:
  - 1-D unweighted seq/par kernels: ``_fast_mae_*``, ``_fast_mse_*``,
    ``_fast_max_error_seq``, ``_fast_r2_score_*``, ``_fast_r2_variance_seq``.
  - 1-D weighted seq/par kernels: ``_fast_mae_weighted_*``,
    ``_fast_mse_weighted_*``, ``_fast_r2_score_weighted_*``.
  - Multioutput aggregation helpers: ``_aggregate_multioutput``, ``_to_2d``.
  - Public wrappers: ``fast_mean_absolute_error``,
    ``fast_mean_squared_error``, ``fast_root_mean_squared_error``,
    ``fast_max_error``, ``fast_r2_score``.
  - Fused 2-pass block: ``_fused_regression_pass1_*``,
    ``_fused_regression_pass2_*``, ``fast_regression_metrics_block``.

Speedups vs sklearn at N=1M:
  MAE   17x  | MSE   15x  | RMSE  same as MSE | max_error 6x | R2 23x
Speedup of the fused block vs 4 separate kernel calls:
  N=10k: 3.3x | N=500k: 2.8x | N=5M: 3.4x.

sklearn input-validation overhead dominates these tiny reductions: at
N=1M, sklearn's ``mean_absolute_error`` takes 12 ms while the numba
kernel does the same work in ~1.5 ms (8x). Adding parallel=True drops
it further to 0.7 ms (17x over sklearn).

Full sklearn signature support:
  - 1-D and 2-D ``y_true`` / ``y_pred`` (multioutput).
  - ``sample_weight`` (1-D weights, broadcast across outputs).
  - ``multioutput`` in ``'raw_values'``, ``'uniform_average'``, an
    array of per-output weights, or (R² only) ``'variance_weighted'``.

1-D unweighted is the fastest path (single numba kernel call). 2-D
multioutput loops per column inside Python (M is typically small).
Weighted variants use a separate numba kernel.
"""
from __future__ import annotations

from typing import Dict, Union

import numpy as np
import numba

from .._numba_params import NUMBA_NJIT_PARAMS, _PARALLEL_REDUCTION_THRESHOLD


# ---------- 1-D unweighted ----------


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_mae_seq(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    s = 0.0
    for i in range(n):
        s += abs(y_true[i] - y_pred[i])
    return s / n


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_mae_par(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    s = 0.0
    for i in numba.prange(n):
        s += abs(y_true[i] - y_pred[i])
    return s / n


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_mse_seq(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    s = 0.0
    for i in range(n):
        d = y_true[i] - y_pred[i]
        s += d * d
    return s / n


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_mse_par(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    s = 0.0
    for i in numba.prange(n):
        d = y_true[i] - y_pred[i]
        s += d * d
    return s / n


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_max_error_seq(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    m = 0.0
    for i in range(n):
        d = abs(y_true[i] - y_pred[i])
        if d > m:
            m = d
    return m


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_r2_score_seq(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Two-pass: mean of y_true, then SS_res and SS_tot."""
    n = len(y_true)
    ymean = 0.0
    for i in range(n):
        ymean += y_true[i]
    ymean /= n
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        d_res = y_true[i] - y_pred[i]
        d_tot = y_true[i] - ymean
        ss_res += d_res * d_res
        ss_tot += d_tot * d_tot
    if ss_tot == 0.0:
        return 0.0  # sklearn convention for constant y_true
    return 1.0 - ss_res / ss_tot


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_r2_score_par(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    ymean = 0.0
    for i in numba.prange(n):
        ymean += y_true[i]
    ymean /= n
    ss_res = 0.0
    ss_tot = 0.0
    for i in numba.prange(n):
        d_res = y_true[i] - y_pred[i]
        d_tot = y_true[i] - ymean
        ss_res += d_res * d_res
        ss_tot += d_tot * d_tot
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_r2_variance_seq(y_true: np.ndarray) -> float:
    """SS_tot for r2_variance_weighted multioutput aggregation."""
    n = len(y_true)
    ymean = 0.0
    for i in range(n):
        ymean += y_true[i]
    ymean /= n
    ss = 0.0
    for i in range(n):
        d = y_true[i] - ymean
        ss += d * d
    return ss


# ---------- 1-D weighted ----------


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_mae_weighted_seq(y_true, y_pred, w):
    n = len(y_true)
    s = 0.0
    wsum = 0.0
    for i in range(n):
        s += abs(y_true[i] - y_pred[i]) * w[i]
        wsum += w[i]
    return s / wsum


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_mae_weighted_par(y_true, y_pred, w):
    n = len(y_true)
    s = 0.0
    wsum = 0.0
    for i in numba.prange(n):
        s += abs(y_true[i] - y_pred[i]) * w[i]
        wsum += w[i]
    return s / wsum


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_mse_weighted_seq(y_true, y_pred, w):
    n = len(y_true)
    s = 0.0
    wsum = 0.0
    for i in range(n):
        d = y_true[i] - y_pred[i]
        s += d * d * w[i]
        wsum += w[i]
    return s / wsum


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_mse_weighted_par(y_true, y_pred, w):
    n = len(y_true)
    s = 0.0
    wsum = 0.0
    for i in numba.prange(n):
        d = y_true[i] - y_pred[i]
        s += d * d * w[i]
        wsum += w[i]
    return s / wsum


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fast_r2_score_weighted_seq(y_true, y_pred, w):
    n = len(y_true)
    ymean = 0.0
    wsum = 0.0
    for i in range(n):
        ymean += y_true[i] * w[i]
        wsum += w[i]
    ymean /= wsum
    ss_res = 0.0
    ss_tot = 0.0
    for i in range(n):
        d_res = y_true[i] - y_pred[i]
        d_tot = y_true[i] - ymean
        ss_res += d_res * d_res * w[i]
        ss_tot += d_tot * d_tot * w[i]
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fast_r2_score_weighted_par(y_true, y_pred, w):
    n = len(y_true)
    ymean = 0.0
    wsum = 0.0
    for i in numba.prange(n):
        ymean += y_true[i] * w[i]
        wsum += w[i]
    ymean /= wsum
    ss_res = 0.0
    ss_tot = 0.0
    for i in numba.prange(n):
        d_res = y_true[i] - y_pred[i]
        d_tot = y_true[i] - ymean
        ss_res += d_res * d_res * w[i]
        ss_tot += d_tot * d_tot * w[i]
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------- multioutput aggregation ----------


def _aggregate_multioutput(values: np.ndarray, multioutput) -> Union[float, np.ndarray]:
    """Apply sklearn's ``multioutput`` aggregation to a per-output values array.

    - ``'raw_values'`` -> ``values`` unchanged.
    - ``'uniform_average'`` -> ``np.mean(values)``.
    - array-like -> ``np.average(values, weights=multioutput)``.
    - other strings / inputs -> ValueError.
    """
    # Type check FIRST: ``multioutput == "raw_values"`` on an ndarray
    # broadcasts and raises 'ambiguous truth value'. Handle the
    # array-weights path before any string compare.
    if isinstance(multioutput, (np.ndarray, list, tuple)):
        return float(np.average(values, weights=np.asarray(multioutput, dtype=np.float64)))
    if multioutput == "raw_values":
        return values
    if multioutput == "uniform_average":
        return float(values.mean())
    raise ValueError(
        f"multioutput must be 'raw_values', 'uniform_average', or an array; got {multioutput!r}"
    )


def _to_2d(arr: np.ndarray) -> np.ndarray:
    """Promote ``(N,)`` to ``(N, 1)``; pass-through ``(N, M)``."""
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    return arr


# ---------- public wrappers ----------


def fast_mean_absolute_error(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput: Union[str, np.ndarray] = "uniform_average",
):
    """Drop-in for ``sklearn.metrics.mean_absolute_error``.

    Faster than sklearn (17× at N=1M) while supporting the full sklearn
    signature: ``sample_weight`` (1-D, broadcast across outputs) and
    ``multioutput`` in ``{'raw_values', 'uniform_average'}`` or an
    array of per-output weights.
    """
    yt = np.ascontiguousarray(np.asarray(y_true), dtype=np.float64)
    yp = np.ascontiguousarray(np.asarray(y_pred), dtype=np.float64)
    if sample_weight is not None:
        w = np.ascontiguousarray(np.asarray(sample_weight), dtype=np.float64)
    else:
        w = None

    if yt.ndim == 1:
        n = yt.shape[0]
        if w is None:
            return _fast_mae_par(yt, yp) if n >= _PARALLEL_REDUCTION_THRESHOLD else _fast_mae_seq(yt, yp)
        return _fast_mae_weighted_par(yt, yp, w) if n >= _PARALLEL_REDUCTION_THRESHOLD else _fast_mae_weighted_seq(yt, yp, w)

    yt2 = _to_2d(yt)
    yp2 = _to_2d(yp)
    n = yt2.shape[0]
    M = yt2.shape[1]
    per_out = np.empty(M, dtype=np.float64)
    use_par = n >= _PARALLEL_REDUCTION_THRESHOLD
    for j in range(M):
        col_yt = np.ascontiguousarray(yt2[:, j])
        col_yp = np.ascontiguousarray(yp2[:, j])
        if w is None:
            per_out[j] = _fast_mae_par(col_yt, col_yp) if use_par else _fast_mae_seq(col_yt, col_yp)
        else:
            per_out[j] = _fast_mae_weighted_par(col_yt, col_yp, w) if use_par else _fast_mae_weighted_seq(col_yt, col_yp, w)
    return _aggregate_multioutput(per_out, multioutput)


def fast_mean_squared_error(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput: Union[str, np.ndarray] = "uniform_average",
):
    """Drop-in for ``sklearn.metrics.mean_squared_error``. 15× faster at
    N=1M with full sample_weight + multioutput support."""
    yt = np.ascontiguousarray(np.asarray(y_true), dtype=np.float64)
    yp = np.ascontiguousarray(np.asarray(y_pred), dtype=np.float64)
    if sample_weight is not None:
        w = np.ascontiguousarray(np.asarray(sample_weight), dtype=np.float64)
    else:
        w = None

    if yt.ndim == 1:
        n = yt.shape[0]
        if w is None:
            return _fast_mse_par(yt, yp) if n >= _PARALLEL_REDUCTION_THRESHOLD else _fast_mse_seq(yt, yp)
        return _fast_mse_weighted_par(yt, yp, w) if n >= _PARALLEL_REDUCTION_THRESHOLD else _fast_mse_weighted_seq(yt, yp, w)

    yt2 = _to_2d(yt)
    yp2 = _to_2d(yp)
    n = yt2.shape[0]
    M = yt2.shape[1]
    per_out = np.empty(M, dtype=np.float64)
    use_par = n >= _PARALLEL_REDUCTION_THRESHOLD
    for j in range(M):
        col_yt = np.ascontiguousarray(yt2[:, j])
        col_yp = np.ascontiguousarray(yp2[:, j])
        if w is None:
            per_out[j] = _fast_mse_par(col_yt, col_yp) if use_par else _fast_mse_seq(col_yt, col_yp)
        else:
            per_out[j] = _fast_mse_weighted_par(col_yt, col_yp, w) if use_par else _fast_mse_weighted_seq(col_yt, col_yp, w)
    return _aggregate_multioutput(per_out, multioutput)


def fast_root_mean_squared_error(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput: Union[str, np.ndarray] = "uniform_average",
):
    """Drop-in for ``sklearn.metrics.root_mean_squared_error``.

    Computes per-output RMSE = sqrt(MSE) BEFORE aggregating
    (``raw_values`` returns per-output RMSEs; ``uniform_average``
    averages them after sqrt). Matches sklearn's behaviour exactly.
    """
    if isinstance(multioutput, str) and multioutput == "raw_values":
        per_out_mse = fast_mean_squared_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput="raw_values",
        )
        return np.sqrt(np.atleast_1d(per_out_mse))
    # For aggregated forms, sklearn computes per-output RMSE then averages.
    per_out_mse = np.atleast_1d(
        fast_mean_squared_error(
            y_true, y_pred, sample_weight=sample_weight, multioutput="raw_values",
        )
    )
    per_out_rmse = np.sqrt(per_out_mse)
    return _aggregate_multioutput(per_out_rmse, multioutput)


def fast_max_error(
    y_true,
    y_pred,
    *,
    multioutput: Union[str, np.ndarray] = "raw_values",
):
    """Drop-in for ``sklearn.metrics.max_error``. ~6× faster at N=1M.

    sklearn's ``max_error`` raises on multioutput input; we extend with
    ``multioutput`` support (default ``'raw_values'`` returns per-output
    max). ``sample_weight`` does NOT apply to max-error (max is max).
    """
    yt = np.ascontiguousarray(np.asarray(y_true), dtype=np.float64)
    yp = np.ascontiguousarray(np.asarray(y_pred), dtype=np.float64)
    if yt.ndim == 1:
        return _fast_max_error_seq(yt, yp)
    yt2 = _to_2d(yt)
    yp2 = _to_2d(yp)
    M = yt2.shape[1]
    per_out = np.empty(M, dtype=np.float64)
    for j in range(M):
        per_out[j] = _fast_max_error_seq(
            np.ascontiguousarray(yt2[:, j]),
            np.ascontiguousarray(yp2[:, j]),
        )
    return _aggregate_multioutput(per_out, multioutput)


def fast_r2_score(
    y_true,
    y_pred,
    *,
    sample_weight=None,
    multioutput: Union[str, np.ndarray] = "uniform_average",
):
    """Drop-in for ``sklearn.metrics.r2_score``. 23× faster at N=1M with
    full sample_weight + multioutput support, including
    ``'variance_weighted'`` aggregation."""
    yt = np.ascontiguousarray(np.asarray(y_true), dtype=np.float64)
    yp = np.ascontiguousarray(np.asarray(y_pred), dtype=np.float64)
    if sample_weight is not None:
        w = np.ascontiguousarray(np.asarray(sample_weight), dtype=np.float64)
    else:
        w = None

    if yt.ndim == 1:
        n = yt.shape[0]
        if w is None:
            return _fast_r2_score_par(yt, yp) if n >= _PARALLEL_REDUCTION_THRESHOLD else _fast_r2_score_seq(yt, yp)
        return _fast_r2_score_weighted_par(yt, yp, w) if n >= _PARALLEL_REDUCTION_THRESHOLD else _fast_r2_score_weighted_seq(yt, yp, w)

    yt2 = _to_2d(yt)
    yp2 = _to_2d(yp)
    n = yt2.shape[0]
    M = yt2.shape[1]
    per_out = np.empty(M, dtype=np.float64)
    use_par = n >= _PARALLEL_REDUCTION_THRESHOLD
    for j in range(M):
        col_yt = np.ascontiguousarray(yt2[:, j])
        col_yp = np.ascontiguousarray(yp2[:, j])
        if w is None:
            per_out[j] = _fast_r2_score_par(col_yt, col_yp) if use_par else _fast_r2_score_seq(col_yt, col_yp)
        else:
            per_out[j] = _fast_r2_score_weighted_par(col_yt, col_yp, w) if use_par else _fast_r2_score_weighted_seq(col_yt, col_yp, w)

    if isinstance(multioutput, str) and multioutput == "variance_weighted":
        # Per-output variance of y_true (weighted if w supplied).
        ss_tots = np.empty(M, dtype=np.float64)
        for j in range(M):
            col_yt = np.ascontiguousarray(yt2[:, j])
            if w is None:
                ss_tots[j] = _fast_r2_variance_seq(col_yt)
            else:
                # Weighted variance = SS_tot_w / sum(w). The relative
                # weights are what matters for the average; using SS_tot
                # directly preserves the proportions sklearn uses.
                wsum = float(w.sum())
                # sample_weight summing to zero (zero-weighted excluded fold
                # rows) would divide by 0 here BEFORE the ss_tots.sum()==0
                # degenerate guard could observe it.
                if wsum <= 0.0:
                    ss_tots[j] = 0.0
                    continue
                wmean = float((col_yt * w).sum() / wsum)
                ss_tots[j] = float(((col_yt - wmean) ** 2 * w).sum())
        if ss_tots.sum() == 0.0:
            # Degenerate: all outputs constant. sklearn returns 0.0
            # when all SS_tot are zero (otherwise raises).
            return 0.0
        return float(np.average(per_out, weights=ss_tots))

    return _aggregate_multioutput(per_out, multioutput)


# ---------- fused MAE / RMSE / MaxError / R^2 (single-pass over data) ----------
#
# 4 separate ``fast_*`` kernel calls hit RAM 4-5 times for the same
# (y_true, y_pred) arrays. The regression-reporting block at
# ``_reporting.py:report_regression_model_perf`` always computes ALL FOUR
# of these on the same arrays at the same call site, so fusing them into
# a single 2-pass kernel (1 pass for sum_abs/sum_sqr/max_abs/sum_y, 1 pass
# for centred SS_tot) gives the same numeric result with 2.3-3.4x speedup.
#
# Numerical notes:
# - SS_tot is the CENTRED sum-of-squares, NOT the algebraic identity
#   ``sum_y_sq - n*y_mean^2`` -- that identity catastrophically cancels
#   on float64 when y has a large mean (y_mean=11500 produced 250-unit
#   drift in R^2 vs sklearn).
# - max_abs reduction inside ``numba.prange`` is RACY -- numba's auto-
#   reduction only handles SUM-style ops safely. We use the explicit
#   per-thread-accumulator pattern (one array slot per thread, serial
#   reduce after the parallel block).
# - The parallel kernel falls back to a sequential variant below
#   ``_PARALLEL_REDUCTION_THRESHOLD`` rows (the parallel kernel's per-
#   thread setup overhead would dominate the work otherwise).
#
# Bench (``mlframe.metrics._benchmarks.bench_fused_regression_metrics``):
#   N=10k:   sep=0.10ms  fused=0.03ms (3.3x)  max_diff=0
#   N=500k:  sep=2.1ms   fused=0.7ms  (2.8x)  max_diff=1e-13
#   N=5M:    sep=18.9ms  fused=5.5ms  (3.4x)  max_diff=0


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fused_regression_pass1_seq(y_true: np.ndarray, y_pred: np.ndarray):
    """Pass 1 of the fused regression-metrics kernel (sequential).

    Returns ``(sum_abs_err, sum_sqr_err, max_abs_err, sum_y_true)`` so
    the caller can derive MAE / RMSE / MaxError and the y_true mean from
    a single walk over both arrays. SS_tot for R^2 lands in a stable
    second pass once the mean is known.
    """
    n = y_true.shape[0]
    sum_abs = 0.0
    sum_sqr = 0.0
    max_abs = 0.0
    sum_y = 0.0
    for i in range(n):
        err = y_true[i] - y_pred[i]
        abs_err = err if err >= 0.0 else -err
        sum_abs += abs_err
        sum_sqr += err * err
        if abs_err > max_abs:
            max_abs = abs_err
        sum_y += y_true[i]
    return sum_abs, sum_sqr, max_abs, sum_y


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fused_regression_pass1_par(y_true: np.ndarray, y_pred: np.ndarray, n_threads: int):
    """Pass 1 of the fused regression-metrics kernel (parallel, per-thread
    accumulators). See module-level rationale on the racy ``max`` reduction
    -- this is the explicit per-thread pattern that sidesteps it.

    ``n_threads`` is passed in rather than fetched via
    ``numba.get_num_threads()`` inside the kernel so the @njit-cache
    persists across runs (ctypes calls block caching with the
    "Cannot cache compiled function" NumbaWarning).
    """
    n = y_true.shape[0]
    chunk_size = (n + n_threads - 1) // n_threads
    local_sum_abs = np.zeros(n_threads, dtype=np.float64)
    local_sum_sqr = np.zeros(n_threads, dtype=np.float64)
    local_max_abs = np.zeros(n_threads, dtype=np.float64)
    local_sum_y = np.zeros(n_threads, dtype=np.float64)
    for tid in numba.prange(n_threads):
        start = tid * chunk_size
        end = min(start + chunk_size, n)
        s_abs = 0.0
        s_sqr = 0.0
        m = 0.0
        s_y = 0.0
        for i in range(start, end):
            err = y_true[i] - y_pred[i]
            abs_err = err if err >= 0.0 else -err
            s_abs += abs_err
            s_sqr += err * err
            if abs_err > m:
                m = abs_err
            s_y += y_true[i]
        local_sum_abs[tid] = s_abs
        local_sum_sqr[tid] = s_sqr
        local_max_abs[tid] = m
        local_sum_y[tid] = s_y
    sum_abs = 0.0
    sum_sqr = 0.0
    max_abs = 0.0
    sum_y = 0.0
    for tid in range(n_threads):
        sum_abs += local_sum_abs[tid]
        sum_sqr += local_sum_sqr[tid]
        if local_max_abs[tid] > max_abs:
            max_abs = local_max_abs[tid]
        sum_y += local_sum_y[tid]
    return sum_abs, sum_sqr, max_abs, sum_y


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fused_regression_pass2_seq(y_true: np.ndarray, y_mean: float) -> float:
    """Pass 2: centred sum-of-squares around the pre-computed mean."""
    n = y_true.shape[0]
    ss = 0.0
    for i in range(n):
        d = y_true[i] - y_mean
        ss += d * d
    return ss


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fused_regression_pass2_par(y_true: np.ndarray, y_mean: float) -> float:
    n = y_true.shape[0]
    ss = 0.0
    for i in numba.prange(n):
        d = y_true[i] - y_mean
        ss += d * d
    return ss


def fast_regression_metrics_block(
    y_true,
    y_pred,
) -> Dict[str, float]:
    """Compute MAE / RMSE / MaxError / R^2 from a single fused pass.

    Returns a dict ``{"MAE": ..., "RMSE": ..., "MaxError": ..., "R2": ...}``.
    Drop-in replacement for the 4 separate ``fast_*`` calls used by the
    regression reporting block; 2.3-3.4x faster across the typical
    ``n in {10k, 500k, 5M}`` regime.

    1-D input only (the report-side call site always passes 1-D arrays
    via ``squeeze``). For 2-D / multioutput regression the caller should
    fall through to the per-output ``fast_*`` helpers; multioutput
    aggregation has a non-trivial dispatch and the speedup of fusing one
    target's pass doesn't compose cleanly across outputs.
    """
    yt = np.ascontiguousarray(np.asarray(y_true), dtype=np.float64)
    yp = np.ascontiguousarray(np.asarray(y_pred), dtype=np.float64)
    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError(
            "fast_regression_metrics_block expects 1-D arrays; got shapes "
            f"y_true={yt.shape}, y_pred={yp.shape}. For 2-D / multioutput "
            "regression call the individual fast_* helpers."
        )
    n = yt.shape[0]
    if n == 0:
        return {"MAE": 0.0, "RMSE": 0.0, "MaxError": 0.0, "R2": 0.0}
    use_par = n >= _PARALLEL_REDUCTION_THRESHOLD
    if use_par:
        sum_abs, sum_sqr, max_abs, sum_y = _fused_regression_pass1_par(yt, yp, numba.get_num_threads())
        y_mean = sum_y / n
        ss_tot = _fused_regression_pass2_par(yt, y_mean)
    else:
        sum_abs, sum_sqr, max_abs, sum_y = _fused_regression_pass1_seq(yt, yp)
        y_mean = sum_y / n
        ss_tot = _fused_regression_pass2_seq(yt, y_mean)
    mae = sum_abs / n
    mse = sum_sqr / n
    rmse = float(np.sqrt(mse))
    if ss_tot <= 0.0:
        # sklearn convention: constant y_true and zero residuals -> R^2 = 0.0
        r2 = 0.0 if sum_sqr == 0.0 else float("-inf")
    else:
        r2 = 1.0 - sum_sqr / ss_tot
    return {"MAE": float(mae), "RMSE": rmse, "MaxError": float(max_abs), "R2": float(r2)}
