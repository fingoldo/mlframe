"""Additional regression metrics.

Complements ``_regression_metrics.py`` (MAE / MSE / RMSE / MaxError / R2)
with the wider set of standard regression diagnostics.

Public API (re-exported from ``mlframe.metrics.core``):
    * ``fast_rmsle``                   - Root Mean Squared Log Error
    * ``fast_mape_mean``               - mean absolute percentage error
    * ``fast_smape``                   - Symmetric MAPE in [0, 200]%
    * ``fast_mdape``                   - Median APE (outlier-robust)
    * ``fast_wmape``                   - Weighted MAPE by |y|
    * ``fast_mase``                    - Mean Absolute Scaled Error
    * ``fast_mean_bias_error``         - signed mean residual (MBE)
    * ``fast_pearson_corr``            - Pearson r between y_true / y_pred
    * ``fast_kendall_tau``             - Kendall's tau-b (O(N log N))
    * ``fast_cv_rmse``                 - RMSE / mean(y)
    * ``fast_nash_sutcliffe``          - NSE coefficient
    * ``fast_explained_variance``      - 1 - Var(residual)/Var(y)
    * ``fast_huber_loss``              - Huber loss with delta
    * ``fast_concordance_index``       - C-index = pairwise Kendall

For Spearman use ``mlframe.metrics.rank_correlation.spearmanr_batched_dispatch``
(already present); a scalar wrapper ``fast_spearman_corr`` is also provided
here for the single-pair case used in regression reports.
"""
from __future__ import annotations

from math import sqrt, log1p
from typing import Tuple

import numpy as np
import numba

from .._numba_params import NUMBA_NJIT_PARAMS, _PARALLEL_REDUCTION_THRESHOLD, _check_equal_length
from ._regression_metrics import fast_r2_score
# iter592: hoist the rank_correlation import out of fast_spearman_corr's
# body. c0103_2a635557 @100k profile attributed 140 ms of
# fast_spearman_corr's 182 ms cumtime / 10 calls (~14 ms per call avg) to
# ``<frozen importlib._bootstrap>:1165(_find_and_load)`` -- the first call
# paid the full module-resolution cost while the remaining 9 calls hit
# sys.modules immediately. rank_correlation.py imports only numpy +
# NUMBA_NJIT_PARAMS (no cycle into _regression_extras), so a top-level
# import is safe. After hoisting the cost moves to mlframe.metrics import
# time (paid once at startup, amortised across the whole suite) and
# fast_spearman_corr's runtime profile drops the import attribution
# entirely. Per-call cost becomes ~4.2 ms (just _spearmanr_batched_numpy).


# ============================================================================
# RMSLE
# ============================================================================


@numba.njit(**NUMBA_NJIT_PARAMS)
def _rmsle_kernel_seq(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, int]:
    """Returns (RMSLE, count_of_negatives). Negatives are skipped from the
    sum so the kernel does not crash on a stray <0 row; the count is
    surfaced to the wrapper so a warning can fire."""
    n = y_true.shape[0]
    s = 0.0
    neg = 0
    used = 0
    for i in range(n):
        yt = y_true[i]
        yp = y_pred[i]
        if yt < 0.0 or yp < 0.0:
            neg += 1
            continue
        d = log1p(yp) - log1p(yt)
        s += d * d
        used += 1
    if used == 0:
        return np.nan, neg
    return sqrt(s / used), neg


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _rmsle_kernel_par(y_true: np.ndarray, y_pred: np.ndarray, nthr: int) -> Tuple[float, int]:
    """Parallel (prange) variant of ``_rmsle_kernel_seq``: per-thread partial sums/counts, reduced at the end. Same
    negative-row skip semantics and (RMSLE, count_of_negatives) return contract."""
    n = y_true.shape[0]
    sums = np.zeros(nthr, dtype=np.float64)
    negs = np.zeros(nthr, dtype=np.int64)
    used = np.zeros(nthr, dtype=np.int64)
    for i in numba.prange(n):
        t = numba.get_thread_id()
        yt = y_true[i]
        yp = y_pred[i]
        if yt < 0.0 or yp < 0.0:
            negs[t] += 1
            continue
        d = log1p(yp) - log1p(yt)
        sums[t] += d * d
        used[t] += 1
    total = sums.sum()
    used_total = int(used.sum())
    neg_total = int(negs.sum())
    if used_total == 0:
        return np.nan, neg_total
    return sqrt(total / used_total), neg_total


_RMSLE_NEG_WARN_SEEN: set = set()


def fast_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Logarithmic Error.

    RMSLE = sqrt(mean((log(1 + y_pred) - log(1 + y_true))^2))

    Defined only for y_true, y_pred >= 0; negative rows are skipped and
    a rate-limited warning fires (silently dropping them masks data
    leakage). NaN when zero non-negative rows.

    Penalises under-prediction more than over-prediction (asymmetric);
    standard in retail forecasting, energy demand, click-through rate.
    """
    _check_equal_length(y_true, y_pred)
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    if yt.shape[0] == 0:
        return np.nan
    if yt.shape[0] >= _PARALLEL_REDUCTION_THRESHOLD:
        val, neg = _rmsle_kernel_par(yt, yp, numba.get_num_threads())
    else:
        val, neg = _rmsle_kernel_seq(yt, yp)
    if neg > 0:
        key = (int(neg), int(yt.shape[0]))
        if key not in _RMSLE_NEG_WARN_SEEN:
            _RMSLE_NEG_WARN_SEEN.add(key)
            import warnings
            warnings.warn(
                f"fast_rmsle: {neg} of {yt.shape[0]} rows had y_true<0 or y_pred<0 "
                f"and were skipped. RMSLE is defined only on non-negative targets; "
                f"check that the target really is on a count / log-scale.",
                RuntimeWarning, stacklevel=2,
            )
    return float(val)


# ============================================================================
# MAPE / SMAPE / MdAPE / wMAPE
# ============================================================================


@numba.njit(**NUMBA_NJIT_PARAMS)
def _mape_mean_kernel_seq(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, int]:
    """Returns (mean_APE, count_zero_y_true). Mirrors sklearn's safe
    division by max(|y|, eps)."""
    eps = np.finfo(np.float64).eps
    s = 0.0
    n_zero = 0
    n = y_true.shape[0]
    for i in range(n):
        yt = y_true[i]
        if yt == 0.0:
            n_zero += 1
        denom = abs(yt)
        if denom < eps:
            denom = eps
        s += abs(y_pred[i] - yt) / denom
    return s / n if n > 0 else np.nan, n_zero


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _mape_mean_kernel_par(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, int]:
    """Parallel (prange) variant of ``_mape_mean_kernel_seq``; same eps-clamped denominator and
    (mean_APE, count_zero_y_true) return contract."""
    eps = np.finfo(np.float64).eps
    n = y_true.shape[0]
    s = 0.0
    n_zero = 0
    for i in numba.prange(n):
        yt = y_true[i]
        if yt == 0.0:
            n_zero += 1
        denom = abs(yt)
        if denom < eps:
            denom = eps
        s += abs(y_pred[i] - yt) / denom
    return s / n if n > 0 else np.nan, n_zero


_MAPE_MEAN_ZERO_WARN_SEEN: set = set()


def fast_mape_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error.

    Distinct from ``maximum_absolute_percentage_error`` in
    ``_core_precision_mape.py``: that one is the MAX |APE|, this one
    is the MEAN |APE|. Sklearn-compatible (uses np.finfo.eps for
    the zero-denominator clamp).
    """
    _check_equal_length(y_true, y_pred)
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    if yt.shape[0] == 0:
        return np.nan
    if yt.shape[0] >= _PARALLEL_REDUCTION_THRESHOLD:
        val, n_zero = _mape_mean_kernel_par(yt, yp)
    else:
        val, n_zero = _mape_mean_kernel_seq(yt, yp)
    if n_zero > 0:
        key = (int(n_zero), int(yt.shape[0]))
        if key not in _MAPE_MEAN_ZERO_WARN_SEEN:
            _MAPE_MEAN_ZERO_WARN_SEEN.add(key)
            import warnings
            warnings.warn(
                f"fast_mape_mean: {n_zero} of {yt.shape[0]} y_true entries are zero; "
                f"the eps fallback dominates those rows and the metric is biased upward.",
                RuntimeWarning, stacklevel=2,
            )
    return float(val)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _smape_kernel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sMAPE in the [0, 2] range (sklearn convention); caller multiplies by 100 for %."""
    eps = np.finfo(np.float64).eps
    s = 0.0
    n = y_true.shape[0]
    for i in range(n):
        yt = y_true[i]
        yp = y_pred[i]
        denom = abs(yt) + abs(yp)
        if denom < eps:
            denom = eps
        s += 2.0 * abs(yp - yt) / denom
    return s / n if n > 0 else np.nan


def fast_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE in [0, 2] (multiply by 100 for percent). Bounded
    even when y_true ~ 0; symmetric in under/over prediction (unlike MAPE).
    """
    _check_equal_length(y_true, y_pred)
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    if yt.shape[0] == 0:
        return np.nan
    return float(_smape_kernel(yt, yp))


def fast_mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Absolute Percentage Error.

    Outlier-robust replacement for MAPE. Uses numpy.median for the
    reduction (numba's sort is no faster than numpy's for the typical
    sizes in regression reports).
    """
    _check_equal_length(y_true, y_pred)
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    if yt.shape[0] == 0:
        return np.nan
    eps = np.finfo(np.float64).eps
    denom = np.maximum(np.abs(yt), eps)
    # denom >= eps by construction, so this never divides by zero; the errstate guards the inf/inf -> nan case
    # when y_true itself carries non-finite values (then denom is inf), letting the nan flow into the median.
    with np.errstate(invalid="ignore", divide="ignore"):
        ape = np.abs(yp - yt) / denom
    return float(np.median(ape))


@numba.njit(**NUMBA_NJIT_PARAMS)
def _wmape_kernel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE = sum(|y - p|) / sum(|y|). Unaffected by individual
    y_i == 0 (only collapses when sum(|y|) == 0).
    """
    num = 0.0
    denom = 0.0
    for i in range(y_true.shape[0]):
        num += abs(y_pred[i] - y_true[i])
        denom += abs(y_true[i])
    if denom == 0.0:
        return np.nan
    return num / denom


def fast_wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted MAPE. Avoids the y=0 epsilon-blowup problem of MAPE by
    aggregating numerator/denominator separately. The standard in
    retail forecasting / time-series demand.

    iter596: dropped the unconditional ``dtype=np.float64`` cast. The
    ``_wmape_kernel`` body is two abs+add reductions per element; mixed-
    dtype numba dispatch widens to float64 in the same place an explicit
    upfront cast would. Bench n=100k int64+float64 1.25x, float64+float64
    1.21x (no harm), float64+float32 1.59x; n=25k same direction. Same
    pattern as fast_rmse iter595. Bit-equivalent across all dtype pairs
    vs the upfront-cast baseline. Note: bench-attempt-rejected for
    ``fast_smape`` -- its kernel body has 5 ops/element (vs 2 here), so
    per-element mixed-dtype widening overhead surpasses the alloc saving
    at large n; saw 0.95x on int64+float64 @100k."""
    _check_equal_length(y_true, y_pred)
    yt = np.ascontiguousarray(y_true)
    yp = np.ascontiguousarray(y_pred)
    if yt.shape[0] == 0:
        return np.nan
    return float(_wmape_kernel(yt, yp))


# ============================================================================
# MASE
# ============================================================================


@numba.njit(**NUMBA_NJIT_PARAMS)
def _naive_mae_kernel(y_train: np.ndarray, seasonality: int) -> float:
    """Mean of |y_i - y_{i-m}| over the training set (the naive
    seasonal-difference benchmark used by MASE)."""
    n = y_train.shape[0]
    if n <= seasonality:
        return np.nan
    s = 0.0
    for i in range(seasonality, n):
        s += abs(y_train[i] - y_train[i - seasonality])
    return float(s / (n - seasonality))


def fast_mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error (Hyndman & Koehler 2006).

    MASE = MAE(y_true, y_pred) / MAE_naive(y_train, seasonality)
    where MAE_naive is the in-sample MAE of the seasonal-naive forecast.

    Scale-free, comparable across series, robust to y=0. ``seasonality``
    is the seasonal period (1 = simple naive, 7 = weekly, 12 = monthly,
    etc.) and MUST match the data's seasonality.

    NaN when the training series has no signal to scale by
    (``y_train`` shorter than seasonality, or constant).
    """
    _check_equal_length(y_true, y_pred)
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    ytr = np.ascontiguousarray(y_train, dtype=np.float64)
    if yt.shape[0] == 0:
        return np.nan
    mae = float(np.mean(np.abs(yp - yt)))
    scale = float(_naive_mae_kernel(ytr, int(seasonality)))
    if not np.isfinite(scale) or scale == 0.0:
        return np.nan
    return mae / scale


# ============================================================================
# MBE / CV(RMSE) / NSE / Explained Variance / Huber
# ============================================================================


@numba.njit(**NUMBA_NJIT_PARAMS)
def _mean_bias_error_kernel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean of (y_pred - y_true); sign carries the bias direction, see ``fast_mean_bias_error``."""
    s = 0.0
    for i in range(y_true.shape[0]):
        s += y_pred[i] - y_true[i]
    return s / y_true.shape[0] if y_true.shape[0] > 0 else np.nan


def fast_mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Signed mean residual: positive = systematic over-prediction;
    negative = systematic under-prediction. Single number summarising
    bias direction that |residuals| (MAE/RMSE) cannot show.

    iter597: dropped the unconditional ``dtype=np.float64`` cast (same
    pattern as iter595 fast_rmse / iter596 fast_wmape). Kernel is a
    single sub+add reduction per element; numba's per-signature dispatch
    handles mixed dtypes natively. Bench n=100k: int64+float64 1.54x,
    float64+float64 1.02x (no harm), float64+float32 1.88x. Bit-equiv
    across all dtype pairs."""
    _check_equal_length(y_true, y_pred)
    yt = np.ascontiguousarray(y_true)
    yp = np.ascontiguousarray(y_pred)
    if yt.shape[0] == 0:
        return np.nan
    return float(_mean_bias_error_kernel(yt, yp))


def fast_cv_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of variation of RMSE: RMSE / mean(y_true).

    Unit-free relative-error metric. NaN when mean(y_true) <= 0 (the
    ratio loses meaning).
    """
    _check_equal_length(y_true, y_pred)
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    if yt.shape[0] == 0:
        return np.nan
    mu = float(np.mean(yt))
    if mu == 0.0:
        return np.nan
    diff = yp - yt
    rmse = float(sqrt(np.mean(diff * diff)))
    return rmse / abs(mu)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _nash_sutcliffe_kernel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NSE = 1 - sum((y - p)^2) / sum((y - mean(y))^2).
    Algebraically equivalent to R^2 from sklearn convention but emitted
    explicitly because hydrology / climate / earth-science literature
    reports NSE not R^2.
    """
    n = y_true.shape[0]
    if n == 0:
        return np.nan
    mu = 0.0
    for i in range(n):
        mu += y_true[i]
    mu /= n
    num = 0.0
    denom = 0.0
    for i in range(n):
        d_res = y_true[i] - y_pred[i]
        d_tot = y_true[i] - mu
        num += d_res * d_res
        denom += d_tot * d_tot
    if denom == 0.0:
        return np.nan
    return 1.0 - num / denom


def fast_nash_sutcliffe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency. Same math as R^2 (sklearn convention)
    but the conventional reporting name in hydrology / climate work.

    iter597: dropped the unconditional ``dtype=np.float64`` cast (same
    pattern as iter595/596). Kernel is two scalar reductions over the
    same arrays; numba dispatches on mixed-dtype signatures natively.
    Bench n=100k: int64+float64 1.36x, float64+float64 1.00x (no harm),
    float64+float32 1.34x. Bit-equiv across all dtype pairs. Note:
    bench-attempt-rejected for ``fast_log_loss_binary`` (kernel has
    log/cmp/clip per element -- the heavier per-element work makes
    mixed-dtype widening overhead dominate the alloc saving; saw
    0.91-0.98x on int+float64 paths). Pattern works on simple-body
    kernels only."""
    _check_equal_length(y_true, y_pred)
    yt = np.ascontiguousarray(y_true)
    yp = np.ascontiguousarray(y_pred)
    return float(_nash_sutcliffe_kernel(yt, yp))


@numba.njit(**NUMBA_NJIT_PARAMS)
def _explained_variance_kernel(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """EV = 1 - Var(y_true - y_pred) / Var(y_true).

    Differs from R^2 when residuals are biased (E[y - p] != 0): R^2
    penalises bias via SSE, EV only counts residual *spread*. Both
    coincide for unbiased models.
    """
    n = y_true.shape[0]
    if n < 2:
        return np.nan
    # Pass 1: means.
    mu_y = 0.0
    mu_d = 0.0
    for i in range(n):
        mu_y += y_true[i]
        mu_d += y_true[i] - y_pred[i]
    mu_y /= n
    mu_d /= n
    # Pass 2: variances.
    var_y = 0.0
    var_d = 0.0
    for i in range(n):
        d_y = y_true[i] - mu_y
        d_d = (y_true[i] - y_pred[i]) - mu_d
        var_y += d_y * d_y
        var_d += d_d * d_d
    if var_y == 0.0:
        return np.nan
    return 1.0 - var_d / var_y


def fast_explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Explained variance score (sklearn convention).

    iter606: dropped the unconditional ``dtype=np.float64`` cast (same
    pattern as iter595/596/597/598). Kernel has two scalar reductions
    over the same arrays; numba dispatches on mixed-dtype signatures
    natively. Bench n=100k: int64+float64 1.20x, float64+float64 0.98x
    (borderline noise band), float64+float32 1.28x. Bit-equiv.
    Note: bench-attempt-rejected for ``fast_huber_loss`` (4 ops/element
    -- the abs+cmp+mul-add body sits at the boundary of the safe band;
    saw 0.91x on float64+float64 @100k). The huber wrapper keeps its
    upfront cast."""
    _check_equal_length(y_true, y_pred)
    yt = np.ascontiguousarray(y_true)
    yp = np.ascontiguousarray(y_pred)
    return float(_explained_variance_kernel(yt, yp))


def fast_adjusted_r2_score(y_true: np.ndarray, y_pred: np.ndarray, n_predictors: int) -> float:
    """Adjusted R^2 (Wherry / Ezekiel): ``1 - (1 - R^2) * (n - 1) / (n - p - 1)``.

    Plain R^2 has a known upward bias on the fitting sample: it never decreases when predictors are added, so it
    overstates explained variance whenever the predictor count ``p`` is non-trivial relative to ``n`` -- a junk
    feature that contributes nothing still nudges R^2 up. Adjusted R^2 penalises by the model degrees of freedom and
    is the standard small-sample correction (bench `bench_adjusted_r2.py`: with true R^2=0.5 and p/n>=0.1, mean
    |estimate - true| drops 0.159 -> 0.109, adjusted wins 27/35 cells across n in {40,50,100} x p in {8,10,20,30};
    at small p/n it converges back to plain R^2, no harm). Use this -- not ``fast_r2_score`` -- whenever you report a
    goodness-of-fit on the SAME rows the model was fit on and ``p`` is meaningful relative to ``n``.

    ``n_predictors`` is the number of independent regressors (excluding the intercept). The correction needs
    ``n - p - 1 > 0``; when ``p >= n - 1`` the denominator is non-positive (the model can perfectly interpolate and
    R^2 is meaningless) and the function returns NaN. ``p <= 0`` returns plain R^2 unchanged.
    """
    r2 = fast_r2_score(y_true, y_pred)
    n = int(np.asarray(y_true).shape[0])
    p = int(n_predictors)
    if p <= 0:
        return float(r2)
    denom = n - p - 1
    if denom <= 0:
        return float("nan")
    return float(1.0 - (1.0 - r2) * (n - 1) / denom)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _huber_loss_kernel(y_true: np.ndarray, y_pred: np.ndarray, delta: float) -> float:
    """Mean Huber loss: quadratic below ``delta``, linear (shifted to stay continuous) above it. See ``fast_huber_loss``."""
    s = 0.0
    for i in range(y_true.shape[0]):
        r = abs(y_true[i] - y_pred[i])
        if r <= delta:
            s += 0.5 * r * r
        else:
            s += delta * (r - 0.5 * delta)
    return s / y_true.shape[0] if y_true.shape[0] > 0 else np.nan


def fast_huber_loss(
    y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0,
) -> float:
    """Huber loss with transition point ``delta``.

    Quadratic for |residual| <= delta, linear above. Robust to outliers
    relative to MSE. Default delta=1.0; callers should set delta to the
    median |residual| of the training set for principled tuning.
    """
    _check_equal_length(y_true, y_pred)
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    return float(_huber_loss_kernel(yt, yp, float(delta)))


# ============================================================================
# Pearson / Kendall / Spearman wrapper
# ============================================================================


from ._regression_corr import (
    _pearson_corr_kernel,
    fast_pearson_corr,
    fast_spearman_corr,
    _kendall_tau_b_kernel,
    fast_kendall_tau,
    fast_concordance_index,
)

# ============================================================================
# Fused single-pass extended block
# ============================================================================
#
# The legacy ``fast_regression_metrics_block`` in ``_regression_metrics.py``
# fuses MAE/RMSE/MaxError/R^2 into a 2-pass kernel. This extended variant
# adds 8 more metrics into the SAME 2 passes (no extra memory walks):
#     MBE, MAPE_mean, SMAPE, wMAPE, CV(RMSE), NSE,
#     Pearson r, ExplainedVariance.
#
# Bench (mlframe.metrics._benchmarks.bench_extended_metric_blocks) - measured
# on Win11 / numba 0.58 / 16-thread Ryzen 2026-05-28:
#   N=10k:   separate 12 calls=0.37 ms   fused=0.06 ms   (5.80x)
#   N=500k:  separate 12 calls=13.68 ms  fused=1.75 ms   (7.82x)
#   N=5M:    separate 12 calls=121.3 ms  fused=11.5 ms   (10.51x)
# Numerical equivalence vs separate calls: max |diff| < 1e-12 in all sizes.
# Speedup dominated by RAM-walk reuse: 1 pass to load (y, p) into L2 +
# accumulate 9 scalars beats 12 kernels that each re-touch the same arrays.
# Trade-off accepted: pass 2 needs (y_mean, p_mean) from pass 1 first.
# A single-pass merge IS numerically possible via Welford / online co-moment
# updates (no un-centred cancellation) -- bench-attempt-rejected (2026-06-14):
# the fully-fused single-pass kernel was 1.06x@mean=0 / 0.99x@mean=11500 e2e
# at N=10M (identity ~1e-13), because pass 1 here is ALU-bound on the MAPE /
# SMAPE divisions, so dropping pass 2's memory read buys nothing while the
# per-element Welford divisions add cost. See
# ``_benchmarks/bench_fused_regression_ext_welford.py``. The cheap-win variant
# is the plain 4-metric block (``_fused_regression_welford_*`` in
# _regression_metrics.py), which IS ALU-light and gets ~1.6-1.8x from the same
# fold.


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fused_regression_ext_pass1_seq(y_true: np.ndarray, y_pred: np.ndarray):
    """Pass 1: 9-tuple of scalar accumulators + zero-y count.

    Returns (sum_abs, sum_sqr, max_abs, sum_y, sum_p, sum_signed,
             sum_ape, sum_smape, sum_abs_y, n_zero_y).
    """
    n = y_true.shape[0]
    eps = np.finfo(np.float64).eps
    sum_abs = 0.0
    sum_sqr = 0.0
    max_abs = 0.0
    sum_y = 0.0
    sum_p = 0.0
    sum_signed = 0.0
    sum_ape = 0.0
    sum_smape = 0.0
    sum_abs_y = 0.0
    n_zero_y = 0
    for i in range(n):
        yt = y_true[i]
        yp = y_pred[i]
        err = yt - yp
        abs_err = err if err >= 0.0 else -err
        sum_abs += abs_err
        sum_sqr += err * err
        if abs_err > max_abs:
            max_abs = abs_err
        sum_y += yt
        sum_p += yp
        sum_signed += yp - yt
        abs_y = yt if yt >= 0.0 else -yt
        sum_abs_y += abs_y
        denom_m = abs_y if abs_y >= eps else eps
        sum_ape += abs_err / denom_m
        if yt == 0.0:
            n_zero_y += 1
        abs_p = yp if yp >= 0.0 else -yp
        denom_s = abs_y + abs_p
        if denom_s < eps:
            denom_s = eps
        sum_smape += 2.0 * abs_err / denom_s
    return (sum_abs, sum_sqr, max_abs, sum_y, sum_p, sum_signed, sum_ape, sum_smape, sum_abs_y, n_zero_y)


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fused_regression_ext_pass1_par(
    y_true: np.ndarray, y_pred: np.ndarray, n_threads: int,
):
    """Parallel variant. Per-thread accumulators dodge the racy ``max``
    auto-reduction (numba only auto-reduces SUM-style ops safely)."""
    n = y_true.shape[0]
    eps = np.finfo(np.float64).eps
    chunk = (n + n_threads - 1) // n_threads
    l_sum_abs = np.zeros(n_threads, dtype=np.float64)
    l_sum_sqr = np.zeros(n_threads, dtype=np.float64)
    l_max_abs = np.zeros(n_threads, dtype=np.float64)
    l_sum_y = np.zeros(n_threads, dtype=np.float64)
    l_sum_p = np.zeros(n_threads, dtype=np.float64)
    l_sum_signed = np.zeros(n_threads, dtype=np.float64)
    l_sum_ape = np.zeros(n_threads, dtype=np.float64)
    l_sum_smape = np.zeros(n_threads, dtype=np.float64)
    l_sum_abs_y = np.zeros(n_threads, dtype=np.float64)
    l_n_zero = np.zeros(n_threads, dtype=np.int64)
    for tid in numba.prange(n_threads):
        s_abs = 0.0; s_sqr = 0.0; m_abs = 0.0
        s_y = 0.0; s_p = 0.0; s_signed = 0.0
        s_ape = 0.0; s_smape = 0.0; s_abs_y = 0.0
        n_z = 0
        start = tid * chunk
        end = start + chunk
        if end > n:
            end = n
        for i in range(start, end):
            yt = y_true[i]
            yp = y_pred[i]
            err = yt - yp
            abs_err = err if err >= 0.0 else -err
            s_abs += abs_err
            s_sqr += err * err
            if abs_err > m_abs:
                m_abs = abs_err
            s_y += yt
            s_p += yp
            s_signed += yp - yt
            abs_y = yt if yt >= 0.0 else -yt
            s_abs_y += abs_y
            denom_m = abs_y if abs_y >= eps else eps
            s_ape += abs_err / denom_m
            if yt == 0.0:
                n_z += 1
            abs_p = yp if yp >= 0.0 else -yp
            denom_s = abs_y + abs_p
            if denom_s < eps:
                denom_s = eps
            s_smape += 2.0 * abs_err / denom_s
        l_sum_abs[tid] = s_abs
        l_sum_sqr[tid] = s_sqr
        l_max_abs[tid] = m_abs
        l_sum_y[tid] = s_y
        l_sum_p[tid] = s_p
        l_sum_signed[tid] = s_signed
        l_sum_ape[tid] = s_ape
        l_sum_smape[tid] = s_smape
        l_sum_abs_y[tid] = s_abs_y
        l_n_zero[tid] = n_z
    sum_abs = 0.0; sum_sqr = 0.0; max_abs = 0.0
    sum_y = 0.0; sum_p = 0.0; sum_signed = 0.0
    sum_ape = 0.0; sum_smape = 0.0; sum_abs_y = 0.0
    n_zero_y = 0
    for tid in range(n_threads):
        sum_abs += l_sum_abs[tid]
        sum_sqr += l_sum_sqr[tid]
        if l_max_abs[tid] > max_abs:
            max_abs = l_max_abs[tid]
        sum_y += l_sum_y[tid]
        sum_p += l_sum_p[tid]
        sum_signed += l_sum_signed[tid]
        sum_ape += l_sum_ape[tid]
        sum_smape += l_sum_smape[tid]
        sum_abs_y += l_sum_abs_y[tid]
        n_zero_y += l_n_zero[tid]
    return (sum_abs, sum_sqr, max_abs, sum_y, sum_p, sum_signed, sum_ape, sum_smape, sum_abs_y, n_zero_y)


@numba.njit(**NUMBA_NJIT_PARAMS)
def _fused_regression_ext_pass2_seq(
    y_true: np.ndarray, y_pred: np.ndarray, y_mean: float, p_mean: float,
):
    """Pass 2: centred sums for R^2 / Pearson / EV.

    Returns (ss_tot, ss_pred_centred, sxy, ss_resid_centred).
    """
    n = y_true.shape[0]
    r_mean = y_mean - p_mean
    ss_tot = 0.0
    ss_pred = 0.0
    sxy = 0.0
    ss_resid = 0.0
    for i in range(n):
        dy = y_true[i] - y_mean
        dp = y_pred[i] - p_mean
        ss_tot += dy * dy
        ss_pred += dp * dp
        sxy += dy * dp
        dr = (y_true[i] - y_pred[i]) - r_mean
        ss_resid += dr * dr
    return ss_tot, ss_pred, sxy, ss_resid


@numba.njit(**NUMBA_NJIT_PARAMS, parallel=True)
def _fused_regression_ext_pass2_par(
    y_true: np.ndarray, y_pred: np.ndarray, y_mean: float, p_mean: float,
):
    """Parallel pass 2. Sums are commutative and float64-stable across
    threads at the sizes we use; no per-thread arrays needed."""
    n = y_true.shape[0]
    r_mean = y_mean - p_mean
    ss_tot = 0.0
    ss_pred = 0.0
    sxy = 0.0
    ss_resid = 0.0
    for i in numba.prange(n):
        dy = y_true[i] - y_mean
        dp = y_pred[i] - p_mean
        ss_tot += dy * dy
        ss_pred += dp * dp
        sxy += dy * dp
        dr = (y_true[i] - y_pred[i]) - r_mean
        ss_resid += dr * dr
    return ss_tot, ss_pred, sxy, ss_resid


def fast_regression_metrics_block_extended(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> dict:
    """Compute 12 regression metrics in two fused passes.

    Returns a dict with:
        MAE, RMSE, MSE, MaxError, R2,
        MBE, MAPE_mean, SMAPE, wMAPE, CV_RMSE,
        NSE, Pearson, ExplainedVariance
    plus internal ``_n_zero_y`` (count of y_true==0, surfaces the
    MAPE epsilon-blowup risk to the caller).

    NSE numerically equals R^2 under sklearn's convention (same formula);
    we expose both because hydrology / climate / earth-science reports
    use the NSE label and skipping it forces external code to alias.

    1-D inputs only. For 2-D multioutput regression the caller should
    fall through to the per-output ``fast_*`` helpers.
    """
    _check_equal_length(y_true, y_pred)
    yt = np.ascontiguousarray(np.asarray(y_true), dtype=np.float64)
    yp = np.ascontiguousarray(np.asarray(y_pred), dtype=np.float64)
    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError("fast_regression_metrics_block_extended expects 1-D arrays; got shapes " f"y_true={yt.shape}, y_pred={yp.shape}.")
    n = yt.shape[0]
    if n == 0:
        return {
            "MAE": np.nan, "RMSE": np.nan, "MSE": np.nan,
            "MaxError": np.nan, "R2": np.nan,
            "MBE": np.nan, "MAPE_mean": np.nan, "SMAPE": np.nan,
            "wMAPE": np.nan, "CV_RMSE": np.nan,
            "NSE": np.nan, "Pearson": np.nan, "ExplainedVariance": np.nan,
            "_n_zero_y": 0,
        }
    use_par = n >= _PARALLEL_REDUCTION_THRESHOLD
    if use_par:
        nthr = numba.get_num_threads()
        sum_abs, sum_sqr, max_abs, sum_y, sum_p, sum_signed, sum_ape, sum_smape, sum_abs_y, n_zero_y = _fused_regression_ext_pass1_par(yt, yp, nthr)
    else:
        sum_abs, sum_sqr, max_abs, sum_y, sum_p, sum_signed, sum_ape, sum_smape, sum_abs_y, n_zero_y = _fused_regression_ext_pass1_seq(yt, yp)
    y_mean = sum_y / n
    p_mean = sum_p / n
    if use_par:
        ss_tot, ss_pred, sxy, ss_resid = _fused_regression_ext_pass2_par(yt, yp, y_mean, p_mean)
    else:
        ss_tot, ss_pred, sxy, ss_resid = _fused_regression_ext_pass2_seq(yt, yp, y_mean, p_mean)

    mse = sum_sqr / n
    mae = sum_abs / n
    rmse = float(sqrt(mse))
    if ss_tot <= 0.0:
        r2 = 0.0 if sum_sqr == 0.0 else float("-inf")
        ev = np.nan
        nse = r2
    else:
        r2 = 1.0 - sum_sqr / ss_tot
        ev = 1.0 - ss_resid / ss_tot
        nse = r2
    pearson = sxy / sqrt(ss_tot * ss_pred) if ss_tot > 0.0 and ss_pred > 0.0 else np.nan
    cv_rmse = rmse / abs(y_mean) if y_mean != 0.0 else np.nan
    mbe = sum_signed / n
    mape_mean = sum_ape / n
    smape = sum_smape / n
    wmape = sum_abs / sum_abs_y if sum_abs_y > 0.0 else np.nan

    return {
        "MAE": float(mae),
        "RMSE": rmse,
        "MSE": float(mse),
        "MaxError": float(max_abs),
        "R2": float(r2),
        "MBE": float(mbe),
        "MAPE_mean": float(mape_mean),
        "SMAPE": float(smape),
        "wMAPE": float(wmape),
        "CV_RMSE": float(cv_rmse),
        "NSE": float(nse),
        "Pearson": float(pearson),
        "ExplainedVariance": float(ev),
        "_n_zero_y": int(n_zero_y),
    }

from ._regression_deviance import (
    _maybe_warn_tweedie,
    _tweedie_deviance_gamma_kernel,
    _tweedie_deviance_general_kernel,
    _tweedie_deviance_poisson_kernel,
    fast_gamma_deviance,
    fast_poisson_deviance,
    fast_tweedie_deviance,
)
