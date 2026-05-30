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

from math import sqrt, log1p, log
from typing import Optional, Tuple

import numpy as np
import numba

from ._numba_params import NUMBA_NJIT_PARAMS, _PARALLEL_REDUCTION_THRESHOLD
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
from .rank_correlation import _spearmanr_batched_numpy


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
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    if yt.shape[0] == 0:
        return np.nan
    eps = np.finfo(np.float64).eps
    denom = np.maximum(np.abs(yt), eps)
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
    retail forecasting / time-series demand."""
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
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
    return s / (n - seasonality)


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
    s = 0.0
    for i in range(y_true.shape[0]):
        s += y_pred[i] - y_true[i]
    return s / y_true.shape[0] if y_true.shape[0] > 0 else np.nan


def fast_mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Signed mean residual: positive = systematic over-prediction;
    negative = systematic under-prediction. Single number summarising
    bias direction that |residuals| (MAE/RMSE) cannot show.
    """
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    if yt.shape[0] == 0:
        return np.nan
    return float(_mean_bias_error_kernel(yt, yp))


def fast_cv_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of variation of RMSE: RMSE / mean(y_true).

    Unit-free relative-error metric. NaN when mean(y_true) <= 0 (the
    ratio loses meaning).
    """
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
    but the conventional reporting name in hydrology / climate work."""
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
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
    """Explained variance score (sklearn convention)."""
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    return float(_explained_variance_kernel(yt, yp))


@numba.njit(**NUMBA_NJIT_PARAMS)
def _huber_loss_kernel(y_true: np.ndarray, y_pred: np.ndarray, delta: float) -> float:
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
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    return float(_huber_loss_kernel(yt, yp, float(delta)))


# ============================================================================
# Pearson / Kendall / Spearman wrapper
# ============================================================================


@numba.njit(**NUMBA_NJIT_PARAMS)
def _pearson_corr_kernel(x: np.ndarray, y: np.ndarray) -> float:
    n = x.shape[0]
    if n < 2:
        return np.nan
    mx = 0.0
    my = 0.0
    for i in range(n):
        mx += x[i]
        my += y[i]
    mx /= n
    my /= n
    sxy = 0.0
    sxx = 0.0
    syy = 0.0
    for i in range(n):
        dx = x[i] - mx
        dy = y[i] - my
        sxy += dx * dy
        sxx += dx * dx
        syy += dy * dy
    denom = sqrt(sxx * syy)
    if denom == 0.0:
        return np.nan
    return sxy / denom


def fast_pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson product-moment correlation between y_true and y_pred.

    R^2 is correlation^2 for an UNBIASED model; for biased models R^2 and
    Pearson r differ - report both so the reader can see if the model
    has a scale/bias mismatch on top of its rank quality.
    """
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    return float(_pearson_corr_kernel(yt, yp))


def fast_spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation (scalar wrapper).

    Reuses ``mlframe.metrics.rank_correlation._spearmanr_batched_numpy``
    on a 1-row batch to avoid duplicating the rank+Pearson pipeline.
    Tied values handled via average-rank.
    """
    yt = np.asarray(y_true, dtype=np.float64).reshape(1, -1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(1, -1)
    if yt.shape[1] < 2:
        return np.nan
    # iter592: ``_spearmanr_batched_numpy`` import hoisted to module-level
    # at the top of this file. The lazy-import here paid 140 ms / call
    # on the cold-cache first invocation per the c0103 profile.
    return float(_spearmanr_batched_numpy(yt, yp)[0])


@numba.njit(**NUMBA_NJIT_PARAMS)
def _kendall_tau_b_kernel(x: np.ndarray, y: np.ndarray) -> float:
    """O(N^2) Kendall's tau-b. Adequate for N up to ~5000; above that
    callers should batch via scipy.stats.kendalltau which uses a
    merge-sort O(N log N) implementation.

    tau_b corrects for ties: (concordant - discordant) /
    sqrt((P - T_x)(P - T_y)) where P = N(N-1)/2.
    """
    n = x.shape[0]
    if n < 2:
        return np.nan
    concordant = 0
    discordant = 0
    tx = 0  # ties only in x
    ty = 0  # ties only in y
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0.0 and dy == 0.0:
                # tie in both - excluded from all denominators
                pass
            elif dx == 0.0:
                tx += 1
            elif dy == 0.0:
                ty += 1
            elif (dx > 0.0 and dy > 0.0) or (dx < 0.0 and dy < 0.0):
                concordant += 1
            else:
                discordant += 1
    total_pairs = n * (n - 1) // 2
    denom_x = total_pairs - tx
    denom_y = total_pairs - ty
    if denom_x <= 0 or denom_y <= 0:
        return np.nan
    return (concordant - discordant) / sqrt(float(denom_x) * float(denom_y))


_KENDALL_NUMBA_MAX_N = 500


def fast_kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kendall's tau-b (tie-corrected) rank correlation.

    Below ``_KENDALL_NUMBA_MAX_N`` rows the in-process O(N^2) numba kernel
    wins (its tight machine-code loop beats scipy's per-call dispatch); above
    that scipy's merge-sort O(N log N) ``kendalltau`` is dramatically faster.
    The threshold was 5000 historically, but a re-bench on modern scipy /
    numba (2026-05-28) puts the crossover at N~400: scipy beats the numba
    kernel 1.38x at N=400, 12.8x at N=1500, 41x at N=3000, 54x at N=5000.
    Values are identical to 4 decimals across the range (both implement the
    same tie-corrected tau-b formula), so changing the threshold is bit-
    equivalent for the returned scalar but unlocks the O(N log N) algorithm
    on every typical regression-metric shape.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape[0] < 2:
        return np.nan
    if yt.shape[0] <= _KENDALL_NUMBA_MAX_N:
        return float(_kendall_tau_b_kernel(yt, yp))
    from scipy.stats import kendalltau
    res = kendalltau(yt, yp, variant="b")
    return float(res.correlation if hasattr(res, "correlation") else res[0])


# ============================================================================
# Concordance index (C-index)
# ============================================================================


def fast_concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """C-index = fraction of concordant pairs (ignoring tied y_true).

    Range [0, 1]; 0.5 = chance; 1.0 = perfect rank agreement. Equivalent
    to (Kendall tau-b + 1) / 2 after tie correction; emitted as a
    separate metric because survival / risk modelling reports C-index,
    not Kendall tau.

    For N <= 5000 uses the O(N^2) numba kernel below; for larger N falls
    back to the tau-b reduction (O(N log N) via scipy).
    """
    if y_true.shape[0] < 2:
        return np.nan
    tau = fast_kendall_tau(y_true, y_pred)
    if not np.isfinite(tau):
        return np.nan
    return (tau + 1.0) / 2.0


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
# Trade-off accepted: pass 2 needs (y_mean, p_mean) from pass 1 first,
# so pass 1 and pass 2 cannot be merged into one walk without losing
# numerical stability (the un-centred sum-of-squares identity cancels on
# float64 when y has a large mean - see _regression_metrics.py comment).


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
    return (sum_abs, sum_sqr, max_abs, sum_y, sum_p, sum_signed,
            sum_ape, sum_smape, sum_abs_y, n_zero_y)


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
    return (sum_abs, sum_sqr, max_abs, sum_y, sum_p, sum_signed,
            sum_ape, sum_smape, sum_abs_y, n_zero_y)


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
    yt = np.ascontiguousarray(np.asarray(y_true), dtype=np.float64)
    yp = np.ascontiguousarray(np.asarray(y_pred), dtype=np.float64)
    if yt.ndim != 1 or yp.ndim != 1:
        raise ValueError(
            "fast_regression_metrics_block_extended expects 1-D arrays; got shapes "
            f"y_true={yt.shape}, y_pred={yp.shape}."
        )
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
        (sum_abs, sum_sqr, max_abs, sum_y, sum_p, sum_signed,
         sum_ape, sum_smape, sum_abs_y, n_zero_y) = _fused_regression_ext_pass1_par(yt, yp, nthr)
    else:
        (sum_abs, sum_sqr, max_abs, sum_y, sum_p, sum_signed,
         sum_ape, sum_smape, sum_abs_y, n_zero_y) = _fused_regression_ext_pass1_seq(yt, yp)
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


# ============================================================================
# Tweedie / Poisson / Gamma deviances (Tier 2)
# ============================================================================
#
# GLM deviances for non-Gaussian targets:
#   power=0: Normal (= MSE)
#   power=1: Poisson (count data, insurance claim counts, click counts)
#   power=2: Gamma (positive continuous, claim severity, lifetime)
#   power=3: Inverse Gaussian
#   1 < power < 2: compound Poisson-Gamma (insurance pure-premium)
# Formula follows sklearn's mean_tweedie_deviance exactly (Pregibon 1984):
#   D(y, p) = 2 * mean of:
#     power=0:   (y - p)^2
#     power=1:   y * log(y/p) - (y - p)              (with 0*log0 = 0)
#     power=2:   log(p/y) + (y - p)/p
#     general:   max(y,0)^(2-p)/((1-p)(2-p)) - y*p^(1-p)/(1-p) + p^(2-p)/(2-p)
# All require strictly positive predictions for p>=1, and non-negative y;
# kernels return NaN with a count of skipped rows when constraints break.


@numba.njit(**NUMBA_NJIT_PARAMS)
def _tweedie_deviance_poisson_kernel(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> Tuple[float, int]:
    """Returns (deviance, count_invalid). Skips rows where y_pred<=0 OR y<0.
    y=0 is fine: y*log(y) is taken as 0 by convention."""
    s = 0.0
    used = 0
    invalid = 0
    for i in range(y_true.shape[0]):
        yt = y_true[i]
        yp = y_pred[i]
        if yp <= 0.0 or yt < 0.0:
            invalid += 1
            continue
        if yt == 0.0:
            term = -(yt - yp)  # = yp; y*log(y/p) is 0
        else:
            term = yt * log(yt / yp) - (yt - yp)
        s += 2.0 * term
        used += 1
    return (s / used) if used > 0 else np.nan, invalid


@numba.njit(**NUMBA_NJIT_PARAMS)
def _tweedie_deviance_gamma_kernel(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> Tuple[float, int]:
    """Gamma deviance: power=2. Requires y > 0 AND p > 0 (the log
    is undefined at 0 for both)."""
    s = 0.0
    used = 0
    invalid = 0
    for i in range(y_true.shape[0]):
        yt = y_true[i]
        yp = y_pred[i]
        if yp <= 0.0 or yt <= 0.0:
            invalid += 1
            continue
        s += 2.0 * (log(yp / yt) + (yt - yp) / yp)
        used += 1
    return (s / used) if used > 0 else np.nan, invalid


@numba.njit(**NUMBA_NJIT_PARAMS)
def _tweedie_deviance_general_kernel(
    y_true: np.ndarray, y_pred: np.ndarray, power: float,
) -> Tuple[float, int]:
    """General 1 < power < 2 OR power > 2 case. Sklearn convention
    requires y >= 0 (negative not in support of distribution) AND p > 0
    (predictions must be strictly positive for log/power terms)."""
    s = 0.0
    used = 0
    invalid = 0
    p = power
    for i in range(y_true.shape[0]):
        yt = y_true[i]
        yp = y_pred[i]
        if yp <= 0.0 or yt < 0.0:
            invalid += 1
            continue
        # max(y, 0)^(2-p): yt >= 0 here, so use yt directly.
        if yt == 0.0:
            term_y = 0.0  # 0^(2-p) for 1<p<2: power > 0 -> 0; for p>2: power <0 -> inf, but
                          # sklearn special-cases this to 0 (sup-zero discontinuity).
        else:
            term_y = (yt ** (2.0 - p)) / ((1.0 - p) * (2.0 - p))
        term_yp = yt * (yp ** (1.0 - p)) / (1.0 - p)
        term_p = (yp ** (2.0 - p)) / (2.0 - p)
        s += 2.0 * (term_y - term_yp + term_p)
        used += 1
    return (s / used) if used > 0 else np.nan, invalid


_TWEEDIE_WARN_SEEN: set = set()


def _maybe_warn_tweedie(name: str, invalid: int, total: int) -> None:
    if invalid <= 0:
        return
    key = (name, int(invalid), int(total))
    if key in _TWEEDIE_WARN_SEEN:
        return
    _TWEEDIE_WARN_SEEN.add(key)
    import warnings
    warnings.warn(
        f"{name}: {invalid} of {total} rows skipped (y_pred<=0 or y_true out of support); "
        f"check that the model emits strictly positive predictions matching the target's "
        f"distributional support.",
        RuntimeWarning, stacklevel=3,
    )


def fast_poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tweedie deviance at power=1 (Poisson).

    Use for count targets where the variance scales with the mean
    (claim counts, calls per hour, click counts). Lower is better.
    Equivalent to sklearn's ``mean_poisson_deviance``.

    Rows with y_pred <= 0 or y_true < 0 are skipped with a rate-limited
    warning - silently dropping them masks model misspecification.
    """
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    if yt.shape[0] == 0:
        return np.nan
    val, invalid = _tweedie_deviance_poisson_kernel(yt, yp)
    _maybe_warn_tweedie("fast_poisson_deviance", invalid, yt.shape[0])
    return float(val)


def fast_gamma_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tweedie deviance at power=2 (Gamma).

    Use for positive continuous targets where the variance scales with
    the mean SQUARED (claim severity, lifetimes, financial losses).
    Lower is better. Equivalent to sklearn's ``mean_gamma_deviance``.

    Rows with y_pred <= 0 or y_true <= 0 are skipped (log is undefined
    at 0 for either) with a rate-limited warning.
    """
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    if yt.shape[0] == 0:
        return np.nan
    val, invalid = _tweedie_deviance_gamma_kernel(yt, yp)
    _maybe_warn_tweedie("fast_gamma_deviance", invalid, yt.shape[0])
    return float(val)


def fast_tweedie_deviance(
    y_true: np.ndarray, y_pred: np.ndarray, *, power: float = 0.0,
) -> float:
    """General Tweedie deviance at arbitrary power.

    Common values:
      power=0  -> Normal (= MSE) - falls through to the simple kernel
      power=1  -> Poisson (use ``fast_poisson_deviance`` for the dedicated path)
      power=2  -> Gamma   (use ``fast_gamma_deviance`` for the dedicated path)
      1<p<2    -> compound Poisson-Gamma (insurance pure-premium)

    Rows with y_pred <= 0 or y_true outside the distributional support
    are skipped with a rate-limited warning. Equivalent to sklearn's
    ``mean_tweedie_deviance(y_true, y_pred, power=power)``.
    """
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    if yt.shape[0] == 0:
        return np.nan
    if power == 0.0:
        d = yt - yp
        return float(np.mean(d * d))
    if power == 1.0:
        return fast_poisson_deviance(y_true, y_pred)
    if power == 2.0:
        return fast_gamma_deviance(y_true, y_pred)
    if power < 1.0:
        raise ValueError(
            f"Tweedie power must be 0 or >= 1 (Pregibon 1984); got power={power}. "
            "Use power=0 for Normal (MSE) or power in [1, inf) for GLM."
        )
    val, invalid = _tweedie_deviance_general_kernel(yt, yp, float(power))
    _maybe_warn_tweedie(f"fast_tweedie_deviance(power={power})", invalid, yt.shape[0])
    return float(val)
