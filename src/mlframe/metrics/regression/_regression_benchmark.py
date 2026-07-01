"""Benchmark-relative and threshold regression functionals (PZAD err_regression).

The regression-metrics lecture (Дьяконов 2020) groups several functionals that the
existing mlframe metrics (RMSLE / MAPE / SMAPE / MASE / wMAPE / Huber / deviances /
pinball) do not yet cover:

- **eB / epsilon-band accuracy** (slide 37): the fraction of predictions within a
  tolerance ``|y - a| <= eps`` (a QUALITY functional, higher is better). This is the
  exact "predict the amount to within $10" functional from the dunnhumby Shopper
  Challenge (see caseclients.md) and any "close enough" acceptance criterion.
- **REL_MAE / MRAE / Percent-Better** (slide 34): error measured RELATIVE to a
  benchmark predictor's errors. mlframe's MASE only benchmarks against the naive
  seasonal forecast; these benchmark against an ARBITRARY supplied prediction vector,
  answering "did my model beat this specific baseline, and by how much".
- **logcosh** (slide 18): a smooth, parameter-free Huber-like loss.

All are njit kernels with a thin validating Python wrapper, matching the module style
of ``_regression_extras``.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "fast_epsilon_band_accuracy",
    "fast_rel_mae",
    "fast_mrae",
    "fast_percent_better",
    "fast_logcosh_loss",
]

_NJIT = dict(fastmath=False, cache=True, nogil=True)


@njit(**_NJIT)
def _epsilon_band_accuracy_kernel(y_true, y_pred, eps):
    n = y_true.shape[0]
    if n == 0:
        return np.nan
    hit = 0
    for i in range(n):
        if abs(y_true[i] - y_pred[i]) <= eps:
            hit += 1
    return hit / n


def fast_epsilon_band_accuracy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float) -> float:
    """Fraction of predictions within ``|y_true - y_pred| <= epsilon`` (a quality functional in [0, 1], higher is better).

    The dunnhumby "guess the spend to within $10" acceptance metric; the optimal constant predictor is the mode of
    the Parzen density (see caseclients.md). ``epsilon`` must be >= 0.
    """
    if epsilon < 0:
        raise ValueError(f"fast_epsilon_band_accuracy: epsilon must be >= 0, got {epsilon}.")
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("fast_epsilon_band_accuracy: y_true and y_pred length mismatch.")
    return float(_epsilon_band_accuracy_kernel(yt, yp, float(epsilon)))


@njit(**_NJIT)
def _rel_mae_kernel(y_true, y_pred, y_bench):
    num = 0.0
    den = 0.0
    for i in range(y_true.shape[0]):
        num += abs(y_true[i] - y_pred[i])
        den += abs(y_true[i] - y_bench[i])
    return num / den if den > 0.0 else np.nan


def fast_rel_mae(y_true: np.ndarray, y_pred: np.ndarray, y_benchmark: np.ndarray) -> float:
    """REL_MAE: ``sum|y-pred| / sum|y-benchmark|``. < 1 means the model beats the benchmark; ratio of MAEs.

    Use a naive/constant/last-value benchmark to report skill relative to a trivial baseline (the lecture's
    "make a simple algorithm and measure error relative to it").
    """
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    yb = np.ascontiguousarray(y_benchmark, dtype=np.float64)
    if not (yt.shape[0] == yp.shape[0] == yb.shape[0]):
        raise ValueError("fast_rel_mae: y_true / y_pred / y_benchmark length mismatch.")
    return float(_rel_mae_kernel(yt, yp, yb))


@njit(**_NJIT)
def _mrae_kernel(y_true, y_pred, y_bench, eps):
    n = y_true.shape[0]
    if n == 0:
        return np.nan
    s = 0.0
    for i in range(n):
        den = abs(y_true[i] - y_bench[i])
        if den < eps:
            den = eps
        s += abs(y_true[i] - y_pred[i]) / den
    return s / n


def fast_mrae(y_true: np.ndarray, y_pred: np.ndarray, y_benchmark: np.ndarray, *, eps: float = 1e-9) -> float:
    """MRAE: mean of the per-element error ratios ``|y-pred| / |y-benchmark|`` (benchmark errors floored at ``eps``).

    Unlike REL_MAE (ratio of summed errors), MRAE averages per-point relative errors, so a few points where the
    benchmark is near-perfect dominate less when the floor is applied.
    """
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    yb = np.ascontiguousarray(y_benchmark, dtype=np.float64)
    if not (yt.shape[0] == yp.shape[0] == yb.shape[0]):
        raise ValueError("fast_mrae: length mismatch.")
    return float(_mrae_kernel(yt, yp, yb, float(eps)))


@njit(**_NJIT)
def _percent_better_kernel(y_true, y_pred, y_bench):
    n = y_true.shape[0]
    if n == 0:
        return np.nan
    better = 0
    for i in range(n):
        if abs(y_true[i] - y_pred[i]) < abs(y_true[i] - y_bench[i]):
            better += 1
    return better / n


def fast_percent_better(y_true: np.ndarray, y_pred: np.ndarray, y_benchmark: np.ndarray) -> float:
    """Percent-Better: fraction of points where the model's absolute error is strictly smaller than the benchmark's.

    A robust, scale-free "how often do we win" score in [0, 1]; 0.5 = tie with the benchmark.
    """
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    yb = np.ascontiguousarray(y_benchmark, dtype=np.float64)
    if not (yt.shape[0] == yp.shape[0] == yb.shape[0]):
        raise ValueError("fast_percent_better: length mismatch.")
    return float(_percent_better_kernel(yt, yp, yb))


@njit(**_NJIT)
def _logcosh_kernel(y_true, y_pred):
    n = y_true.shape[0]
    if n == 0:
        return np.nan
    s = 0.0
    for i in range(n):
        z = y_pred[i] - y_true[i]
        az = abs(z)
        # log(cosh(z)) = |z| + log((1 + exp(-2|z|))/2); the |z| form avoids overflow for large residuals.
        s += az + np.log((1.0 + np.exp(-2.0 * az)) * 0.5)
    return s / n


def fast_logcosh_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean ``log(cosh(pred - true))``: a smooth, parameter-free loss ~ MSE for small residuals, ~ MAE for large.

    Overflow-safe (uses the ``|z| + log((1+e^{-2|z|})/2)`` identity). A Huber-like alternative with no delta to tune.
    """
    yt = np.ascontiguousarray(y_true, dtype=np.float64)
    yp = np.ascontiguousarray(y_pred, dtype=np.float64)
    if yt.shape[0] != yp.shape[0]:
        raise ValueError("fast_logcosh_loss: length mismatch.")
    return float(_logcosh_kernel(yt, yp))
