"""Robust location estimators: redescending M-estimator mean + geometric median.

From PZAD «Оценки среднего, вероятности и плотности» (Дьяконов 2020): the arithmetic
mean is non-robust to outliers, the median is robust but (in >1D) ill-defined. The
lecture's unifying tool is the minimum-contrast / M-estimator fixed point

    a = sum_i x_i * xi(x_i - a) / sum_i xi(x_i - a)

solved by iterative reweighting, where ``xi`` is a weight function that downweights
points far from the current estimate. Meshalkin's Gaussian weight ``xi(z)=exp(-lam*z^2/2)``
(Princeton 1972 robustness study) gives a smooth *redescending* estimator: extreme
outliers get ~zero weight, so a heavy-tailed / contaminated sample is summarized by its
bulk. We also provide Huber and Tukey-biweight weights.

For the multivariate case the lecture points to the **geometric median** (spatial median /
1-median / Torricelli point), the minimizer of ``sum_i ||x_i - a||`` computed by Weiszfeld's
iteration ``a = sum_i x_i/||x_i-a|| / sum_i 1/||x_i-a||`` — rotation-invariant, reduces to the
1-D median, and far more outlier-robust than the coordinate-wise mean.

Both are reusable robust aggregators: a robust per-group summary feature, a robust ensemble
combiner, or a robust centroid for embeddings.
"""

from __future__ import annotations

import logging

import os

import numpy as np
from numba import njit, prange

logger = logging.getLogger(__name__)

__all__ = ["robust_mean_mestimator", "geometric_median", "WEIGHTS"]

WEIGHTS = ("meshalkin", "huber", "tukey")

# Parallelize the IRLS reweighting pass over samples once n amortises the prange spawn. Env-overridable.
_ROBUST_MEAN_PARALLEL_MIN_N = int(os.environ.get("MLFRAME_ROBUST_MEAN_PARALLEL_MIN_N", "50000"))


@njit(fastmath=False, cache=True)
def _mad_scale(x: np.ndarray) -> float:
    """Median-absolute-deviation scale (x1.4826 for consistency with the normal std), floored to a tiny positive."""
    n = x.shape[0]
    med = np.median(x)
    dev = np.empty(n, dtype=np.float64)
    for i in range(n):
        dev[i] = abs(x[i] - med)
    mad = np.median(dev) * 1.4826
    return mad if mad > 1e-12 else 1.0


@njit(fastmath=False, cache=True)
def _weight(z: float, wcode: int, param: float) -> float:
    """Weight for scaled residual z. 0=meshalkin exp(-lam z^2/2), 1=huber, 2=tukey biweight."""
    az = abs(z)
    if wcode == 0:
        return np.exp(-0.5 * param * z * z)
    if wcode == 1:
        return 1.0 if az <= param else param / az
    # tukey biweight
    if az >= param:
        return 0.0
    t = 1.0 - (z / param) * (z / param)
    return t * t


@njit(fastmath=False, cache=True)
def _robust_mean_irls(x: np.ndarray, wcode: int, param: float, scale: float, max_iter: int, tol: float) -> float:
    """Iteratively-reweighted 1-D location M-estimator. Init at the median; scale<=0 -> MAD."""
    n = x.shape[0]
    if n == 0:
        return np.nan
    if n == 1:
        return x[0]
    s = scale if scale > 0.0 else _mad_scale(x)
    a = np.median(x)
    for _ in range(max_iter):
        num = 0.0
        den = 0.0
        for i in range(n):
            w = _weight((x[i] - a) / s, wcode, param)
            num += w * x[i]
            den += w
        if den <= 0.0:
            break
        a_new = num / den
        if abs(a_new - a) <= tol * (abs(a) + tol):
            a = a_new
            break
        a = a_new
    return a


@njit(fastmath=False, cache=True, parallel=True)
def _robust_mean_irls_parallel(x: np.ndarray, wcode: int, param: float, scale: float, max_iter: int, tol: float) -> float:
    """prange variant of the IRLS location M-estimator; wins at large n (scalar num/den reductions)."""
    n = x.shape[0]
    if n == 0:
        return np.nan
    if n == 1:
        return x[0]
    s = scale if scale > 0.0 else _mad_scale(x)
    a = np.median(x)
    for _ in range(max_iter):
        num = 0.0
        den = 0.0
        for i in prange(n):
            w = _weight((x[i] - a) / s, wcode, param)
            num += w * x[i]
            den += w
        if den <= 0.0:
            break
        a_new = num / den
        if abs(a_new - a) <= tol * (abs(a) + tol):
            a = a_new
            break
        a = a_new
    return a


def robust_mean_mestimator(
    x: np.ndarray,
    *,
    weight: str = "meshalkin",
    param: float | None = None,
    scale: float = -1.0,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> float:
    """Robust 1-D location via an iteratively-reweighted M-estimator.

    Parameters
    ----------
    x : np.ndarray
        1-D sample.
    weight : {'meshalkin', 'huber', 'tukey'}
        Weight (xi) function. 'meshalkin' = redescending Gaussian; 'huber' = clip; 'tukey' = hard-redescending.
    param : float, optional
        Tuning constant: Meshalkin lambda (default 1.0), Huber k in scale units (default 1.345), Tukey c (default 4.685).
        The Huber/Tukey defaults give ~95% efficiency at the normal.
    scale : float
        Residual scale. ``<= 0`` -> MAD of ``x``.
    max_iter, tol : int, float
        IRLS stopping controls.

    Returns
    -------
    float
        Robust location estimate. At the identity (Huber k -> inf) this tends to the mean; Meshalkin lambda -> 0 also -> mean.
    """
    if weight not in WEIGHTS:
        raise ValueError(f"robust_mean_mestimator: weight must be one of {WEIGHTS}, got {weight!r}.")
    x = np.ascontiguousarray(x, dtype=np.float64)
    if param is None:
        param = {"meshalkin": 1.0, "huber": 1.345, "tukey": 4.685}[weight]
    param = float(param)
    if param <= 0.0:
        raise ValueError(f"robust_mean_mestimator: param must be > 0, got {param}.")
    wcode = WEIGHTS.index(weight)
    if x.shape[0] >= _ROBUST_MEAN_PARALLEL_MIN_N:
        return float(_robust_mean_irls_parallel(x, wcode, param, float(scale), int(max_iter), float(tol)))
    return float(_robust_mean_irls(x, wcode, param, float(scale), int(max_iter), float(tol)))


@njit(fastmath=False, cache=True)
def _geometric_median_weiszfeld(X: np.ndarray, max_iter: int, tol: float, eps: float) -> np.ndarray:
    """Weiszfeld iteration for the geometric median of rows of X (n x d). Init at the coordinate mean.

    Coincident-point safeguard: distances are floored at ``eps`` so a sample lying on the current
    estimate does not divide by zero (a light-touch variant of the Vardi-Zhang correction).
    """
    n, d = X.shape
    mu = np.zeros(d, dtype=np.float64)
    for j in range(d):
        s = 0.0
        for i in range(n):
            s += X[i, j]
        mu[j] = s / n
    for _ in range(max_iter):
        num = np.zeros(d, dtype=np.float64)
        den = 0.0
        for i in range(n):
            dist = 0.0
            for j in range(d):
                diff = X[i, j] - mu[j]
                dist += diff * diff
            dist = dist**0.5
            if dist < eps:
                dist = eps
            inv = 1.0 / dist
            den += inv
            for j in range(d):
                num[j] += X[i, j] * inv
        if den <= 0.0:
            break
        shift = 0.0
        for j in range(d):
            new = num[j] / den
            diff = new - mu[j]
            shift += diff * diff
            mu[j] = new
        if shift**0.5 <= tol:
            break
    return mu


def geometric_median(
    X: np.ndarray,
    *,
    max_iter: int = 200,
    tol: float = 1e-7,
    eps: float = 1e-10,
) -> np.ndarray:
    """Geometric (spatial) median of the rows of ``X`` via Weiszfeld's algorithm.

    Parameters
    ----------
    X : np.ndarray
        Shape ``(n, d)``. A 1-D input is treated as ``(n, 1)`` and the result matches the ordinary median.
    max_iter, tol : int, float
        Iteration controls (converges linearly; a few dozen iterations usually suffice).
    eps : float
        Coincident-point distance floor.

    Returns
    -------
    np.ndarray
        Shape ``(d,)`` robust center minimizing the sum of Euclidean distances.
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.shape[0] == 0:
        return np.full(X.shape[1] if X.ndim == 2 else 1, np.nan, dtype=np.float64)
    return _geometric_median_weiszfeld(X, int(max_iter), float(tol), float(eps))
