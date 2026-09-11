"""Dependence that lives in the tail, which a Cholesky factor cannot express and a binned estimator cannot see.

Almost every synthetic correlated dataset in circulation is built the same way: draw standard normals,
multiply by a Cholesky factor, done. That is a Gaussian copula, and a Gaussian copula has **zero tail
dependence** at every correlation below one -- conditional on one variable being extreme, the probability
that the other is extreme too vanishes as the threshold moves out. Real joint extremes do not behave that
way, and neither do the datasets people actually care about.

The omission is convenient for the wrong reason. With ten equal-mass bins, an entire joint tail falls into a
single cell of the joint histogram, so a binned mutual-information estimator is **structurally blind** to a
dependence that lives only there -- not inaccurate, blind. A suite built exclusively on Gaussian copulas can
never show that, and this repository's own results already point at binned MI as the weakest family, so the
bed that would demonstrate its specific failure mode is exactly the one that was missing.

Three families, chosen because they differ in where the dependence concentrates rather than by how much:

* **Gaussian** -- the control. Correlation without tail dependence, so a method that only fails on the t
  copula is failing on tails and not on correlation.
* **Student t** -- symmetric tail dependence in both tails, controlled by the degrees of freedom. At
  ``df=4`` it is strong; as ``df`` grows the copula converges to the Gaussian, which makes it a knob that
  sweeps between the two rather than a separate universe.
* **Clayton** -- lower-tail dependence only. Asymmetric, so a method that handles the symmetric case by
  symmetry alone still has something to fail on.

All three return uniform margins. Marginals are the caller's business: keeping dependence and margin
separate is the entire point of a copula, and mixing them is how a "heavy-tailed" bed ends up testing the
margin when it meant to test the joint.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["COPULA_FAMILIES", "gaussian_copula", "t_copula", "clayton_copula", "sample_copula", "upper_tail_coefficient"]

COPULA_FAMILIES: Tuple[str, ...] = ("gaussian", "t", "clayton")


def _equicorrelation(dim: int, rho: float) -> np.ndarray:
    """Return a ``dim x dim`` equicorrelation matrix, checked for positive definiteness.

    Raises:
        ValueError: If ``rho`` is outside the range that keeps the matrix a valid correlation matrix. For
            equicorrelation the lower bound is ``-1/(dim-1)``, not ``-1``, and a caller who passes -0.9 for
            three columns is asking for something that does not exist rather than something extreme.
    """
    if dim < 2:
        raise ValueError(f"a copula needs at least two columns; got {dim}")
    lower = -1.0 / (dim - 1)
    if not lower < rho < 1.0:
        raise ValueError(f"equicorrelation rho must lie in ({lower:.3f}, 1) for {dim} columns; got {rho}")
    matrix = np.full((dim, dim), float(rho), dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def gaussian_copula(rng: np.random.Generator, n: int, dim: int, rho: float = 0.7) -> np.ndarray:
    """Return ``n x dim`` uniforms with Gaussian dependence and, by construction, no tail dependence."""
    from scipy import stats

    normals = rng.multivariate_normal(np.zeros(dim), _equicorrelation(dim, rho), size=int(n), method="cholesky")
    return np.asarray(stats.norm.cdf(normals), dtype=np.float64)


def t_copula(rng: np.random.Generator, n: int, dim: int, rho: float = 0.7, df: float = 4.0) -> np.ndarray:
    """Return ``n x dim`` uniforms with Student-t dependence: symmetric dependence in both tails.

    Built the standard way -- a Gaussian vector divided by ``sqrt(chi2_df / df)`` -- because the shared
    divisor is exactly what makes extremes co-occur: one small draw of the divisor inflates every coordinate
    at once, which is the mechanism a Gaussian copula lacks.
    """
    from scipy import stats

    if df <= 0:
        raise ValueError(f"degrees of freedom must be positive; got {df}")
    normals = rng.multivariate_normal(np.zeros(dim), _equicorrelation(dim, rho), size=int(n), method="cholesky")
    divisor = np.sqrt(rng.chisquare(df, size=int(n)) / df)[:, None]
    return np.asarray(stats.t.cdf(normals / divisor, df=df), dtype=np.float64)


def clayton_copula(rng: np.random.Generator, n: int, dim: int, theta: float = 2.0) -> np.ndarray:
    """Return ``n x dim`` uniforms with Clayton dependence: lower-tail dependence only.

    Marshall-Olkin construction: draw a gamma frailty, then independent exponentials conditioned on it. The
    frailty is shared, so small draws pull every coordinate towards zero together and the dependence sits in
    the LOWER tail; the upper tail is asymptotically independent.
    """
    if theta <= 0:
        raise ValueError(f"Clayton theta must be positive; got {theta}")
    frailty = rng.gamma(shape=1.0 / theta, scale=1.0, size=int(n))[:, None]
    exponentials = rng.exponential(size=(int(n), int(dim)))
    return np.asarray((1.0 + exponentials / frailty) ** (-1.0 / theta), dtype=np.float64)


def sample_copula(rng: np.random.Generator, n: int, dim: int, family: str = "gaussian", rho: float = 0.7, df: float = 4.0, theta: float = 2.0) -> np.ndarray:
    """Dispatch to one copula family, returning ``n x dim`` uniform margins.

    Raises:
        ValueError: On an unknown family, so a typo cannot silently fall back to the Gaussian control -- the
            one family whose whole role is to have no tail dependence.
    """
    if family == "gaussian":
        return gaussian_copula(rng, n, dim, rho=rho)
    if family == "t":
        return t_copula(rng, n, dim, rho=rho, df=df)
    if family == "clayton":
        return clayton_copula(rng, n, dim, theta=theta)
    raise ValueError(f"unknown copula family {family!r}; expected one of {COPULA_FAMILIES}")


def upper_tail_coefficient(uniforms: np.ndarray, quantile: float = 0.98) -> float:
    """Return the empirical upper-tail dependence: P(V > q | U > q), averaged over column pairs.

    The diagnostic that separates the families without appealing to their definitions. A Gaussian copula
    tends to zero as the quantile moves out; a t copula tends to a positive constant. Estimated at a fixed
    quantile it is noisy by construction -- the estimate uses only the rows beyond it -- so read it as a
    comparison between families on the same data size, never as a point estimate of the limit.
    """
    data = np.asarray(uniforms, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("upper-tail dependence needs a 2-D array with at least two columns")
    beyond = data > float(quantile)
    pairs = []
    for i in range(data.shape[1]):
        for j in range(i + 1, data.shape[1]):
            conditioning = int(beyond[:, i].sum())
            if conditioning:
                pairs.append(float(np.sum(beyond[:, i] & beyond[:, j]) / conditioning))
    return float(np.mean(pairs)) if pairs else float("nan")
