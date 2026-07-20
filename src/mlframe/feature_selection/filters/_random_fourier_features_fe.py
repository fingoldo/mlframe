"""Random Fourier Features (random kitchen sinks) multi-column kernel-approximation block
(mrmr_audit_2026-07-20 fe_expansion.md "Random Fourier Features (random kitchen sinks) multi-column
kernel-approximation block").

Rahimi & Recht (2007, "Random Features for Large-Scale Kernel Machines"): draw a random Gaussian
projection matrix ``W`` (p x m) and phases ``b ~ Uniform(0, 2*pi)``, emit::

    phi(x) = sqrt(2/m) * cos(X @ W / bandwidth + b)

as ``m`` new columns; the inner product ``phi(x_i).phi(x_j)`` approximates an RBF kernel
``k(x_i, x_j) = exp(-||x_i-x_j||^2 / (2*bandwidth^2))`` in expectation.

Why this catches a shape the catalog misses: every existing basis (Hermite/Legendre/Chebyshev/
Laguerre/wavelet/hinge/Fourier/spline) is a PER-COLUMN expansion, and every cross-basis family
(pair/triplet/quadruplet/adaptive-arity) is a PRODUCT of per-leg bases -- none of them build a
feature that is jointly a smooth function of MANY (5+) raw columns simultaneously without
combinatorial blow-up. Concrete scenario: ``y = exp(-||x||^2 / 2)`` on p=10 jointly-informative
numeric columns (a radial/Gaussian-bump target in 10-D) -- no pairwise product term or even a
quadruplet arity-4 cell captures a genuinely 10-way radial structure, while a handful of random
Fourier features linearly recovers it because the RBF kernel IS the radial-Gaussian target class.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["random_fourier_features"]


def _median_pairwise_distance(X: np.ndarray, *, max_sample: int = 500, random_state: int = 0) -> float:
    """Median pairwise Euclidean distance on a (deterministic) subsample of ``X`` -- the standard
    RBF bandwidth heuristic (Gretton 2005), shared with the module's own HSIC sibling. Returns 1.0
    on a degenerate (n<2 or all-identical-rows) subsample so the caller never divides by zero."""
    n = X.shape[0]
    if n < 2:
        return 1.0
    if n > max_sample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=max_sample, replace=False)
        X = X[idx]
        n = max_sample
    sq = np.sum(X * X, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(d2, 0.0, out=d2)
    iu = np.triu_indices(n, k=1)
    dist = np.sqrt(d2[iu])
    med = float(np.median(dist)) if dist.size else 0.0
    return med if med > 1e-12 else 1.0


def random_fourier_features(
    X: np.ndarray,
    *,
    m: int = 64,
    bandwidth: Optional[float] = None,
    random_state: int = 0,
) -> np.ndarray:
    """Random Fourier Feature expansion of a ``(n, p)`` block into ``(n, m)`` new columns
    approximating an RBF kernel over the FULL joint column set.

    Parameters
    ----------
    X : (n, p) array
        The candidate raw/engineered columns to jointly expand (p can be large -- this is exactly
        the family's point: a joint smooth function of many columns without combinatorial blow-up).
    m : int
        Number of random features to emit. Larger ``m`` approximates the RBF kernel more tightly
        (variance of the approximation shrinks as ``O(1/m)``) at the cost of ``m`` new columns.
    bandwidth : float, optional
        RBF bandwidth. ``None`` (default) uses the median-pairwise-distance heuristic on a
        deterministic subsample of ``X`` (the standard Gretton 2005 choice, shared with the
        module's HSIC sibling).
    random_state : int
        Seed for the projection matrix ``W``, the phase offsets ``b``, and the bandwidth
        subsample -- deterministic and replay-safe (the SAME seed reproduces the SAME features,
        which is load-bearing for a fit/transform contract: ``W``/``b`` must be frozen at fit time
        and reused verbatim at transform time, not re-drawn).

    Returns
    -------
    (n, m) float64 array. Degenerate input (n<1, p<1, or m<1) returns an ``(n, 0)`` array rather
    than raising -- callers treat zero emitted columns as "nothing to add", the same convention
    the audit's own sketch documents for a top_k=0-style empty family output.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    n, p = X.shape
    if n < 1 or p < 1 or m < 1:
        return np.empty((n, 0), dtype=np.float64)
    if not np.isfinite(X).all():
        return np.empty((n, 0), dtype=np.float64)

    bw = float(bandwidth) if bandwidth is not None else _median_pairwise_distance(X, random_state=random_state)
    bw = bw if bw > 1e-12 else 1.0

    rng = np.random.default_rng(random_state)
    W = rng.standard_normal((p, m))
    b = rng.uniform(0.0, 2.0 * np.pi, m)
    Z = (X @ W) / bw + b
    return np.asarray(np.sqrt(2.0 / m) * np.cos(Z))
