"""``rank_ecdf_residual`` composite transform.

``T = ecdf_y(y) - ecdf_base(base)``, inverse ``y = quantile_y(T_hat +
ecdf_base(base))``, where ``ecdf_y`` / ``ecdf_base`` are the TRAIN empirical CDFs
and ``quantile_y`` is the train quantile function (inverse ECDF) of ``y``.

Motivation. ``linear_residual`` subtracts a fitted LINE ``alpha*base + beta``; on
a heavy-tailed target, or one that is a MONOTONE-but-nonlinear warp of the base
(``y = sinh(k*base)``, ``y = base**3``, lognormal noise), the line leaves a large
structured residual and its inverse extrapolates catastrophically on the tails.
Mapping both ``y`` and ``base`` into their common empirical-CDF (rank) space
collapses ANY monotone distortion to the identity, so the rank-space residual is
clean and the reconstruction stays bounded in ``[min(y_train), max(y_train)]`` by
construction (the quantile function cannot leave the train support).

ECDF representation + inverse method. ``fit`` stores, for ``y`` and for ``base``,
the sorted UNIQUE train values as knots plus their plotting-position CDF
``u = (rank + 0.5)/n`` collapsed to the last occurrence of each tie (so the knot
map is strictly monotone and exactly invertible). Forward evaluates
``ecdf(x) = interp(x, knots, u)`` and the inverse quantile function is the
SAME piecewise-linear map read the other way, ``quantile(u) = interp(u, u_knots,
knots)``. ``np.interp`` clamps out-of-support inputs to the edge knots, which IS
the out-of-support handling: a predict-time ``base`` beyond the train range maps
to ``u in {u_min, u_max}`` and a recovered ``u`` beyond ``[u_min, u_max]`` maps
to ``y in {y_min, y_max}``. On the training points (which are knots) the two
maps are exact inverses, so ``inverse(forward(y)) == y`` to float precision.

cProfile (see ``_benchmarks/bench_rank_ecdf.py``). ``fit`` is dominated by the
two ``np.unique`` sorts (one for ``y``, one for ``base``); forward / inverse are
``np.interp`` binary searches. An ``@njit`` rewrite of the searchsorted-based
interp was benched against ``np.interp`` and rejected: numpy's C ``interp`` is
already a tight branchless binary search over contiguous float64 knots and numba
reproduces the identical sort/search with added JIT + dispatch cost. The unique
sort dominates fit regardless of backend. No actionable speedup; numpy is the
default and at the vectorised floor.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _ecdf_knots(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted-unique knots + strictly-increasing plotting-position CDF ``u``.

    Ties collapse to the last occurrence so ``knots -> u`` is one-to-one and the
    inverse ``u -> knots`` interpolation is exact at every knot.
    """
    x_f = np.asarray(x, dtype=np.float64).reshape(-1)
    x_f = x_f[np.isfinite(x_f)]
    n = x_f.size
    if n == 0:
        return np.array([0.0]), np.array([0.5])
    order = np.argsort(x_f, kind="quicksort")
    xs = x_f[order]
    # Midrank CDF at each sorted position; collapse to last-occurrence per unique
    # value so a tied plateau maps to a single strictly-increasing knot.
    u_full = (np.arange(n, dtype=np.float64) + 0.5) / n
    # Keep the LAST occurrence of each unique value so the knot's CDF equals
    # P(X <= value) and the ``knots -> u`` map is strictly increasing.
    keep = np.ones(xs.size, dtype=bool)
    keep[:-1] = xs[1:] != xs[:-1]
    knots = xs[keep]
    u = u_full[keep]
    if knots.size == 1:
        # Constant column: a degenerate 2-knot ramp keeps ``interp`` invertible
        # without dividing by a zero span.
        knots = np.array([knots[0], knots[0] + 1.0])
        u = np.array([0.5, 0.5 + 1e-9])
    return knots, u


def _rank_ecdf_residual_fit(
    y: np.ndarray, base: np.ndarray,
    sample_weight: np.ndarray | None = None,  # noqa: ARG001 - API symmetry; ECDF is unweighted
) -> dict[str, Any]:
    """Store train ECDF knots for ``y`` (invertible) and ``base`` (forward-only)."""
    y_knots, y_cdf = _ecdf_knots(y)
    base_knots, base_cdf = _ecdf_knots(base)
    return {
        "y_knots": y_knots,
        "y_cdf": y_cdf,
        "base_knots": base_knots,
        "base_cdf": base_cdf,
    }


def _rank_ecdf_residual_forward(
    y: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    y_knots = np.asarray(params["y_knots"], dtype=np.float64)
    y_cdf = np.asarray(params["y_cdf"], dtype=np.float64)
    base_knots = np.asarray(params["base_knots"], dtype=np.float64)
    base_cdf = np.asarray(params["base_cdf"], dtype=np.float64)
    u_y = np.interp(np.asarray(y, dtype=np.float64), y_knots, y_cdf)
    u_base = np.interp(np.asarray(base, dtype=np.float64), base_knots, base_cdf)
    return u_y - u_base


def _rank_ecdf_residual_inverse(
    t_hat: np.ndarray, base: np.ndarray, params: dict[str, Any],
) -> np.ndarray:
    y_knots = np.asarray(params["y_knots"], dtype=np.float64)
    y_cdf = np.asarray(params["y_cdf"], dtype=np.float64)
    base_knots = np.asarray(params["base_knots"], dtype=np.float64)
    base_cdf = np.asarray(params["base_cdf"], dtype=np.float64)
    u_base = np.interp(np.asarray(base, dtype=np.float64), base_knots, base_cdf)
    # Recover the y-rank, then read the train quantile function; ``interp``
    # clamps a recovered rank outside [u_min, u_max] to the train y-support.
    u_y = np.asarray(t_hat, dtype=np.float64) + u_base
    return np.interp(u_y, y_cdf, y_knots)


def _rank_ecdf_residual_domain(
    y: np.ndarray | None, base: np.ndarray,
) -> np.ndarray:
    base_ok = np.isfinite(np.asarray(base, dtype=np.float64))
    if y is None:
        return base_ok
    return np.asarray(base_ok & np.isfinite(np.asarray(y, dtype=np.float64)))
