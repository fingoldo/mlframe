"""Numba @njit kernels for ``mlframe.training.neural`` helpers.

Sibling carve-out keeping the parent ``base.py`` focused on the
estimator API. Numba is an optional dep here: import failure flips
``_NUMBA_AVAILABLE`` to ``False`` and callers fall back to numpy.

What lives here:
  * ``finite_min_max_std`` -- single-pass (min, max, mean, std) over a
    1-D float array via Welford's online variance accumulator. Two
    explicit upsides over the ``y.min(); y.max(); y.std()`` triple
    used in ``_fit_inner_network``'s auto-derive path:
      1. ONE pass over the array instead of three (each numpy call
         iterates the buffer independently). For ``output_activation_*``
         auto-derive on a 4M-row y this saves ~3x memory bandwidth.
      2. Welford's accumulator is numerically stable for high-range
         targets (e.g. y on a 1e7 scale); the naive E[X^2] - E[X]^2
         formula loses precision on those inputs.
    Also returns the finite-element count so the caller can short-
    circuit the empty / 1-finite case without a separate ``isfinite``
    pass; non-finite values (NaN / +-inf) are skipped in the loop.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


def _finite_min_max_std_python(y: np.ndarray) -> tuple[int, float, float, float, float]:
    """Pure-Python fallback for hosts without numba. Mirrors the kernel's
    contract: returns ``(n_finite, ymin, ymax, ymean, ystd)`` where the
    stats are zero when ``n_finite == 0`` and ``ystd`` is zero when
    ``n_finite == 1`` (Welford's variance is undefined for a single sample).
    """
    n_finite = 0
    ymin = math.inf
    ymax = -math.inf
    mean = 0.0
    M2 = 0.0
    for i in range(y.shape[0]):
        v = float(y[i])
        if not math.isfinite(v):
            continue
        n_finite += 1
        if v < ymin:
            ymin = v
        if v > ymax:
            ymax = v
        # Welford online update.
        delta = v - mean
        mean += delta / n_finite
        delta2 = v - mean
        M2 += delta * delta2
    if n_finite == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    if n_finite == 1:
        return 1, ymin, ymax, mean, 0.0
    var = M2 / n_finite  # population variance (matches numpy's default ddof=0)
    std = math.sqrt(var) if var > 0.0 else 0.0
    return n_finite, ymin, ymax, mean, std


if _NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=False)
    def _finite_min_max_std_njit(y: np.ndarray) -> tuple:
        n_finite = 0
        ymin = np.inf
        ymax = -np.inf
        mean = 0.0
        M2 = 0.0
        for i in range(y.shape[0]):
            v = y[i]
            # ``not isfinite`` covers NaN, +inf, -inf in one branch.
            if not np.isfinite(v):
                continue
            n_finite += 1
            if v < ymin:
                ymin = v
            if v > ymax:
                ymax = v
            delta = v - mean
            mean += delta / n_finite
            delta2 = v - mean
            M2 += delta * delta2
        if n_finite == 0:
            return 0, 0.0, 0.0, 0.0, 0.0
        if n_finite == 1:
            return 1, ymin, ymax, mean, 0.0
        var = M2 / n_finite
        std = np.sqrt(var) if var > 0.0 else 0.0
        return n_finite, ymin, ymax, mean, std


def finite_min_max_std(y: np.ndarray) -> tuple[int, float, float, float, float]:
    """Return ``(n_finite, min, max, mean, std)`` for the finite entries of
    a 1-D float array, computed in a SINGLE pass. ``std`` is the population
    std (ddof=0) to match numpy's default; ``mean`` is included so the
    caller can use it without a separate pass when needed.

    Dispatch: numba @njit kernel when available, plain-Python fallback
    otherwise. The kernel skips non-finite values inline, so the caller
    does not need to materialise an ``isfinite`` mask beforehand.
    """
    arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if _NUMBA_AVAILABLE:
        return _finite_min_max_std_njit(arr)  # type: ignore[name-defined]
    return _finite_min_max_std_python(arr)
