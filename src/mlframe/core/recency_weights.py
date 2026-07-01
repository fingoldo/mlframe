"""Parametric recency / importance weight vectors over ordered histories.

Generalizes the exponential special case in :mod:`mlframe.core.ewma` to the
three monotone-decaying weight families used by Dyakonov's weighted-scheme
estimators (dunnhumby Shopper Challenge winner): given an ordered history of
length ``d`` where index ``i`` (1-based) counts back from the OLDEST toward the
most recent observation, assign more weight to fresher observations.

    w_i^N = ((d - i + 1) / d) ** delta      # 'poly'   delta in [0, +inf)
    w_i^N = lam ** i                         # 'exp'    lam   in (0, 1]
    w_i^N = 1 / i ** gamma                    # 'power'  gamma in [0, +inf)

then normalized so ``sum_i w_i == 1``. All three collapse to uniform weights at
their identity parameter (delta=0, lam=1, gamma=0), so any downstream estimator
that plugs these in is bit-identical to the unweighted form at the identity
value — the safe default for opt-in recency weighting.

Convention: ``w[0]`` is the weight of the OLDEST observation and ``w[-1]`` the
weight of the MOST RECENT, monotonically non-decreasing. Callers holding a
history already ordered most-recent-first should reverse either the history or
the returned vector.
"""

from __future__ import annotations

import logging

import numpy as np
from numba import njit

logger = logging.getLogger(__name__)

__all__ = ["recency_weights", "SCHEMES"]

SCHEMES = ("poly", "exp", "power")


@njit(fastmath=False, cache=True)
def _recency_weights_njit(d: int, scheme_code: int, param: float, normalize: bool) -> np.ndarray:
    """Core kernel. ``scheme_code``: 0=poly, 1=exp, 2=power. Returns f64 weights, oldest-first."""
    out = np.empty(d, dtype=np.float64)
    if d == 0:
        return out
    # i counts back from the OLDEST (i=d) to the MOST RECENT (i=1); we fill oldest-first so out[0] is i=d.
    total = 0.0
    for pos in range(d):
        i = d - pos  # 1-based recency index: pos=0 -> oldest (i=d), pos=d-1 -> newest (i=1)
        if scheme_code == 0:
            w = ((d - i + 1) / d) ** param
        elif scheme_code == 1:
            w = param**i
        else:
            w = 1.0 / (i**param)
        out[pos] = w
        total += w
    if normalize and total > 0.0:
        for pos in range(d):
            out[pos] /= total
    return out


def recency_weights(d: int, scheme: str = "poly", param: float = 1.0, *, normalize: bool = True) -> np.ndarray:
    """Return a length-``d`` recency weight vector, oldest observation first.

    Parameters
    ----------
    d : int
        History length (number of ordered observations). ``d == 0`` returns an empty array.
    scheme : {'poly', 'exp', 'power'}
        Weight family (see module docstring).
    param : float
        ``delta`` for poly (>= 0), ``lam`` for exp (in (0, 1]), ``gamma`` for power (>= 0).
    normalize : bool
        If True (default) weights sum to 1. If False, raw (unnormalized) weights are returned;
        useful when the caller applies its own normalization (e.g. a weighted mean divides by sum(w) anyway).

    Returns
    -------
    np.ndarray
        float64, shape ``(d,)``, monotonically non-decreasing (oldest -> newest).

    Examples
    --------
    >>> np.allclose(recency_weights(4, 'poly', 0.0), 0.25)   # identity -> uniform
    True
    >>> w = recency_weights(5, 'exp', 0.5); bool(w[-1] > w[0])  # newest heaviest
    True
    """
    if d < 0:
        raise ValueError(f"recency_weights: d must be >= 0, got {d}.")
    if scheme not in SCHEMES:
        raise ValueError(f"recency_weights: scheme must be one of {SCHEMES}, got {scheme!r}.")
    param = float(param)
    if scheme == "poly" and param < 0.0:
        raise ValueError(f"recency_weights: poly delta must be >= 0, got {param}.")
    if scheme == "exp" and not (0.0 < param <= 1.0):
        raise ValueError(f"recency_weights: exp lam must be in (0, 1], got {param}.")
    if scheme == "power" and param < 0.0:
        raise ValueError(f"recency_weights: power gamma must be >= 0, got {param}.")
    if d == 0:
        return np.empty(0, dtype=np.float64)

    scheme_code = SCHEMES.index(scheme)
    return _recency_weights_njit(int(d), scheme_code, param, normalize)
