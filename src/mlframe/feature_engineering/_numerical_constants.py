"""Leaf module for numerical-FE constants shared by ``numerical`` and
``_numerical_numba``. Lives separately so the two siblings don't form an
import cycle (numerical pulls in _numerical_numba at the bottom for
re-export; _numerical_numba needs these constants at @njit-decoration time).
"""

from __future__ import annotations

from typing import Tuple

from scipy import stats

NUMBA_NJIT_PARAMS = dict(fastmath=False, cache=True, nogil=True)

# Default quantile marks for the nunique/modes/quantiles aggregates. Tuple is hashable + immutable; callers that need a
# list/ndarray convert via list(default_quantiles) where the ~10% per-loop speed difference matters.
default_quantiles: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)

distributions = (stats.levy_l,)

LARGE_CONST = 1e3

# Geometric-mean overflow / underflow thresholds. When the running product crosses either, the
# kernel switches to log-mode accumulation to avoid float64 over/underflow.
GEOMEAN_OVERFLOW_HI: float = 1e100
GEOMEAN_OVERFLOW_LO: float = 1e-100
