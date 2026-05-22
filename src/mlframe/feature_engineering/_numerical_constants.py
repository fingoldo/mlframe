"""Leaf module for numerical-FE constants shared by ``numerical`` and
``_numerical_numba``. Lives separately so the two siblings don't form an
import cycle (numerical pulls in _numerical_numba at the bottom for
re-export; _numerical_numba needs these constants at @njit-decoration time).
"""

from __future__ import annotations

from scipy import stats

NUMBA_NJIT_PARAMS = dict(fastmath=False, cache=True, nogil=True)

distributions = (stats.levy_l,)

LARGE_CONST = 1e3

# Geometric-mean overflow / underflow thresholds. When the running product crosses either, the
# kernel switches to log-mode accumulation to avoid float64 over/underflow.
GEOMEAN_OVERFLOW_HI: float = 1e100
GEOMEAN_OVERFLOW_LO: float = 1e-100
