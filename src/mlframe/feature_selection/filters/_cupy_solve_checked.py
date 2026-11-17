"""``cupy.linalg.solve`` that raises ``numpy.linalg.LinAlgError`` on a singular system."""

from __future__ import annotations

from typing import Any

import numpy as np


def cupy_solve_checked(cp: Any, a: Any, b: Any) -> Any:
    """``cp.linalg.solve(a, b)``, raising ``np.linalg.LinAlgError`` when the solution is not finite.

    cupy does not raise for a singular matrix (its default is to return NaN), so an ``except np.linalg.LinAlgError`` around a bare ``cp.linalg.solve`` never fires and the NaN
    coefficients flow on. The check costs one device-to-host flag read, which is small next to the solve itself.
    """
    out = cp.linalg.solve(a, b)
    if not bool(cp.isfinite(out).all()):
        raise np.linalg.LinAlgError("singular system: cupy solve returned non-finite values")
    return out
