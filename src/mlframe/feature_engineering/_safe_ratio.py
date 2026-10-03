"""Scale-free guarded division for feature builders.

An additive ``den + 1e-12`` pad is harmless at unit scale but dominates the denominator once it is itself ~1e-12 (squared tiny amplitudes, tiny variances),
silently shrinking the ratio. ``safe_div`` divides exactly and returns 0.0 only where the denominator is exactly zero.
"""

from __future__ import annotations

import numpy as np


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Elementwise ``num / den`` with 0.0 where ``den == 0``; NaN in either operand propagates."""
    num_arr = np.asarray(num, dtype=np.float64)
    den_arr = np.asarray(den, dtype=np.float64)
    out = np.zeros(np.broadcast(num_arr, den_arr).shape, dtype=np.float64)
    np.divide(num_arr, den_arr, out=out, where=den_arr != 0.0)
    return out
