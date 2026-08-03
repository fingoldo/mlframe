"""Numpy-version-agnostic trapezoidal integration: ``np.trapz`` was deprecated in NumPy 2.0 and
``np.trapezoid`` only exists from 2.0 onward, so neither name alone is safe across the project's
supported NumPy range.
"""
from __future__ import annotations

import numpy as np


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral of ``y`` over ``x``, via the manual formula so it works on every supported NumPy version."""
    _y = np.asarray(y, dtype=np.float64)
    _x = np.asarray(x, dtype=np.float64)
    return float(np.sum((_x[1:] - _x[:-1]) * (_y[1:] + _y[:-1]) * 0.5))
