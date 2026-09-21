"""Magnitude-relative guards for scale denominators.

``x / (std + 1e-12)`` protects against a zero divisor with an ABSOLUTE pad, but a scale is only meaningful against the data's own
magnitude. A column whose spread is genuinely ~1e-13 (a normalised residual, a difference of two nearly-equal engineered columns) is then
divided by ~1e-12 rather than by its own spread, so every standardised value is wrong by an order of magnitude while looking finite. A
truly degenerate column, on the other hand, gets divided by 1e-12 and explodes to ~1e12 instead of reading as "no spread to express".

The guards here compare the scale with ``_REL_TOL * max|values|`` — the same magnitude-relative degeneracy test the pair/usability kernels
already use — and return zeros for the degenerate case, which is what "this quantity carries no variation" should look like downstream.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

# A scale below this multiple of the data's own magnitude carries no information: 32 ULP of the largest absolute value.
_REL_TOL = 32.0 * float(np.finfo(np.float64).eps)


def scale_is_usable(scale: Any, reference: Any, xp: Any = np) -> Any:
    """True where ``scale`` is meaningfully positive against ``reference``'s own magnitude (elementwise for arrays)."""
    return scale > _REL_TOL * xp.abs(xp.asarray(reference)).max()


def standardise(values: Any, xp: Any = np, axis: int = 0) -> Any:
    """Z-score ``values`` along ``axis``, returning zeros where the spread is degenerate relative to the column's own magnitude.

    The padded form ``(x - mean) / (std + 1e-12)`` turns a constant column into ~1e12-magnitude noise; zeros say what is actually true.
    """
    arr = xp.asarray(values)
    mean = arr.mean(axis=axis, keepdims=True)
    std = arr.std(axis=axis, keepdims=True)
    mag = xp.abs(arr).max(axis=axis, keepdims=True)
    ok = std > _REL_TOL * mag
    return xp.where(ok, (arr - mean) / xp.where(ok, std, 1.0), 0.0)


def unit_interval(values: Any, xp: Any = np) -> Any:
    """Rescale to ``[0, 1]``; a column with degenerate range returns zeros instead of ``(c - min) / 1e-12``."""
    arr = xp.asarray(values)
    lo = arr.min()
    span = arr.max() - lo
    if not bool(span > _REL_TOL * xp.abs(arr).max()):
        return xp.zeros_like(arr)
    return (arr - lo) / span


def unit_vector(vec: Any, xp: Any = np) -> Any:
    """Direction of ``vec``; a vector whose norm is degenerate relative to its own entries returns zeros, not a ~1e12-magnitude direction."""
    arr = xp.asarray(vec)
    norm = float(xp.linalg.norm(arr))
    if not (norm > _REL_TOL * float(xp.abs(arr).max()) and norm > 0.0):
        return xp.zeros_like(arr)
    return arr / norm


def silverman_bandwidth(values: Any, xp: Any = np) -> Optional[float]:
    """Silverman's rule-of-thumb bandwidth, floored at the data's own magnitude scale rather than an absolute 1e-12.

    Returns ``None`` when the column has no usable spread at all, so a caller can skip a density it cannot estimate.
    """
    arr = xp.asarray(values)
    std = float(arr.std())
    mag = float(xp.abs(arr).max())
    n = int(arr.shape[0]) if arr.ndim else 1
    if not (std > _REL_TOL * mag):
        return None
    return float(1.06 * std * (max(n, 1) ** (-1.0 / 5.0)))
