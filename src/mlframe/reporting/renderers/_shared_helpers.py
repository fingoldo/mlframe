"""Renderer-agnostic helpers shared by the matplotlib and plotly renderers.

Both renderers thin heatmap tick labels the same way and need the same
finite value-range over a matrix for cell-text color resolution; the single
implementation lives here so the two backends can't drift.
"""
from __future__ import annotations

import numpy as np

# A density heatmap bins into ~80x80 cells, so one tick per cell-label overlaps into unreadable soup. Above this
# many labels, show at most this many evenly-spaced ticks (the rest of the grid is still drawn).
_HEATMAP_MAX_TICKS = 8


def _thin_tick_positions(n: int, max_ticks: int = _HEATMAP_MAX_TICKS):
    """Evenly-spaced tick indices for an axis of ``n`` labels, always including the first and last."""
    if n <= max_ticks:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (max_ticks - 1))) for i in range(max_ticks)})


def _finite_range(mat):
    """``(vmin, vmax)`` over finite entries, or ``None`` when the matrix is empty / all non-finite.

    Heatmap cell-text color resolution needs a real value range; ``np.nanmin`` raises on an empty array and
    returns NaN on an all-NaN matrix, so callers gate the per-cell text loop on a non-None result.
    """
    a = np.asarray(mat, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    return float(finite.min()), float(finite.max())
