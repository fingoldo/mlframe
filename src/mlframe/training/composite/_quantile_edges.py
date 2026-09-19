"""Quantile bin edges that keep every value of a discrete axis in its own bin.

``np.unique(np.quantile(v, linspace(0, 1, n_bins + 1)))`` is the usual tie-safe edge set, but on a discrete axis with few
values it merges values: a binary 0/1 base yields edges ``[0, 1]``, i.e. ONE bin, so a per-bin median residual degenerates
to the global median and a regime map shows one row. A production run hit this on ``has_explicit_budget`` and warned that
the residual granularity collapsed. When the axis has at most ``n_bins`` distinct values each value gets its own bin, with
boundaries at the midpoints between consecutive values.
"""

from __future__ import annotations

import numpy as np


def quantile_bin_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Return the full, strictly increasing edge array (``len == bins + 1``) for quantile-binning ``values`` (finite, 1-D).

    A constant axis returns its single value (``len == 1``), matching the ``np.unique`` convention callers already handle.
    """
    v = np.asarray(values, dtype=np.float64)
    edges = np.unique(np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size == n_bins + 1:
        return edges
    distinct = np.unique(v)
    if distinct.size <= 1 or distinct.size > n_bins:
        return edges
    return np.concatenate(([distinct[0]], 0.5 * (distinct[:-1] + distinct[1:]), [distinct[-1]]))
