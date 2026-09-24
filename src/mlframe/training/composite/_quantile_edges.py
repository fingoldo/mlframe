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


# A binned conditional median is a step function whose steps must shrink as the data grows, or its bias never does: with a
# fixed 10-20 bins the reconstruction error on a smooth relation stayed flat from n=2k to 20k. The histogram-optimal bin count
# grows like n^(1/3) (bias ~ 1/bins against variance ~ bins/n); n^0.3 keeps the fitted table within 2x per 10x rows, so params
# stay bounded in the training size. The count is ``requested`` up to ``requested * _ROWS_PER_BIN`` rows and grows from there.
_ROWS_PER_BIN = 100
_GROWTH_EXPONENT = 0.3
_MAX_BINS = 200


def bins_for_rows(requested: int, n_rows: int, rows_per_bin: int = _ROWS_PER_BIN) -> int:
    """The bin count for ``n_rows`` fit rows: ``requested`` at small n, then growing like ``n^0.3`` up to ``_MAX_BINS``.

    ``rows_per_bin`` is where growth starts; a per-bin statistic noisier than a median (an IQR) needs more rows before it.
    """
    requested = int(requested)
    n_ref = max(1, requested * int(rows_per_bin))
    if n_rows <= n_ref:
        return requested
    return max(requested, min(_MAX_BINS, int(requested * (int(n_rows) / n_ref) ** _GROWTH_EXPONENT)))
