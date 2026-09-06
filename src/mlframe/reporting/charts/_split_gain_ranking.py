"""Rank features by how sharply a single split on them separates high error from low.

The weak-segment diagnostic used a depth-3 ``DecisionTreeRegressor`` purely to RANK columns, and paid
sklearn's exact splitter for it: every feature sorted at every node, O(p * n log n) even at depth 3. On the
capped input the dispatch already feeds it (100k rows x 200 columns) that fit was 8.70 s of a 9.22 s chart.

Ranking does not need an exact tree. What the tree is asked for is "which columns, split somewhere, separate
the high-error rows" -- which is the depth-1 variance reduction of each column, and that can be read off a
quantile-binned histogram of (count, error sum) per bin in one pass per column. The bin edges are the same
ones ``slice_finder._bin_matrix`` computes for the very next step of the same chart.

The gain is written in the sum/count form rather than as an explicit SSE, because ``sum(y^2)`` is the same
for every candidate split of a column and cancels: maximising ``sum_L^2/n_L + sum_R^2/n_R`` is maximising
the variance reduction, with one fewer pass over the data and no cancellation-prone subtraction.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Bins per column for the ranking histogram. The split threshold itself is never reported -- only the column
# ORDER is -- so the resolution has to be enough to find the right column, not the exact cut.
DEFAULT_RANKING_BINS = 32


# Rows used to place the bin EDGES. np.quantile sorts what it is given, and that sort was the whole cost of
# this ranker; the edges only decide where the candidate cuts sit, while the gain at each cut is still
# computed over every row. Estimating them from a strided sample keeps the same cuts to within a bin.
EDGE_SAMPLE_ROWS = 20_000


def _column_codes(col: np.ndarray, nbins: int) -> Optional[np.ndarray]:
    """Quantile bin codes for one column, or ``None`` when it cannot be split (constant / all-missing)."""
    finite_mask = np.isfinite(col)
    if not finite_mask.any():
        return None
    finite = col[finite_mask]
    edge_src = finite if finite.size <= EDGE_SAMPLE_ROWS else finite[:: max(finite.size // EDGE_SAMPLE_ROWS, 1)]
    edges = np.unique(np.quantile(edge_src, np.linspace(0.0, 1.0, nbins + 1)))
    if edges.size < 3:  # fewer than two bins: nothing to split on
        return None
    # ``searchsorted`` already sends every non-finite value past the last edge, so the missing rows land in
    # a bin of their own above all the finite ones -- which is what lets a column whose MISSINGNESS carries
    # the error be found at all. The median split this replaces dropped those rows and scored the column 0.
    return np.searchsorted(edges[1:-1], col, side="right").astype(np.int64)


def split_gain_per_feature(mat: np.ndarray, err: np.ndarray, *, nbins: int = DEFAULT_RANKING_BINS) -> np.ndarray:
    """Best single-split variance reduction for each column of ``mat`` against ``err``.

    Returns one non-negative score per column; a column that cannot be split scores 0.
    """
    y = np.asarray(err, dtype=np.float64).ravel()
    n_rows, n_cols = int(mat.shape[0]), int(mat.shape[1])
    gains = np.zeros(n_cols, dtype=np.float64)
    if n_rows < 2 or n_cols == 0:
        return gains
    finite_y = np.isfinite(y)
    if not finite_y.all():
        y = np.where(finite_y, y, 0.0)
    total_sum = float(y.sum())
    total_n = float(n_rows)
    baseline = total_sum * total_sum / total_n
    for j in range(n_cols):
        codes = _column_codes(np.asarray(mat[:, j], dtype=np.float64), nbins)
        if codes is None:
            continue
        n_bins_here = int(codes.max()) + 1
        counts = np.bincount(codes, minlength=n_bins_here).astype(np.float64)
        sums = np.bincount(codes, weights=y, minlength=n_bins_here)
        # Every cut between two bins at once: the left side of cut k is bins 0..k inclusive.
        left_n = np.cumsum(counts)[:-1]
        left_s = np.cumsum(sums)[:-1]
        right_n = total_n - left_n
        right_s = total_sum - left_s
        usable = (left_n > 0) & (right_n > 0)
        if not usable.any():
            continue
        gain = np.zeros_like(left_n)
        gain[usable] = left_s[usable] ** 2 / left_n[usable] + right_s[usable] ** 2 / right_n[usable] - baseline
        gains[j] = max(float(gain.max()), 0.0)
    return gains


# A runner-up must reach this fraction of the winner's gain to be reported at all. EVERY column has a
# positive best-split gain on finite data -- pick the luckiest cut in pure noise and it separates something
# -- so "gain > 0" would promote a noise column into the chart as a second axis and invent a dimension the
# data does not have. sklearn's tree gave those columns an importance of exactly zero and drew a 1-D grid.
RUNNER_UP_MIN_SHARE = 0.05


def rank_features_by_split_gain(mat: np.ndarray, err: np.ndarray, n_features: int, *, nbins: int = DEFAULT_RANKING_BINS) -> List[int]:
    """Column indices of the ``n_features`` best single-split separators of ``err``, best first."""
    return top_by_gain(split_gain_per_feature(mat, err, nbins=nbins), n_features)


def top_by_gain(gains: np.ndarray, n_features: int) -> List[int]:
    """The best ``n_features`` columns of a gain vector, dropping runners-up that are noise beside the winner."""
    gains = np.asarray(gains, dtype=np.float64)
    if not np.any(gains > 0):
        return []
    order = np.argsort(gains)[::-1]
    floor = float(gains[order[0]]) * RUNNER_UP_MIN_SHARE
    return [int(j) for j in order[:n_features] if gains[j] >= floor and gains[j] > 0]
