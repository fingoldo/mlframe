"""njit kernels of the row-statistic operator: one row statistic over a subset of (standardised) columns, scored by plug-in MI.

``eval_candidates`` scores a batch of ``(subset, statistic)`` candidates in parallel without storing a single candidate column: for each one it computes the statistic of a ``N_EDGE_SAMPLE``
sample of the even rows to take the bin edges, then one pass over all rows bins the statistic, counts the joint histogram on the even rows (the rows the search selects on) and on the odd rows
(held out), and returns both MIs. The statistic of a row is computed in place from the ``k`` standardised values of the subset (sorting for the median is an insertion sort of at most ``MAX_SUBSET``
values), so the work per candidate is ``n * k`` and nothing needs the full-column argsort a quantile binning would.
"""

from __future__ import annotations

# The kernels live in ``_numba_utils`` (the shared host of every cross-module ``@njit`` helper); re-exported here for existing import paths.
from ._numba_utils import LSE_BETA, MAX_SUBSET, N_EDGE_SAMPLE, STAT_NAMES, eval_candidates, stat_column

__all__ = ["STAT_NAMES", "MAX_SUBSET", "LSE_BETA", "N_EDGE_SAMPLE", "eval_candidates", "stat_column"]
