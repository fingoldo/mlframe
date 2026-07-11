"""``row_wise_extremality_index``: per-row average of each feature's within-column quantile-extremality.

Generalizes the "row-wise missing-value count" pattern (how unusual is this row, summarized as a single
number) to fully-observed numeric values: for each column, rank every value within that column (0=lowest,
1=highest), convert to a distance-from-median "extremality" score (0 at the column median, 1 at either
extreme), then average that per-column extremality ACROSS all requested columns for each row. Unlike
:func:`mlframe.feature_engineering.row_wise_summary.row_wise_summary_stats` (which summarizes the RAW feature
values per row -- mean/std/quantile of whatever scale each column happens to be on, so a row with one huge-
scale feature can dominate the row mean), this first puts every column on the SAME [0, 1] extremality scale
via its own within-column rank distribution, so no single feature's raw scale can dominate the row-level
score -- a scale-invariant "how anomalous does this row look overall" meta-feature, directly comparable to
the missing-value-count idea's intent (a compact per-row unusualness signal) but for observed values.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _ordinal_rank(x: np.ndarray) -> np.ndarray:
    """1-based ordinal rank via a double argsort -- no tie-averaging.

    ``scipy.stats.rankdata`` computes the statistically-precise tie-averaged rank, but pays real
    array-api-compat dispatch overhead per call (measured as the dominant cost when called once per column:
    734s cProfile / 53ms-per-call at n=200000, vs 13.5ms-per-call for this direct numpy version -- a ~4x
    difference that compounds badly over hundreds of columns). Continuous feature columns rarely have enough
    exact ties to matter, and this index only needs a monotonic within-column ordering (not exact tie-average
    precision) to produce a symmetric distance-from-median score, so the precision trade is safe here.
    """
    order = np.argsort(x, kind="quicksort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks


def row_wise_extremality_index(X: pd.DataFrame, columns: Optional[Sequence[str]] = None, column_name: str = "row_extremality_index") -> pd.Series:
    """Per-row mean within-column-rank extremality, averaged across ``columns``.

    Parameters
    ----------
    X
        Feature frame.
    columns
        Column subset to summarize per row (default: every numeric column of ``X``).
    column_name
        Output column/Series name.

    Returns
    -------
    pd.Series
        ``(n,)`` float64, one value per row: ``0`` if every one of the row's values sits exactly at its
        column's median, approaching ``1`` as values sit at the extremes of their columns' distributions
        (averaged across columns; NaN values are excluded from both the ranking and the row-level average).
    """
    cols = list(columns) if columns is not None else list(X.select_dtypes(include=[np.number]).columns)
    values = X[cols].to_numpy(dtype=np.float64)
    n_rows, n_cols = values.shape

    extremality = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    for j in range(n_cols):
        col = values[:, j]
        valid = ~np.isnan(col)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        # normalize the ordinal rank to (0,1) then fold around the median (0.5) to get a symmetric
        # 0 (median) -> 1 (extreme) extremality score.
        ranks = _ordinal_rank(col[valid]) / (n_valid + 1)
        extremality[valid, j] = np.abs(ranks - 0.5) * 2.0

    return pd.Series(np.nanmean(extremality, axis=1), index=X.index, name=column_name)


__all__ = ["row_wise_extremality_index"]
