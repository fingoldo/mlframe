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


def _compute_extremality_matrix(X: pd.DataFrame, columns: Optional[Sequence[str]]) -> tuple[np.ndarray, list]:
    """Shared per-column rank-extremality computation used by both public functions in this module.

    Returns the ``(n_rows, n_cols)`` extremality matrix (NaN where the source value was NaN) and the
    resolved column list, so callers needing the raw per-column scores (not just their row-mean) don't
    have to recompute the ranking.
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

    return extremality, cols


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
    extremality, _cols = _compute_extremality_matrix(X, columns)
    return pd.Series(np.nanmean(extremality, axis=1), index=X.index, name=column_name)


def _build_column_extremality_summary(top_cols: np.ndarray, top_scores: np.ndarray, cols: list, row_mask: Optional[np.ndarray]) -> pd.DataFrame:
    """Column-level rollup of a row-wise top-k result, restricted to ``row_mask`` (all rows if ``None``).

    Aggregates over rows so a caller with an already-identified batch of interesting rows (e.g. true
    anomaly/incident labels, or any other externally-defined subset) can answer "which columns explain THIS
    batch overall" for a report, instead of eyeballing the per-row breakdown row by row. Note this is
    deliberately a rollup of an externally-chosen row subset, not an unsupervised "find the noisy column"
    detector: because each column's within-column rank is, by construction, a fixed permutation of the same
    ``{1/(n+1), ..., n/(n+1)}`` set regardless of that column's own values, every column's top-k membership
    rate converges to the same chance floor (``k / n_cols``) when aggregated over ALL rows -- restricting to
    a subset whose selection correlates with a column's actual values (e.g. rows a real anomaly label flags)
    is what breaks that symmetry and produces a genuine per-column signal.
    """
    if row_mask is not None:
        top_cols = top_cols[row_mask]
        top_scores = top_scores[row_mask]
    n_selected_rows = top_cols.shape[0]

    flat_cols = pd.Series(top_cols.ravel())
    flat_scores = pd.Series(top_scores.ravel())
    valid = flat_cols.notna()

    grouped = pd.DataFrame({"column": flat_cols[valid].to_numpy(), "score": flat_scores[valid].to_numpy()}).groupby("column")["score"].agg(["count", "mean"])

    summary = pd.DataFrame(index=pd.Index(cols, name="column"))
    summary["count"] = grouped["count"].reindex(summary.index).fillna(0).astype(np.int64)
    summary["frequency"] = summary["count"] / n_selected_rows if n_selected_rows else np.nan
    summary["mean_score"] = grouped["mean"].reindex(summary.index)

    return summary.sort_values("count", ascending=False)


def row_wise_top_k_extreme_columns(
    X: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    k: int = 3,
    return_column_summary: bool = False,
    summary_rows: Optional[Sequence[bool] | np.ndarray] = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Per-row top-``k`` columns by within-column-rank extremality -- "why is this row anomalous".

    Reuses the same per-column rank-extremality computation as :func:`row_wise_extremality_index` (via
    :func:`_compute_extremality_matrix`) so the aggregate anomaly score and this per-row breakdown of which
    specific columns drive it are always consistent with each other.

    Parameters
    ----------
    X
        Feature frame.
    columns
        Column subset to consider per row (default: every numeric column of ``X``).
    k
        Number of top columns to report per row (clipped to the number of available columns).
    return_column_summary
        If ``True``, also return a per-column rollup: how often (count/frequency) and how severely (mean
        score) each column appears in the top-``k`` of the rows selected by ``summary_rows`` -- a
        column-level "which features explain this batch" report (default ``False``, matching the prior
        return contract).
    summary_rows
        Boolean mask (aligned to ``X.index``) selecting which rows to aggregate over when
        ``return_column_summary`` is ``True`` -- typically an externally-known anomalous/flagged-row subset
        (e.g. confirmed incidents), since aggregating over ALL rows converges every column to the same
        chance floor by construction (see :func:`_build_column_extremality_summary`). Defaults to every row
        if omitted. Ignored when ``return_column_summary`` is ``False``.

    Returns
    -------
    pd.DataFrame
        ``(n, 2*k)`` frame indexed like ``X``, columns ``top1_column, top1_score, ..., topk_column,
        topk_score``, sorted by descending extremality. A row with fewer than ``k`` valid (non-NaN) values
        pads the remaining slots with ``None``/``NaN``. If ``return_column_summary`` is ``True``, returns a
        ``(per_row, per_column)`` tuple instead, where ``per_column`` is indexed by column name with
        ``count`` / ``frequency`` / ``mean_score``, sorted by descending ``count``.
    """
    extremality, cols = _compute_extremality_matrix(X, columns)
    n_rows, n_cols = extremality.shape
    k = min(k, n_cols)

    # push NaNs to the bottom of the descending sort without disturbing real (always non-negative) scores.
    sort_key = np.where(np.isnan(extremality), -1.0, extremality)
    # argpartition is O(n_cols) vs argsort's O(n_cols log n_cols) -- for k << n_cols (the typical top-k use
    # case) only the k winners need a full ordering, not all n_cols; a full argsort here showed up as ~18%
    # of this function's wall time in cProfile at n_cols=200, k=3 (measured against the shared column-rank
    # helper's own cost, which both functions pay identically).
    if k < n_cols:
        candidates = np.argpartition(-sort_key, kth=k - 1, axis=1)[:, :k]
    else:
        candidates = np.tile(np.arange(n_cols), (n_rows, 1))
    candidate_scores = np.take_along_axis(sort_key, candidates, axis=1)
    local_order = np.argsort(-candidate_scores, axis=1, kind="quicksort")
    order = np.take_along_axis(candidates, local_order, axis=1)

    top_scores = np.take_along_axis(extremality, order, axis=1)
    cols_arr = np.asarray(cols, dtype=object)
    top_cols = cols_arr[order]
    invalid = np.isnan(top_scores)
    top_cols = top_cols.astype(object)
    top_cols[invalid] = None

    data = {}
    for i in range(k):
        data[f"top{i + 1}_column"] = top_cols[:, i]
        data[f"top{i + 1}_score"] = top_scores[:, i]

    result = pd.DataFrame(data, index=X.index)
    if not return_column_summary:
        return result

    row_mask = None if summary_rows is None else np.asarray(summary_rows, dtype=bool)
    return result, _build_column_extremality_summary(top_cols, top_scores, cols, row_mask)


__all__ = ["row_wise_extremality_index", "row_wise_top_k_extreme_columns"]
