"""Scan candidate features for correlation with a fold/split/time assignment — a defensive leak check.

A feature that strongly correlates with which fold/split/time-bucket a row landed in is a classic silent
leak: an id-like or date-derived column that accidentally encodes the split structure itself (e.g. a
"days since account refresh" column built from the same clock used to draw the train/test cutoff). A model
that picks this up learns to exploit the split boundary rather than the underlying signal — it looks great in
CV/on a public leaderboard and then degrades on genuinely new data. ``scan_temporal_leak`` is a cheap,
always-run-before-shipping diagnostic: rank-correlate every candidate column against the split/fold/time
label and flag anything above a threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rank_columns(x: np.ndarray) -> np.ndarray:
    """Vectorised per-column rank (0-indexed, ties broken by original order) via one argsort + a scatter.

    Applied along axis 0 so every column of an ``(n, k)`` matrix is ranked independently in one pass. Ties
    are NOT averaged (unlike ``scipy.stats.rankdata``) — acceptable for a screening diagnostic on
    continuous/near-continuous id/date-like columns, and avoids an O(k) Python loop over per-column
    ``scipy.stats.rankdata`` calls.

    Uses a SINGLE ``argsort`` + ``put_along_axis`` scatter rather than the textbook "argsort of argsort"
    rank trick (two full ``O(n log n)`` sorts). Profiled (``_benchmarks/bench_leak_scan.py``, n=100k,
    cols=200): argsort was 59s of a 68s total scan — the second argsort call is pure waste, since the
    inverse permutation of one argsort already IS the rank vector. ~1.9x faster, bit-identical ranks
    (same argsort tie-breaking on both paths since only one argsort call now exists).
    """
    n = x.shape[0]
    order = np.argsort(x, axis=0)
    ranks = np.empty_like(order)
    row_ids = np.broadcast_to(np.arange(n)[:, None], order.shape)
    np.put_along_axis(ranks, order, row_ids, axis=0)
    return ranks.astype(np.float64)


def scan_temporal_leak(
    X: pd.DataFrame,
    split_labels: np.ndarray,
    columns: list[str] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Rank-correlate every candidate column against a split/fold/time label; flag likely leaks.

    Parameters
    ----------
    X
        Feature frame to scan.
    split_labels
        ``(n_samples,)`` array identifying which fold/split/time-bucket each row belongs to (e.g. fold index,
        a binary train=0/test=1 indicator, or an ordinal time-bucket id). Any dtype ``np.argsort`` accepts
        (numeric, or datetime64 — cast to numeric first for non-numeric labels).
    columns
        Subset of ``X`` columns to scan. Defaults to every numeric column in ``X`` (non-numeric columns are
        silently skipped — cast/encode them first if they should be scanned).
    threshold
        Absolute Spearman-style rank correlation above which a column is flagged (``flagged=True``).

    Returns
    -------
    pd.DataFrame
        One row per scanned column: ``{"column", "correlation", "flagged"}``, sorted by
        ``|correlation|`` descending (most leak-suspicious first).
    """
    if columns is None:
        columns = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if not columns:
        return pd.DataFrame(columns=["column", "correlation", "flagged"])

    y = np.asarray(split_labels, dtype=np.float64).ravel()
    n = y.shape[0]
    if n != len(X):
        raise ValueError(f"scan_temporal_leak: split_labels length {n} must match X row count {len(X)}")

    x_mat = X[columns].to_numpy(dtype=np.float64)
    ranks_x = _rank_columns(x_mat)
    ranks_y = np.argsort(np.argsort(y))

    rx_centered = ranks_x - ranks_x.mean(axis=0, keepdims=True)
    ry_centered = ranks_y - ranks_y.mean()
    numerator = rx_centered.T @ ry_centered
    denom = np.sqrt((rx_centered**2).sum(axis=0)) * np.sqrt((ry_centered**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0, numerator / denom, 0.0)

    result = pd.DataFrame({"column": columns, "correlation": corr})
    result["flagged"] = result["correlation"].abs() >= threshold
    return result.reindex(result["correlation"].abs().sort_values(ascending=False).index).reset_index(drop=True)


__all__ = ["scan_temporal_leak"]
