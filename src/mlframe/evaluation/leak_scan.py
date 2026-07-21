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

from itertools import combinations

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


def _spearman_against(x_mat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorised Spearman-style rank correlation of every column of ``x_mat`` against ``y``."""
    ranks_x = _rank_columns(x_mat)
    # y is a plain 1-D vector, so it goes through the SAME single-argsort+scatter _rank_columns path as
    # every column of x_mat (reshaped to a single-column matrix, then flattened back).
    ranks_y = _rank_columns(y[:, None]).ravel()

    rx_centered = ranks_x - ranks_x.mean(axis=0, keepdims=True)
    ry_centered = ranks_y - ranks_y.mean()
    numerator = rx_centered.T @ ry_centered
    denom = np.sqrt((rx_centered**2).sum(axis=0)) * np.sqrt((ry_centered**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, numerator / denom, 0.0)


def _build_derived_matrix(
    X: pd.DataFrame,
    columns: list[str],
    derived_ops: tuple[str, ...],
    max_derived_features: int | None,
) -> tuple[np.ndarray, list[str]]:
    """Build a matrix of pairwise diff/ratio features between numeric columns, capped at ``max_derived_features``.

    Real production leaks are frequently a DIFFERENCE (or ratio) of two innocuous raw columns rather than any
    single raw feature — e.g. the Home-Credit "days since ..." leaks, where neither of the two underlying date
    columns correlates with the split on its own, but their difference reconstructs the split-defining clock.
    Scanning raw columns alone is structurally blind to this leak family, hence this opt-in extension.

    Ratios guard against division-by-zero/overflow by neutralising non-finite results to 0.0 (a non-finite
    ratio carries no rank-correlation signal, so it must not corrupt the argsort-based ranking below).
    """
    pair_names: list[tuple[str, str, str]] = []
    columns_out: list[str] = []
    for a, b in combinations(columns, 2):
        if max_derived_features is not None and len(columns_out) >= max_derived_features:
            break
        if "diff" in derived_ops:
            pair_names.append((a, b, "diff"))
            columns_out.append(f"{a} - {b}")
        if max_derived_features is not None and len(columns_out) >= max_derived_features:
            break
        if "ratio" in derived_ops:
            pair_names.append((a, b, "ratio"))
            columns_out.append(f"{a} / {b}")

    if not columns_out:
        return np.empty((len(X), 0), dtype=np.float64), []

    derived = np.empty((len(X), len(columns_out)), dtype=np.float64)
    for i, (a, b, op) in enumerate(pair_names):
        col_a = X[a].to_numpy(dtype=np.float64)
        col_b = X[b].to_numpy(dtype=np.float64)
        if op == "diff":
            derived[:, i] = col_a - col_b
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = col_a / col_b
            derived[:, i] = np.where(np.isfinite(ratio), ratio, 0.0)
    return derived, columns_out


def scan_temporal_leak(
    X: pd.DataFrame,
    split_labels: np.ndarray,
    columns: list[str] | None = None,
    threshold: float = 0.5,
    scan_derived: bool = False,
    derived_ops: tuple[str, ...] = ("diff", "ratio"),
    max_derived_features: int | None = 2000,
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
    scan_derived
        Opt-in (default ``False``, output is bit-identical to the pre-existing behavior when omitted). When
        ``True``, additionally scans pairwise DIFFERENCE and/or RATIO features built from every combination of
        the scanned columns — catches leaks that only show up in a combination of two clean-looking raw
        columns (e.g. a date-difference leak), which single-column scanning is structurally blind to.
    derived_ops
        Which derived-feature kinds to build when ``scan_derived=True``: any of ``"diff"``, ``"ratio"``.
    max_derived_features
        Caps the number of derived features actually scanned (pairwise combos are ``O(k^2)`` in the column
        count) — ``None`` disables the cap. Combos are taken in deterministic ``itertools.combinations`` order,
        diff before ratio per pair, so the same cap always yields the same subset.

    Returns
    -------
    pd.DataFrame
        One row per scanned column: ``{"column", "correlation", "flagged"}`` (plus ``"derived"`` when
        ``scan_derived=True``), sorted by ``|correlation|`` descending (most leak-suspicious first).
    """
    if columns is None:
        columns = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    if not columns:
        empty_cols = ["column", "correlation", "flagged", "derived"] if scan_derived else ["column", "correlation", "flagged"]
        return pd.DataFrame(columns=empty_cols)

    y = np.asarray(split_labels, dtype=np.float64).ravel()
    n = y.shape[0]
    if n != len(X):
        raise ValueError(f"scan_temporal_leak: split_labels length {n} must match X row count {len(X)}")

    x_mat = X[columns].to_numpy(dtype=np.float64)
    corr = _spearman_against(x_mat, y)

    result = pd.DataFrame({"column": columns, "correlation": corr})
    if scan_derived:
        result["derived"] = False
        derived_mat, derived_names = _build_derived_matrix(X, columns, derived_ops, max_derived_features)
        if derived_names:
            derived_corr = _spearman_against(derived_mat, y)
            derived_result = pd.DataFrame({"column": derived_names, "correlation": derived_corr, "derived": True})
            result = pd.concat([result, derived_result], ignore_index=True)

    result["flagged"] = result["correlation"].abs() >= threshold
    return result.reindex(result["correlation"].abs().sort_values(ascending=False).index).reset_index(drop=True)


__all__ = ["scan_temporal_leak"]
