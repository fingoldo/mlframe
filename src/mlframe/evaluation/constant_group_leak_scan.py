"""Scan for a target that's suspiciously near-constant within groups of a low-cardinality/date column.

A real, repeatedly-reported data-generation quirk: grouping the target by some innocuous-looking column
(application date, batch id, a categorical field) reveals the target is nearly CONSTANT within each group --
not because that column is genuinely predictive, but because of how the data was generated/labeled (a
Smart-Recruits-2016 team found "the target variable when grouped by application date gave a constant
percentage", a de facto leak that dominated everything else once found). This is a lightweight EDA-style
companion to adversarial validation: for each candidate grouping column, compare each group's target
mean/variance against the overall distribution and flag groups that are suspiciously more deterministic than
chance.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd


def _column_stats(df: pd.DataFrame, group_cols: Sequence[str], y: np.ndarray, min_group_size: int) -> tuple[pd.DataFrame, int]:
    """Shared groupby-var/size computation for a single column or a tuple of columns (compound key)."""
    if len(group_cols) == 1:
        group_key = df[group_cols[0]].to_numpy()
    else:
        # a tuple-of-values key so distinct compound combinations don't collide (pandas groupby handles a
        # list of Series natively, but building the key once keeps the single- and multi-column paths identical).
        group_key = pd.MultiIndex.from_frame(df[list(group_cols)])
    stats = pd.DataFrame({"group": group_key, "y": y}).groupby("group", observed=True)["y"].agg(["var", "size"])
    eligible = stats[stats["size"] >= min_group_size].dropna(subset=["var"])
    return eligible, int(stats.shape[0])


def _scan_one(name: object, group_cols: Sequence[str], df: pd.DataFrame, y: np.ndarray, min_group_size: int, overall_var: float, variance_ratio_threshold: float) -> dict:
    """Score one candidate column (or column combo) for the constant-target-per-group leak pattern."""
    eligible, n_groups = _column_stats(df, group_cols, y, min_group_size)
    if eligible.empty:
        return {
            "column": name,
            "n_groups": n_groups,
            "min_group_variance_ratio": np.nan,
            "worst_group_value": None,
            "worst_group_size": 0,
            "flagged": False,
        }
    # positional argmin + iloc (not idxmin + .loc[tuple, col]) -- a tuple worst-group label from a MultiIndex
    # (compound-key combo mode) is otherwise mis-parsed by .loc as a (row, col) locator pair.
    worst_pos = int(np.argmin(eligible["var"].to_numpy()))
    worst = eligible.index[worst_pos]
    min_ratio = float(eligible["var"].iloc[worst_pos] / overall_var)
    return {
        "column": name,
        "n_groups": n_groups,
        "min_group_variance_ratio": min_ratio,
        "worst_group_value": worst,
        "worst_group_size": int(eligible["size"].iloc[worst_pos]),
        "flagged": min_ratio < variance_ratio_threshold,
    }


def constant_group_target_scan(
    df: pd.DataFrame,
    y: np.ndarray,
    candidate_cols: Sequence[str],
    min_group_size: int = 20,
    variance_ratio_threshold: float = 0.1,
    combo_max_size: int = 1,
    combo_max_cols: int = 8,
) -> pd.DataFrame:
    """Flag candidate grouping columns whose groups have suspiciously low target variance vs the overall pool.

    Parameters
    ----------
    df
        Frame containing the candidate grouping columns.
    y
        ``(n_samples,)`` target array (binary or continuous), aligned to ``df``.
    candidate_cols
        Low-cardinality/date-like columns to test as potential leak keys.
    min_group_size
        Groups smaller than this are excluded from a column's "worst group" summary (too few rows for a
        variance estimate to mean anything, would otherwise dominate the flag by chance).
    variance_ratio_threshold
        A column is flagged when its LOWEST within-group variance (among groups meeting ``min_group_size``),
        divided by the overall target variance, falls below this ratio -- i.e. at least one group is far more
        deterministic than the target is overall.
    combo_max_size
        Opt-in multi-column mode. ``1`` (default) reproduces the original single-column-only scan exactly.
        ``2`` additionally scans every 2-way combination of ``candidate_cols`` (up to ``combo_max_cols`` of
        them) as a COMPOUND grouping key -- some leaks only produce a constant target percentage under a
        joint key (e.g. branch x weekday) even though no individual column shows the pattern alone. ``3``
        additionally scans 3-way combinations. Combination cost is O(k^combo_max_size) in the number of
        columns considered, hence the ``combo_max_cols`` cap.
    combo_max_cols
        Caps how many of the leading ``candidate_cols`` are considered for combination generation (single-
        column rows are still emitted for every column in ``candidate_cols`` regardless of this cap). Bounds
        the combinatorial blow-up: e.g. ``combo_max_cols=8, combo_max_size=2`` is at most C(8,2)=28 extra
        groupbys, not C(len(candidate_cols), 2).

    Returns
    -------
    pd.DataFrame
        One row per candidate column (and, when ``combo_max_size > 1``, per scanned column combination):
        ``{"column", "n_groups", "min_group_variance_ratio", "worst_group_value", "worst_group_size",
        "flagged"}``, sorted by ``min_group_variance_ratio`` ascending (most leak-suspicious first).
        ``column`` holds a single column name for single-column rows and a tuple of column names for combo
        rows. ``NaN``/skipped when a column (or combo) has no group meeting ``min_group_size``.
    """
    if combo_max_size < 1:
        raise ValueError(f"constant_group_target_scan: combo_max_size must be >= 1, got {combo_max_size}")

    y = np.asarray(y, dtype=np.float64)
    overall_var = float(np.var(y, ddof=1)) if len(y) > 1 else 0.0
    if overall_var <= 0.0:
        raise ValueError("constant_group_target_scan: y has zero overall variance -- nothing to compare groups against")

    rows = [_scan_one(col, (col,), df, y, min_group_size, overall_var, variance_ratio_threshold) for col in candidate_cols]

    if combo_max_size > 1:
        combo_cols = list(candidate_cols)[:combo_max_cols]
        for depth in range(2, combo_max_size + 1):
            rows.extend(_scan_one(combo, combo, df, y, min_group_size, overall_var, variance_ratio_threshold) for combo in combinations(combo_cols, depth))

    result = pd.DataFrame(rows)
    return result.sort_values("min_group_variance_ratio", ascending=True, na_position="last").reset_index(drop=True)


__all__ = ["constant_group_target_scan"]
