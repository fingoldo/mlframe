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

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def constant_group_target_scan(
    df: pd.DataFrame,
    y: np.ndarray,
    candidate_cols: Sequence[str],
    min_group_size: int = 20,
    variance_ratio_threshold: float = 0.1,
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

    Returns
    -------
    pd.DataFrame
        One row per candidate column: ``{"column", "n_groups", "min_group_variance_ratio",
        "worst_group_value", "worst_group_size", "flagged"}``, sorted by ``min_group_variance_ratio``
        ascending (most leak-suspicious first). ``NaN``/skipped when a column has no group meeting
        ``min_group_size``.
    """
    y = np.asarray(y, dtype=np.float64)
    overall_var = float(np.var(y, ddof=1)) if len(y) > 1 else 0.0
    if overall_var <= 0.0:
        raise ValueError("constant_group_target_scan: y has zero overall variance -- nothing to compare groups against")

    rows = []
    for col in candidate_cols:
        stats = pd.DataFrame({"group": df[col].to_numpy(), "y": y}).groupby("group")["y"].agg(["var", "size"])
        eligible = stats[stats["size"] >= min_group_size].dropna(subset=["var"])
        if eligible.empty:
            rows.append(
                {
                    "column": col,
                    "n_groups": int(stats.shape[0]),
                    "min_group_variance_ratio": np.nan,
                    "worst_group_value": None,
                    "worst_group_size": 0,
                    "flagged": False,
                }
            )
            continue

        worst = eligible["var"].idxmin()
        min_ratio = float(eligible.loc[worst, "var"] / overall_var)
        rows.append(
            {
                "column": col,
                "n_groups": int(stats.shape[0]),
                "min_group_variance_ratio": min_ratio,
                "worst_group_value": worst,
                "worst_group_size": int(eligible.loc[worst, "size"]),
                "flagged": min_ratio < variance_ratio_threshold,
            }
        )

    result = pd.DataFrame(rows)
    return result.sort_values("min_group_variance_ratio", ascending=True, na_position="last").reset_index(drop=True)


__all__ = ["constant_group_target_scan"]
