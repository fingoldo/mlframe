"""Absolute-occurrence-count rare-category collapsing and feature dropping.

Source: 3rd_mercedes-benz-greener-manufacturing.md -- categories occurring <=10 times collapsed to a single
"other" bucket; features occurring <20 times across the whole dataset dropped entirely. ABSOLUTE-count
thresholds, not fraction-of-data thresholds, because sample size (4209 rows) was tiny -- a 0.5%-frequency
threshold means something very different at 4,000 rows (20 occurrences) than at 4,000,000 (20,000
occurrences), and for small-N datasets specifically, an absolute floor is the more meaningful knob.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd


def collapse_rare_categories(df: pd.DataFrame, columns: Sequence[str], min_count: int, other_label: str = "__other__") -> pd.DataFrame:
    """Collapse categorical values occurring fewer than ``min_count`` times (per column) into ``other_label``.

    Parameters
    ----------
    df
        Source frame.
    columns
        Categorical columns to screen.
    min_count
        Values with a total occurrence count strictly below this are collapsed.
    other_label
        Replacement value for collapsed (rare) categories.

    Returns
    -------
    pd.DataFrame
        ``df`` (shallow copy) with rare values in ``columns`` replaced by ``other_label``.
    """
    out = df.copy(deep=False)
    for col in columns:
        counts = df[col].value_counts()
        rare_values = counts[counts < min_count].index
        if len(rare_values) > 0:
            out[col] = df[col].where(~df[col].isin(rare_values), other_label)
    return out


def drop_rare_features(df: pd.DataFrame, columns: Optional[Sequence[str]] = None, min_total_count: int = 20, rare_value: object = 0) -> List[str]:
    """Return column names whose "informative" occurrence count falls below ``min_total_count`` -- candidates
    to drop as too-sparse-to-learn-from.

    Matches the source's actual context: sparse one-hot/binary-indicator columns (e.g. expanded from
    high-cardinality categoricals), where "occurring" means the minority/non-default value's count, not raw
    non-null count (a fully-populated binary column with 4189 zeros and 20 ones is exactly the "occurring <20
    times" case the source describes, even though every row is non-null).

    Parameters
    ----------
    df
        Source frame.
    columns
        Columns to screen; defaults to every column of ``df``.
    min_total_count
        Minimum informative-value occurrence count to survive.
    rare_value
        The "default"/uninformative value (e.g. ``0`` for indicator columns, ``False`` for boolean); a
        column's informative count is ``(df[col] != rare_value).sum()`` among non-null rows.

    Returns
    -------
    list of str
        Column names to consider dropping.
    """
    cols = list(columns) if columns is not None else list(df.columns)
    informative_mask = df[cols].ne(rare_value) & df[cols].notna()
    counts = informative_mask.sum(axis=0)
    return [c for c in cols if counts[c] < min_total_count]


__all__ = ["collapse_rare_categories", "drop_rare_features"]
