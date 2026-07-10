"""Bulk statement-to-statement diff features for panel/longitudinal entity data.

Computing ``value_t - value_{t-1}`` per entity for a handful of hand-picked columns is a routine step -- but
a 3rd-place AmEx-default-prediction team found systematically diffing EVERY numeric column per customer (2604
diff features, the single largest and most impactful feature block, called out as "the magic") beat manual
cherry-picking. This is a thin, bulk-multi-column wrapper around the existing boundary-safe
``per_group_shift`` (never leaks a diff across entities), generalizing the pattern to an arbitrary column list
in one call rather than requiring a hand-written diff per column.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from mlframe.feature_engineering.grouped import per_group_shift


def entity_diff_features(
    df: pd.DataFrame,
    entity_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    n: int = 1,
    suffix: str = "_diff",
) -> pd.DataFrame:
    """Compute ``value_t - value_{t-n}`` per entity for every column in ``feature_cols``, in one call.

    Parameters
    ----------
    df
        Panel/longitudinal frame, already sorted chronologically WITHIN each entity (row order defines the
        "statement sequence"; this function does not re-sort).
    entity_col
        Grouping key.
    feature_cols
        Numeric columns to diff; defaults to every numeric column other than ``entity_col``.
    n
        Lag distance for the diff (``1`` = statement-to-statement, ``2`` for a coarser diff, etc.).
    suffix
        Appended to each source column name for its diff column.

    Returns
    -------
    pd.DataFrame
        ``df`` with one new ``{col}{suffix}`` column per diffed column (``NaN`` for an entity's first ``n``
        rows, per ``per_group_shift``'s boundary-safe contract -- never bleeds across entities).
    """
    if feature_cols is None:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != entity_col]
    feature_cols = list(feature_cols)

    out = df.copy()
    group_ids = df[entity_col].to_numpy()
    for col in feature_cols:
        values = df[col].to_numpy(dtype=np.float64)
        lagged = per_group_shift(values, group_ids, n=n)
        out[f"{col}{suffix}"] = values - lagged

    return out


__all__ = ["entity_diff_features"]
