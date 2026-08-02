"""Shared recipe-replay coercion helper for the grouped-aggregate FE family
(_grouped_agg_fe, _grouped_quantile_fe, _ratio_delta_fe): independently duplicated across those
modules under different names, consolidated here so a fix can't silently drift out of sync
across copies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def auto_detect_num_cols_plain(X: pd.DataFrame, group_cols, max_cols: int = 8) -> list:
    """Pick up to ``max_cols`` numeric candidate columns excluding ``group_cols``: all float columns
    qualify, integer columns only if high-cardinality (>500 uniques, i.e. not really categorical). No
    ``grp``-prefix exclusion -- shared by the group_distance_fe / composite_group_agg_fe pair."""
    group_set = set(group_cols)
    out: list = []
    for c in X.columns:
        if c in group_set:
            continue
        col = X[c]
        if not pd.api.types.is_numeric_dtype(col):
            continue
        if pd.api.types.is_float_dtype(col):
            out.append(str(c))
            continue
        if int(col.nunique(dropna=True)) > 500:
            out.append(str(c))
    return out[:max_cols]


def coerce_X_for_grouped(X, group_col: str, num_col: str, recipe_name: str) -> pd.DataFrame:
    """Extract only ``group_col``/``num_col`` into a narrow pandas frame for recipe replay, accepting
    pandas/polars/structured-ndarray input without a full-frame copy."""
    if isinstance(X, pd.DataFrame):
        return X
    try:
        import polars as _pl

        if isinstance(X, _pl.DataFrame):
            return pd.DataFrame(
                {
                    group_col: X[group_col].to_numpy(),
                    num_col: X[num_col].to_numpy(),
                }
            )
    except ImportError:
        pass
    if isinstance(X, np.ndarray) and X.dtype.names is not None:
        return pd.DataFrame({group_col: X[group_col], num_col: X[num_col]})
    raise TypeError(f"recipe '{recipe_name}': cannot extract {group_col!r}/{num_col!r} from X of type {type(X).__name__}")
