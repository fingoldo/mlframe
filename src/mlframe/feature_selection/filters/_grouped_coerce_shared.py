"""Shared recipe-replay coercion helper for the grouped-aggregate FE family
(_grouped_agg_fe, _grouped_quantile_fe, _ratio_delta_fe): independently duplicated across those
modules under different names, consolidated here so a fix can't silently drift out of sync
across copies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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
