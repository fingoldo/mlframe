"""Shared recipe-replay coercion helper for the grouped-aggregate FE family
(_grouped_agg_fe, _grouped_quantile_fe, _ratio_delta_fe): independently duplicated across those
modules under different names, consolidated here so a fix can't silently drift out of sync
across copies.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def broadcast_lookup(g_keys: np.ndarray, lookup: dict, glob: float) -> np.ndarray:
    """Map each row's group key through ``lookup`` (str-keyed), unseen -> ``glob``.

    Group columns are low-cardinality, so the ``str(key)`` + ``dict.get`` is resolved once per UNIQUE key
    (``np.unique(return_inverse=True)``) and broadcast back via the inverse index, not once per row --
    the per-row listcomp form was a Layer-88 hotspot. Bit-identical to the per-row mapping (same
    ``str()``+``get`` per distinct key). Falls back to a per-row mapping on the ``TypeError``/``ValueError``
    ``np.unique`` raises for unorderable mixed-type objects. Shared by the grouped_quantile_fe /
    group_distance_fe pair."""
    g_keys = np.asarray(g_keys)
    try:
        uniq, inverse = np.unique(g_keys, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        uniq_vals = np.array([lookup.get(str(_k), glob) for _k in uniq], dtype=np.float64)
        out = uniq_vals[inverse]
    except (TypeError, ValueError):
        out = np.array([lookup.get(str(_k), glob) for _k in g_keys], dtype=np.float64)
    return np.nan_to_num(out, nan=glob, posinf=glob, neginf=glob)


def auto_detect_num_cols_plain(X: pd.DataFrame, group_cols: "list | tuple | set", max_cols: int = 8) -> list:
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


def auto_detect_num_cols_skip_grp(X: pd.DataFrame, group_cols: "list | tuple | set", max_cols: int = 8) -> list:
    """Pick up to ``max_cols`` numeric candidate columns excluding ``group_cols`` AND already-``grp``-prefixed
    engineered columns (a per-group stat of one of those would build a nested recipe that can't replay from
    raw X at transform time, and the aggregate is constant within group anyway): shared by the
    grouped_quantile_fe / grouped_agg_fe pair."""
    group_set = set(group_cols)
    out: list = []
    for c in X.columns:
        if c in group_set:
            continue
        if str(c).startswith("grp"):
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


def coerce_X_for_grouped(X: "pd.DataFrame | Any", group_col: str, num_col: str, recipe_name: str) -> pd.DataFrame:
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
