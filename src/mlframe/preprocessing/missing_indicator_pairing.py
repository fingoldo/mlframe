"""``impute_with_missing_indicator``: pair imputation with a companion "was missing" boolean column.

Sklearn's ``SimpleImputer(add_indicator=True)`` does exactly this, but mlframe's own imputation helpers
(``preprocessing.regime_conditioned_imputation.regime_conditioned_median_fill``,
``preprocessing.sibling_group_cold_start_fill.sibling_group_cold_start_fill``, and the median-fill logic
inside ``preprocessing.cleaning``) all return only the filled values -- none emits a paired missingness flag,
so a downstream model cannot distinguish "genuinely observed at this value" from "imputed" (a real distinction
whenever missingness itself is informative, e.g. MNAR data). This is a standalone, imputation-method-agnostic
wrapper: it works with ANY fill (mean/median/mode/constant/a caller-supplied per-column fill value), not just
sklearn's own imputers, so it composes with mlframe's existing fill logic rather than replacing it.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd


def _mode_or_nan(s: pd.Series) -> Union[int, float, str]:
    """First mode of ``s`` ignoring NaNs, or ``np.nan`` if ``s`` has no non-null values."""
    mode_vals = s.mode(dropna=True)
    return mode_vals.iloc[0] if len(mode_vals) else np.nan


def impute_with_missing_indicator(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    strategy: str = "median",
    fill_values: Optional[Dict[str, Union[int, float, str]]] = None,
    indicator_suffix: str = "_was_missing",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Impute missing values per column and append a paired ``{col}{indicator_suffix}`` boolean column.

    Parameters
    ----------
    df
        Source frame.
    columns
        Columns to impute; defaults to every column in ``df`` that has at least one missing value.
    strategy
        ``"median"``, ``"mean"``, or ``"mode"`` for numeric/categorical fill; ignored for a column present in
        ``fill_values`` (that value wins).
    fill_values
        Optional explicit ``{column: value}`` overrides, applied instead of ``strategy`` for those columns.
    indicator_suffix
        Suffix for the companion boolean indicator column.
    group_col
        Optional column name. When supplied, the fill statistic is computed per-group (grouping by this
        column) instead of a single global statistic -- useful whenever the feature's typical value varies
        meaningfully across groups (e.g. impute missing "income" with the median income within the person's
        region rather than the global median). A group with zero non-missing values falls back to the global
        statistic, so no fill is ever NaN. ``group_col`` itself is never imputed by this call. Ignored for a
        column present in ``fill_values`` (that value wins, exactly as without ``group_col``).

    Returns
    -------
    pd.DataFrame
        ``df`` (shallow copy) with each imputed column's NaNs filled in place, plus one new boolean
        ``{col}{indicator_suffix}`` column per imputed column marking which rows were originally missing.
    """
    if strategy not in ("median", "mean", "mode"):
        raise ValueError(f"impute_with_missing_indicator: strategy must be 'median', 'mean', or 'mode'; got {strategy!r}.")
    fill_values = fill_values or {}

    cols = list(columns) if columns is not None else [c for c in df.columns if df[c].isna().any()]
    out = df.copy(deep=False)
    # grouped once and reused across all columns -- rebuilding the groupby index per column would repeat
    # the (non-trivial) group-key factorization work for every imputed column.
    group_by = out.groupby(group_col, dropna=False) if group_col is not None else None

    for col in cols:
        mask = out[col].isna()
        if not mask.any():
            continue
        out[f"{col}{indicator_suffix}"] = mask.to_numpy()
        if col in fill_values:
            out[col] = out[col].fillna(fill_values[col])
            continue

        if strategy == "median":
            global_fill = out[col].median()
        elif strategy == "mean":
            global_fill = out[col].mean()
        else:  # mode
            global_fill = _mode_or_nan(out[col])

        if group_col is None:
            out[col] = out[col].fillna(global_fill)
        else:
            assert group_by is not None
            if strategy == "median":
                group_fill = group_by[col].transform("median")
            elif strategy == "mean":
                group_fill = group_by[col].transform("mean")
            else:  # mode: no vectorized groupby reduction, but per-group (not per-row) cost via agg + map
                group_stat = group_by[col].agg(_mode_or_nan)
                group_fill = out[group_col].map(group_stat)
            group_fill = group_fill.fillna(global_fill)  # groups with zero non-missing values
            out[col] = out[col].fillna(group_fill)

    return out


__all__ = ["impute_with_missing_indicator"]
