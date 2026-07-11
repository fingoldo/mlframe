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


def impute_with_missing_indicator(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    strategy: str = "median",
    fill_values: Optional[Dict[str, Union[int, float, str]]] = None,
    indicator_suffix: str = "_was_missing",
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

    for col in cols:
        mask = out[col].isna()
        if not mask.any():
            continue
        out[f"{col}{indicator_suffix}"] = mask.to_numpy()
        if col in fill_values:
            fill_value = fill_values[col]
        elif strategy == "median":
            fill_value = out[col].median()
        elif strategy == "mean":
            fill_value = out[col].mean()
        else:  # mode
            mode_vals = out[col].mode(dropna=True)
            fill_value = mode_vals.iloc[0] if len(mode_vals) else np.nan
        out[col] = out[col].fillna(fill_value)

    return out


__all__ = ["impute_with_missing_indicator"]
