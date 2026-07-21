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

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd


def _mode_or_nan(s: pd.Series) -> Union[int, float, str]:
    """First mode of ``s`` ignoring NaNs, or ``np.nan`` if ``s`` has no non-null values."""
    mode_vals = s.mode(dropna=True)
    return mode_vals.iloc[0] if len(mode_vals) else np.nan


def fit_missing_indicator_imputation(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    strategy: str = "median",
    fill_values: Optional[Dict[str, Union[int, float, str]]] = None,
    group_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Learn, per column, the fill statistic (global and, if ``group_col`` given, per-group) to replay via
    :func:`apply_missing_indicator_imputation`.

    Fit on train ONCE and replay the returned state on val/test/inference data -- calling
    ``impute_with_missing_indicator`` (or this function) separately on each split derives each split's fill
    values from its OWN distribution, a train/serve statistic mismatch (and, if a frame spans train+val/test,
    direct look-ahead leakage into train's fill values).

    Columns with zero missing values in ``df`` are skipped (nothing to learn), matching
    ``impute_with_missing_indicator``'s own skip behavior.

    Parameters mirror :func:`impute_with_missing_indicator` (minus ``indicator_suffix``, an apply-time
    concern only).

    Returns
    -------
    dict
        Opaque fit state to pass to :func:`apply_missing_indicator_imputation`.
    """
    if strategy not in ("median", "mean", "mode"):
        raise ValueError(f"fit_missing_indicator_imputation: strategy must be 'median', 'mean', or 'mode'; got {strategy!r}.")
    fill_values = fill_values or {}

    cols = list(columns) if columns is not None else [c for c in df.columns if df[c].isna().any()]
    group_by = df.groupby(group_col, dropna=False) if group_col is not None else None

    columns_state: Dict[str, Dict[str, Any]] = {}
    for col in cols:
        if not df[col].isna().any():
            continue
        if col in fill_values:
            columns_state[col] = {"kind": "explicit", "value": fill_values[col]}
            continue

        if strategy == "median":
            global_fill = df[col].median()
        elif strategy == "mean":
            global_fill = df[col].mean()
        else:  # mode
            global_fill = _mode_or_nan(df[col])

        if group_col is None:
            columns_state[col] = {"kind": "global", "value": global_fill}
        else:
            assert group_by is not None
            if strategy == "median":
                group_stat = group_by[col].median()
            elif strategy == "mean":
                group_stat = group_by[col].mean()
            else:  # mode: no vectorized groupby reduction
                group_stat = group_by[col].agg(_mode_or_nan)
            columns_state[col] = {"kind": "group", "group_stat": group_stat, "global_fallback": global_fill}

    return {"group_col": group_col, "columns": columns_state}


def apply_missing_indicator_imputation(df: pd.DataFrame, fit_state: Dict[str, Any], indicator_suffix: str = "_was_missing") -> pd.DataFrame:
    """Replay a fit state learned by :func:`fit_missing_indicator_imputation` onto ``df`` (train, val, test,
    or a single inference row -- the same learned fill values every time, no recomputation from ``df``
    itself).

    The paired ``{col}{indicator_suffix}`` column always reflects THIS ``df``'s own missingness (that part is
    never "learned" -- it is inherently per-row); only the FILL VALUE is replayed from ``fit_state``. Columns
    in ``fit_state`` but absent from ``df`` are silently skipped.
    """
    group_col = fit_state["group_col"]
    out = df.copy(deep=False)
    for col, info in fit_state["columns"].items():
        if col not in out.columns:
            continue
        mask = out[col].isna()
        out[f"{col}{indicator_suffix}"] = mask.to_numpy()
        if not mask.any():
            continue

        kind = info["kind"]
        if kind in ("explicit", "global"):
            out[col] = out[col].fillna(info["value"])
        else:  # "group"
            assert group_col is not None
            group_fill = out[group_col].map(info["group_stat"]).fillna(info["global_fallback"])
            out[col] = out[col].fillna(group_fill)

    return out


def impute_with_missing_indicator(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    strategy: str = "median",
    fill_values: Optional[Dict[str, Union[int, float, str]]] = None,
    indicator_suffix: str = "_was_missing",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """Impute missing values per column and append a paired ``{col}{indicator_suffix}`` boolean column.

    Single-frame fit+apply convenience wrapper around :func:`fit_missing_indicator_imputation` +
    :func:`apply_missing_indicator_imputation`. For train/test (or train/inference) consistency, call those
    two functions directly instead: fit once on train, apply the SAME learned fill values to every other
    split -- calling this combined wrapper separately per split reproduces the train/serve statistic mismatch.

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
    fit_state = fit_missing_indicator_imputation(df, columns, strategy, fill_values, group_col)
    return apply_missing_indicator_imputation(df, fit_state, indicator_suffix=indicator_suffix)


__all__ = ["impute_with_missing_indicator", "fit_missing_indicator_imputation", "apply_missing_indicator_imputation"]
