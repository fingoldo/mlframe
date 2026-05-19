"""Unsupervised pre-screen filter for the training suite.

Drops columns that are trivially useless BEFORE the expensive per-target FS pipeline
(MRMR / RFECV / BorutaShap) runs. Conservative defaults only:

  - variance below ``max(variance_threshold, 1e-24)`` (default 0.0): constant / no-information columns.
    The 1e-24 floor catches float64 FP noise on literal-constant columns (where ``np.full(n, c).var()`` lands
    at ~1e-28); without it ``var() == 0`` literally never matches for non-integer constants.
  - null fraction > ``null_fraction_threshold`` (default 0.99): columns that are almost entirely missing.

Train-only fit by contract: caller computes the drop set on the train split, then reapplies
to val / test. The pre-screen never reads val / test, so it cannot leak distribution
information from held-out data. Aggressive correlation / cardinality filters are intentionally
out of scope - those risk dropping joint-informative features and would break MRMR's strength.

Returns the list of columns to drop. ``compute_unsupervised_drops`` is polars / pandas agnostic;
``apply_drops`` reapplies the list across both backends.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

try:
    import polars as pl
except Exception:
    pl = None

try:
    import pandas as pd
except Exception:
    pd = None


def compute_unsupervised_drops(
    train_df,
    variance_threshold: float = 0.0,
    null_fraction_threshold: float = 0.99,
    protected_columns: Iterable[str] = (),
) -> list[str]:
    """Compute the list of columns to drop given a train-only frame.

    Parameters
    ----------
    train_df : polars.DataFrame | pandas.DataFrame
        The training split (val / test must NOT be passed - train-only fit by contract).
    variance_threshold : float
        Drop columns where ``var() <= variance_threshold``. Default 0.0 (drops only effectively
        constant columns; the <= is robust to ~1e-28 FP noise on literal-constant float columns).
    null_fraction_threshold : float
        Drop columns where ``null_count / len > null_fraction_threshold``. Default 0.99.
    protected_columns : iterable of str
        Column names that must never be dropped (e.g. target columns, group_id columns).

    Returns
    -------
    list[str]
        Sorted list of column names to drop. Empty list if nothing matches.
    """
    if train_df is None:
        return []
    protected = set(protected_columns) if protected_columns else set()
    n_rows = int(train_df.shape[0]) if hasattr(train_df, "shape") else 0
    if n_rows == 0:
        return []

    drops: set[str] = set()
    # Float-noise floor: pd.Series(np.full(n, 3.14)).var() returns ~1.24e-28, not 0.0, so a literal
    # ``<= 0.0`` check would never fire on constant float columns. Use the larger of caller's threshold
    # and 1e-24 to make "variance = 0" the intuitive contract regardless of underlying FP arithmetic.
    _var_cutoff = max(float(variance_threshold), 1e-24)

    # Polars branch
    if pl is not None and isinstance(train_df, pl.DataFrame):
        null_cutoff = null_fraction_threshold * n_rows
        for col_name in train_df.columns:
            if col_name in protected:
                continue
            col = train_df[col_name]
            try:
                null_count = int(col.null_count())
            except Exception:
                null_count = 0
            if null_count > null_cutoff:
                drops.add(col_name)
                continue
            # Variance is well-defined only for numeric dtypes. For non-numeric (string / categorical /
            # struct), the variance==0 rule does not apply; constant string columns are typically caught
            # downstream by the null-fraction or by per-target FS itself. Polars raises for var() on
            # non-numeric so guard via dtype check.
            try:
                dt = col.dtype
                is_numeric = dt.is_numeric() if hasattr(dt, "is_numeric") else False
            except Exception:
                is_numeric = False
            if not is_numeric:
                continue
            try:
                var_val = col.var()
            except Exception:
                var_val = None
            if var_val is None:
                # All-null after the null check would have been caught above; treat as constant.
                drops.add(col_name)
                continue
            try:
                # Use <= rather than == because floating-point arithmetic on constant columns can
                # yield var ~= 1e-28 instead of exact zero (numerical floor of float64 fma).
                if float(var_val) <= _var_cutoff:
                    drops.add(col_name)
            except (TypeError, ValueError):
                pass
        return sorted(drops)

    # Pandas branch
    if pd is not None and isinstance(train_df, pd.DataFrame):
        null_cutoff = null_fraction_threshold * n_rows
        for col_name in train_df.columns:
            if col_name in protected:
                continue
            col = train_df[col_name]
            try:
                null_count = int(col.isna().sum())
            except Exception:
                null_count = 0
            if null_count > null_cutoff:
                drops.add(col_name)
                continue
            if not np.issubdtype(col.dtype, np.number):
                continue
            try:
                var_val = float(col.var())
            except Exception:
                var_val = None
            if var_val is None or np.isnan(var_val):
                drops.add(col_name)
                continue
            if var_val <= _var_cutoff:
                drops.add(col_name)
        return sorted(drops)

    return []


def apply_drops(df, drop_cols: list[str]):
    """Return df with ``drop_cols`` removed. Polars / pandas agnostic, idempotent on missing columns."""
    if df is None or not drop_cols:
        return df
    existing = [c for c in drop_cols if c in getattr(df, "columns", [])]
    if not existing:
        return df
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.drop(existing)
    if pd is not None and isinstance(df, pd.DataFrame):
        return df.drop(columns=existing)
    return df
