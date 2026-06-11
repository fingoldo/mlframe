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

Located in ``mlframe.feature_selection`` (NOT under ``.filters``) so importing it does not
trigger the heavy ``filters/__init__.py`` star-import that cascades into ``_numba_utils``
and pays ~0.8s of @njit decorator init at module load (numba caching.py stat-checks). On
combos that never use MRMR this saved cold-start time per process.
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
            # SparseDtype-aware null check: ``Series.isna()`` on a
            # pd.SparseDtype column with ``fill_value=NaN`` returns True
            # for every UNFILLED cell, conflating sparse storage (which
            # is by design "mostly fill_value") with "mostly null". For
            # TF-IDF passthrough (tfidf_keep_sparse=True default), 50
            # vocab features on a 1k-row frame land ~99% unfilled and
            # the pre-screen dropped EVERY tfidf column. Detect sparse
            # and count nulls only among the explicitly-stored values
            # (``sp_values``), which represents the real non-fill data
            # the model will actually see.
            _is_sparse = isinstance(col.dtype, pd.SparseDtype)
            try:
                if _is_sparse:
                    sp_arr = col.values  # pd.arrays.SparseArray
                    fill_v = sp_arr.fill_value
                    fill_is_nan = isinstance(fill_v, float) and np.isnan(fill_v)
                    # null_count = NaN in fill + NaN in stored sp_values
                    sp_vals = np.asarray(sp_arr.sp_values)
                    n_stored = sp_vals.size
                    n_unfilled = n_rows - n_stored
                    stored_nan_count = int(np.isnan(sp_vals).sum()) if sp_vals.dtype.kind == "f" else 0
                    null_count = (n_unfilled if fill_is_nan else 0) + stored_nan_count
                else:
                    # Fast null-count path (bit-identical to ``col.isna().sum()``): for a plain numpy
                    # float column, ``isna`` is exactly ``np.isnan`` on the underlying buffer, but
                    # ``col.isna()`` allocates a fresh boolean Series + dispatches through pandas'
                    # nanops before summing (~6x slower per cProfile 2026-06-12: 61us -> 11us/col).
                    # For numpy integer / bool columns no value can be null, so the count is exactly 0
                    # without touching the data at all. Every other dtype (object/None, datetime/NaT,
                    # nullable Int/Float ext, category, etc.) falls back to the exact ``isna().sum()``.
                    _np_dt = col.dtype
                    if _np_dt == np.float64 or _np_dt == np.float32:
                        null_count = int(np.isnan(col.to_numpy()).sum())
                    elif _np_dt == np.int64 or _np_dt == np.int32 or _np_dt == np.int16 or _np_dt == np.int8 or _np_dt == bool:
                        null_count = 0
                    else:
                        null_count = int(col.isna().sum())
            except Exception:
                null_count = 0
            if null_count > null_cutoff:
                drops.add(col_name)
                continue
            # ``np.issubdtype(col.dtype, np.number)`` raises TypeError on pandas
            # extension dtypes (CategoricalDtype, StringDtype, DatetimeTZDtype) because
            # those are NOT numpy dtypes. Pre-fix the raise bubbled out of this function
            # entirely on any frame containing a Categorical / String column, taking the
            # whole pre-screen pass down with it. Use pd.api.types.is_numeric_dtype which
            # handles every pandas extension dtype gracefully (returns False for cats /
            # strings, True for the nullable Int / Float ExtensionDtypes).
            if not pd.api.types.is_numeric_dtype(col.dtype):
                continue
            try:
                # Sparse columns: variance is degenerate when computed on
                # the dense materialisation (most cells = fill_value). Use
                # variance of the stored sp_values instead -- if every
                # stored cell is identical, the column is constant in the
                # meaningful sense; otherwise it carries signal.
                if _is_sparse:
                    sp_vals = np.asarray(col.values.sp_values)
                    if sp_vals.size <= 1:
                        var_val = 0.0
                    else:
                        var_val = float(np.nanvar(sp_vals))
                else:
                    var_val = float(col.var())
            except (TypeError, ValueError):
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
