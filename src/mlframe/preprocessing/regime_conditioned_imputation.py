"""Regime-conditioned median imputation: fill NaNs with the median WITHIN each row's regime group.

A single global median imputes every missing value the same way regardless of context -- but when a
categorical/regime column genuinely shifts the feature's distribution (a Jane-Street-market-prediction
writeup's single most valuable trick: filling NaNs conditioned on a binary regime indicator column), a
regime-conditioned median is a cheap "simple version of training an auxiliary model to fill NaNs based on
other columns" that captures most of that context at near-zero extra cost.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


def regime_conditioned_median_fill(
    df: pd.DataFrame,
    regime_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    extra_regime_cols: Optional[Sequence[str]] = None,
    min_group_size: int = 1,
) -> pd.DataFrame:
    """Fill each numeric column's NaNs with the median computed WITHIN its row's ``regime_col`` group.

    Falls back to the column's GLOBAL median when a (regime, column) combination has no observed values at
    all (all-NaN within that regime) -- never leaves a fillable NaN unfilled just because one regime happens
    to be sparse for that particular column.

    ``extra_regime_cols`` opts into a hierarchical, multi-column composite-regime mode: the fill is first
    conditioned on the joint group ``(regime_col, *extra_regime_cols)`` (the composite key genuinely captures
    interaction effects neither column carries alone), falling back to progressively coarser conditioning --
    composite group -> ``regime_col``-only group -> global median -- whenever a composite group has fewer
    than ``min_group_size`` observed values for that column, since a median computed from a handful of rows
    is unreliable even though it is technically "observed". Omitting ``extra_regime_cols`` (the default)
    reproduces the original single-column behavior exactly.

    Parameters
    ----------
    df
        Frame to impute.
    regime_col
        Categorical/regime column whose groups define the conditioning; rows with a missing ``regime_col``
        value fall back to the column's global median (no regime to condition on).
    feature_cols
        Numeric columns to impute; defaults to every numeric column other than ``regime_col`` (and
        ``extra_regime_cols``, when given).
    extra_regime_cols
        Additional categorical/regime columns to compose with ``regime_col`` into a joint group key for a
        finer-grained conditional median, with hierarchical fallback to the coarser ``regime_col``-only
        median (and then the global median) when the composite group is too sparse. ``None`` (default)
        keeps the original single-regime-column behavior, bit-identical.
    min_group_size
        Minimum count of observed (non-NaN) values a composite group must have for a given column before
        its median is trusted; below this the fill falls back to the coarser ``regime_col`` median. Only
        consulted when ``extra_regime_cols`` is given.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with NaNs in ``feature_cols`` filled.
    """
    if feature_cols is None:
        exclude = {regime_col, *(extra_regime_cols or ())}
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    feature_cols = list(feature_cols)

    out = df.copy()
    regime_groups = out[regime_col]
    global_medians = out[feature_cols].median()
    regime_medians = out.groupby(regime_col)[feature_cols].transform("median")

    if extra_regime_cols:
        composite_cols = [regime_col, *extra_regime_cols]
        composite_valid = out[composite_cols].notna().all(axis=1)
        composite_counts = out.groupby(composite_cols)[feature_cols].transform("count")
        composite_medians = out.groupby(composite_cols)[feature_cols].transform("median")

    for col in feature_cols:
        fill_values = regime_medians[col].where(regime_medians[col].notna(), global_medians[col])
        fill_values = fill_values.where(regime_groups.notna(), global_medians[col])

        if extra_regime_cols:
            composite_reliable = composite_valid & composite_medians[col].notna() & (composite_counts[col] >= min_group_size)
            fill_values = composite_medians[col].where(composite_reliable, fill_values)

        out[col] = out[col].where(out[col].notna(), fill_values)

    return out


__all__ = ["regime_conditioned_median_fill"]
