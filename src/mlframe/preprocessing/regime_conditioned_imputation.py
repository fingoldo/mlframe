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
) -> pd.DataFrame:
    """Fill each numeric column's NaNs with the median computed WITHIN its row's ``regime_col`` group.

    Falls back to the column's GLOBAL median when a (regime, column) combination has no observed values at
    all (all-NaN within that regime) -- never leaves a fillable NaN unfilled just because one regime happens
    to be sparse for that particular column.

    Parameters
    ----------
    df
        Frame to impute.
    regime_col
        Categorical/regime column whose groups define the conditioning; rows with a missing ``regime_col``
        value fall back to the column's global median (no regime to condition on).
    feature_cols
        Numeric columns to impute; defaults to every numeric column other than ``regime_col``.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with NaNs in ``feature_cols`` filled.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != regime_col]
    feature_cols = list(feature_cols)

    out = df.copy()
    regime_groups = out[regime_col]
    global_medians = out[feature_cols].median()
    regime_medians = out.groupby(regime_col)[feature_cols].transform("median")

    for col in feature_cols:
        fill_values = regime_medians[col].where(regime_medians[col].notna(), global_medians[col])
        fill_values = fill_values.where(regime_groups.notna(), global_medians[col])
        out[col] = out[col].where(out[col].notna(), fill_values)

    return out


__all__ = ["regime_conditioned_median_fill"]
