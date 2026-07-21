"""Regime-conditioned median imputation: fill NaNs with the median WITHIN each row's regime group.

A single global median imputes every missing value the same way regardless of context -- but when a
categorical/regime column genuinely shifts the feature's distribution (a Jane-Street-market-prediction
writeup's single most valuable trick: filling NaNs conditioned on a binary regime indicator column), a
regime-conditioned median is a cheap "simple version of training an auxiliary model to fill NaNs based on
other columns" that captures most of that context at near-zero extra cost.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def fit_regime_conditioned_median(
    df: pd.DataFrame,
    regime_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    extra_regime_cols: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Learn global and regime-conditioned median fill statistics to replay via
    :func:`apply_regime_conditioned_median_fill`.

    Fit on train ONCE and replay the returned state on val/test/inference data -- calling
    ``regime_conditioned_median_fill`` (or this function) separately on each split recomputes fill values
    from EACH split's own distribution (a train/serve statistic mismatch), same architectural gap as F1/F7
    .

    Parameters mirror :func:`regime_conditioned_median_fill` (minus ``min_group_size``, an apply-time
    concern only).

    Returns
    -------
    dict
        Opaque fit state to pass to :func:`apply_regime_conditioned_median_fill`.
    """
    if feature_cols is None:
        exclude = {regime_col, *(extra_regime_cols or ())}
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    feature_cols = list(feature_cols)

    global_medians = df[feature_cols].median()
    regime_medians_by_group = df.groupby(regime_col)[feature_cols].median()

    composite_cols: Optional[List[str]] = None
    composite_medians_by_group = None
    composite_counts_by_group = None
    if extra_regime_cols:
        composite_cols = [regime_col, *extra_regime_cols]
        composite_medians_by_group = df.groupby(composite_cols)[feature_cols].median()
        composite_counts_by_group = df.groupby(composite_cols)[feature_cols].count()

    return {
        "regime_col": regime_col,
        "feature_cols": feature_cols,
        "extra_regime_cols": list(extra_regime_cols) if extra_regime_cols else None,
        "composite_cols": composite_cols,
        "global_medians": global_medians,
        "regime_medians_by_group": regime_medians_by_group,
        "composite_medians_by_group": composite_medians_by_group,
        "composite_counts_by_group": composite_counts_by_group,
    }


def apply_regime_conditioned_median_fill(df: pd.DataFrame, fit_state: Dict[str, Any], min_group_size: int = 1) -> pd.DataFrame:
    """Replay a fit state learned by :func:`fit_regime_conditioned_median` onto ``df`` (train, val, test, or
    inference) -- the same learned medians every time, no recomputation from ``df`` itself.

    A regime value (or composite group) unseen at fit time, or a composite group too sparse per
    ``min_group_size``, falls back the same way the combined :func:`regime_conditioned_median_fill` does:
    composite -> regime -> global.
    """
    regime_col = fit_state["regime_col"]
    feature_cols = fit_state["feature_cols"]
    extra_regime_cols = fit_state["extra_regime_cols"]
    composite_cols = fit_state["composite_cols"]
    global_medians = fit_state["global_medians"]
    regime_medians_by_group = fit_state["regime_medians_by_group"]

    # Shallow copy: every feature_cols column is fully replaced via out[col] = ...where(...) below
    # (never mutated in place), so df's original buffers are never touched.
    out = df.copy(deep=False)
    regime_groups = out[regime_col]

    composite_key_index = None
    composite_valid = None
    if extra_regime_cols:
        assert composite_cols is not None
        composite_key_index = pd.MultiIndex.from_frame(out[composite_cols])
        composite_valid = out[composite_cols].notna().all(axis=1)

    for col in feature_cols:
        fill_values = pd.Series(regime_medians_by_group[col].reindex(regime_groups).to_numpy(), index=out.index)
        fill_values = fill_values.where(fill_values.notna(), global_medians[col])
        fill_values = fill_values.where(regime_groups.notna(), global_medians[col])

        if extra_regime_cols:
            composite_medians_by_group = fit_state["composite_medians_by_group"]
            composite_counts_by_group = fit_state["composite_counts_by_group"]
            composite_median_col = pd.Series(composite_medians_by_group[col].reindex(composite_key_index).to_numpy(), index=out.index)
            composite_count_col = pd.Series(composite_counts_by_group[col].reindex(composite_key_index).to_numpy(), index=out.index)
            composite_reliable = composite_valid & composite_median_col.notna() & (composite_count_col >= min_group_size)
            fill_values = composite_median_col.where(composite_reliable, fill_values)

        out[col] = out[col].where(out[col].notna(), fill_values)

    return out


def regime_conditioned_median_fill(
    df: pd.DataFrame,
    regime_col: str,
    feature_cols: Optional[Sequence[str]] = None,
    extra_regime_cols: Optional[Sequence[str]] = None,
    min_group_size: int = 1,
) -> pd.DataFrame:
    """Fill each numeric column's NaNs with the median computed WITHIN its row's ``regime_col`` group.

    Single-frame fit+apply convenience wrapper around :func:`fit_regime_conditioned_median` +
    :func:`apply_regime_conditioned_median_fill`. For train/test (or train/inference) consistency, call
    those two functions directly instead: fit once on train, apply the SAME learned medians to every other
    split -- calling this combined wrapper separately per split reproduces the train/serve statistic
    mismatch.

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
    fit_state = fit_regime_conditioned_median(df, regime_col, feature_cols, extra_regime_cols)
    return apply_regime_conditioned_median_fill(df, fit_state, min_group_size=min_group_size)


__all__ = ["regime_conditioned_median_fill", "fit_regime_conditioned_median", "apply_regime_conditioned_median_fill"]
