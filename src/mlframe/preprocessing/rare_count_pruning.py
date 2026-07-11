"""Absolute-occurrence-count rare-category collapsing and feature dropping.

Source: 3rd_mercedes-benz-greener-manufacturing.md -- categories occurring <=10 times collapsed to a single
"other" bucket; features occurring <20 times across the whole dataset dropped entirely. ABSOLUTE-count
thresholds, not fraction-of-data thresholds, because sample size (4209 rows) was tiny -- a 0.5%-frequency
threshold means something very different at 4,000 rows (20 occurrences) than at 4,000,000 (20,000
occurrences), and for small-N datasets specifically, an absolute floor is the more meaningful knob.

``target_aware`` extension: a fixed absolute count threshold is blind to WHY a category is rare -- some rare
values are pure sampling noise (safe to fold into the catch-all bucket), others are rare-but-genuinely-
distinctive (e.g. a defect code that is rare precisely because it only fires on a specific failure mode, and
correlates strongly with the target whenever it does occur). Collapsing the latter destroys real signal. When
``target_aware=True`` (and ``y`` supplied), a rare value survives collapsing if its target statistic is
STATISTICALLY DISTINGUISHABLE from the catch-all/rest-of-column rate -- reusing
``reporting.charts.category_discriminability.level_woe`` (Laplace-smoothed per-level Weight-of-Evidence,
njit-accelerated) for the rate estimate, plus a standard log-odds-ratio z-test (Woolf's method) against
``z_threshold`` to separate real signal from small-sample noise. Only rare values that fail the significance
test (i.e. are NOT distinguishable from the rest) get collapsed; the rest keep their own value.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from mlframe.reporting.charts.category_discriminability import level_woe


def _target_aware_collapse_mask(codes: np.ndarray, uniques: np.ndarray, y_arr: np.ndarray, rare_values: pd.Index, alpha: float, z_threshold: float) -> np.ndarray:
    """Return a boolean mask over ``uniques`` flagging which of the ``rare_values`` levels to actually collapse.

    A rare level is collapsed only when its Laplace-smoothed WoE against the rest of the column is NOT
    statistically significant (``|z| < z_threshold``, Woolf's log-odds-ratio standard error) -- i.e. its
    target rate is indistinguishable from noise around the catch-all rate. A significant rare level (rare but
    genuinely informative) is left untouched.
    """
    n_levels = len(uniques)
    codes64 = np.ascontiguousarray(codes, dtype=np.int64)
    keep = codes64 >= 0
    tot = np.bincount(codes64[keep], minlength=n_levels).astype(np.float64)
    pos = np.bincount(codes64[keep], weights=y_arr[keep], minlength=n_levels)
    base_rate = float(y_arr[keep].mean()) if keep.any() else 0.5
    woe, _ = level_woe(codes64, y_arr, n_levels, base_rate=base_rate, alpha=alpha)

    total_pos = float(pos.sum())
    total_tot = float(tot.sum())
    a = pos + alpha
    b = (tot - pos) + alpha
    rest_pos = total_pos - pos
    rest_tot = total_tot - tot
    c = rest_pos + alpha
    d = (rest_tot - rest_pos) + alpha
    se = np.sqrt(1.0 / a + 1.0 / b + 1.0 / c + 1.0 / d)
    z = np.divide(woe, se, out=np.zeros_like(woe), where=se > 0)

    rare_value_set = set(rare_values)
    rare_level_mask = np.array([v in rare_value_set for v in uniques])
    result: np.ndarray = rare_level_mask & (np.abs(z) < z_threshold)
    return result


def collapse_rare_categories(
    df: pd.DataFrame,
    columns: Sequence[str],
    min_count: int,
    other_label: str = "__other__",
    *,
    y: Optional[np.ndarray] = None,
    target_aware: bool = False,
    alpha: float = 0.5,
    z_threshold: float = 1.96,
) -> pd.DataFrame:
    """Collapse categorical values occurring fewer than ``min_count`` times (per column) into ``other_label``.

    Parameters
    ----------
    df
        Source frame.
    columns
        Categorical columns to screen.
    min_count
        Values with a total occurrence count strictly below this are collapsed.
    other_label
        Replacement value for collapsed (rare) categories.
    y
        Binary target (0/1), same row order as ``df``. Required when ``target_aware=True``; ignored otherwise.
    target_aware
        Opt-in, default ``False`` (prior absolute-count-only behavior, bit-identical). When ``True``, a rare
        value is collapsed only if its target rate is NOT statistically distinguishable (Woolf's log-odds-
        ratio z-test) from the rest of the column -- a rare-but-genuinely-informative value survives instead
        of being folded into the catch-all bucket.
    alpha
        Laplace smoothing added to positive/negative counts in the WoE and z-test (``target_aware`` only).
    z_threshold
        A rare value survives (is NOT collapsed) when ``|z| >= z_threshold`` (``target_aware`` only).

    Returns
    -------
    pd.DataFrame
        ``df`` (shallow copy) with rare values in ``columns`` replaced by ``other_label``.
    """
    if target_aware and y is None:
        raise ValueError("collapse_rare_categories: target_aware=True requires y")
    y_arr: Optional[np.ndarray] = None
    if target_aware:
        y_arr = np.ascontiguousarray(np.asarray(y), dtype=np.float64)
        if y_arr.shape[0] != len(df):
            raise ValueError(f"collapse_rare_categories: y has {y_arr.shape[0]} rows, df has {len(df)}")

    out = df.copy(deep=False)
    for col in columns:
        counts = df[col].value_counts()
        rare_values = counts[counts < min_count].index
        if len(rare_values) == 0:
            continue
        if not target_aware:
            out[col] = df[col].where(~df[col].isin(rare_values), other_label)
            continue

        codes, uniques = pd.factorize(df[col], sort=False)
        collapse_level_mask = _target_aware_collapse_mask(codes, uniques, y_arr, rare_values, alpha, z_threshold)  # type: ignore[arg-type]
        collapse_values = uniques[collapse_level_mask]
        if len(collapse_values) > 0:
            out[col] = df[col].where(~df[col].isin(collapse_values), other_label)
    return out


def drop_rare_features(df: pd.DataFrame, columns: Optional[Sequence[str]] = None, min_total_count: int = 20, rare_value: object = 0) -> List[str]:
    """Return column names whose "informative" occurrence count falls below ``min_total_count`` -- candidates
    to drop as too-sparse-to-learn-from.

    Matches the source's actual context: sparse one-hot/binary-indicator columns (e.g. expanded from
    high-cardinality categoricals), where "occurring" means the minority/non-default value's count, not raw
    non-null count (a fully-populated binary column with 4189 zeros and 20 ones is exactly the "occurring <20
    times" case the source describes, even though every row is non-null).

    Parameters
    ----------
    df
        Source frame.
    columns
        Columns to screen; defaults to every column of ``df``.
    min_total_count
        Minimum informative-value occurrence count to survive.
    rare_value
        The "default"/uninformative value (e.g. ``0`` for indicator columns, ``False`` for boolean); a
        column's informative count is ``(df[col] != rare_value).sum()`` among non-null rows.

    Returns
    -------
    list of str
        Column names to consider dropping.
    """
    cols = list(columns) if columns is not None else list(df.columns)
    informative_mask = df[cols].ne(rare_value) & df[cols].notna()
    counts = informative_mask.sum(axis=0)
    return [c for c in cols if counts[c] < min_total_count]


__all__ = ["collapse_rare_categories", "drop_rare_features"]
