"""``add_sentinel_missing_count_feature``: per-row missing/sentinel-value count, configurable sentinel.

Source: 2nd_porto-seguro-safe-driver-prediction.md -- ``train['missing'] = (train == -1).sum(axis=1)``, a
trivial but recurring winner. mlframe already has a NaN-based row-missing-count generator
(``feature_selection.filters._missingness_fe.missingness_count_fit``/``apply_missingness_count``), but that's
hardwired to pandas' own ``.isna()`` semantics -- it can't count rows of an EXPLICIT SENTINEL value (-1, -999,
"N/A", ...), the actual encoding many real datasets use for missingness instead of true NaN. This is the
genuinely missing piece: a configurable-sentinel row-count feature, distinct from (and complementary to) the
existing NaN-based one.

Extension: real datasets rarely use ONE sentinel value uniformly across every column (one column's missing
code might be ``-1``, another's ``-999``, another's ``"N/A"``) -- a single global ``sentinel`` either misses
columns that don't use it or, worse, silently miscounts a column whose legitimate values happen to collide
with it. ``per_column_sentinels`` lets each column carry its own sentinel value (or set of values), and
``auto_detect_sentinels`` flags a likely sentinel per column by spotting a single non-NaN value occurring far
more often than the rest of that column's empirical value-frequency distribution would predict (a frequency
spike), the same signature a human eyeballing ``value_counts()`` would use to spot a sentinel code.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

SentinelSpec = Union[Any, Sequence[Any]]


def _as_sentinel_set(spec: SentinelSpec) -> set:
    if isinstance(spec, (list, tuple, set, frozenset, np.ndarray)):
        return set(spec)
    return {spec}


def detect_column_sentinel(
    series: pd.Series,
    min_ratio: float = 5.0,
    min_fraction: float = 0.01,
) -> Optional[Any]:
    """Flag a likely sentinel value in ``series`` via an anomalous value-frequency spike.

    A sentinel-coded column typically has one value (e.g. ``-1``) occurring MUCH more often than any other
    distinct value would under an otherwise-smooth empirical distribution -- unlike a genuine mode, which
    tends to be only mildly more frequent than its neighbors. This heuristic compares the most frequent
    value's count to the MEDIAN count of all other distinct values: a real sentinel code produces a large
    ratio (few other values repeat anywhere near as often), while ordinary numeric/categorical data does not.

    Parameters
    ----------
    series
        Column to inspect.
    min_ratio
        Minimum ratio of (top value's count) / (median count of the remaining values) to flag a spike.
    min_fraction
        Minimum fraction of non-null rows the top value must cover to be considered (guards against
        flagging a value that's merely "most common among many rare ones" in a tiny/sparse column).

    Returns
    -------
    Optional[Any]
        The detected sentinel value, or ``None`` if no anomalous spike was found (fewer than 2 distinct
        non-null values also returns ``None`` -- there's nothing to compare against).
    """
    value_counts = series.value_counts(dropna=True)
    if len(value_counts) < 2:
        return None

    top_value = value_counts.index[0]
    top_count = value_counts.iloc[0]
    other_counts = value_counts.iloc[1:]
    median_other = other_counts.median()
    if median_other <= 0:
        median_other = 1.0

    non_null_total = int(value_counts.sum())
    ratio = top_count / median_other
    fraction = top_count / non_null_total if non_null_total else 0.0

    if ratio >= min_ratio and fraction >= min_fraction:
        return top_value
    return None


def detect_sentinel_values(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    min_ratio: float = 5.0,
    min_fraction: float = 0.01,
) -> dict[str, Any]:
    """Run :func:`detect_column_sentinel` over ``columns`` (defaults to all of ``df``), skipping columns with no spike."""
    cols = list(columns) if columns is not None else list(df.columns)
    detected: dict[str, Any] = {}
    for col in cols:
        value = detect_column_sentinel(df[col], min_ratio=min_ratio, min_fraction=min_fraction)
        if value is not None:
            detected[col] = value
    return detected


def add_sentinel_missing_count_feature(
    df: pd.DataFrame,
    sentinel: Any = -1,
    columns: Optional[Sequence[str]] = None,
    feature_name: str = "sentinel_missing_count",
    per_column_sentinels: Optional[Mapping[str, SentinelSpec]] = None,
    auto_detect_sentinels: bool = False,
    auto_detect_min_ratio: float = 5.0,
    auto_detect_min_fraction: float = 0.01,
) -> pd.DataFrame:
    """Append a per-row count of ``sentinel``-valued cells as a new column.

    Parameters
    ----------
    df
        Source frame.
    sentinel
        The value treated as "missing" (e.g. ``-1``, ``-999``, ``"N/A"``) -- NOT necessarily NaN. Used as-is
        for every column not covered by ``per_column_sentinels``/``auto_detect_sentinels``.
    columns
        Columns to scan; defaults to all columns in ``df``.
    feature_name
        Name for the appended count column.
    per_column_sentinels
        Opt-in per-column override: maps a column name to its own sentinel value, or a sequence of sentinel
        values for that column (a column can have more than one missingness code). Columns not present in
        this mapping fall back to auto-detection (if enabled) then the global ``sentinel``. Passing this
        (or ``auto_detect_sentinels=True``) switches to the per-column code path; omitting both reproduces
        the original single-global-sentinel behavior bit-for-bit.
    auto_detect_sentinels
        If ``True``, run :func:`detect_column_sentinel` on any column not already covered by
        ``per_column_sentinels`` and use the flagged value (if any) as that column's sentinel instead of
        the global ``sentinel``.
    auto_detect_min_ratio, auto_detect_min_fraction
        Forwarded to :func:`detect_column_sentinel`.

    Returns
    -------
    pd.DataFrame
        ``df`` (shallow copy) plus one new ``int64`` column counting, per row, how many of ``columns`` equal
        their resolved sentinel(s).
    """
    cols = list(columns) if columns is not None else list(df.columns)

    if per_column_sentinels is None and not auto_detect_sentinels:
        # Original path, kept byte-for-byte: a single whole-array comparison + axis-1 sum beats a per-column
        # Python loop (each iteration paid pandas Series-extraction overhead -- _ixs/_box_col_values/
        # __finalize__, the same column-access bottleneck class fixed elsewhere this session).
        values = df[cols].to_numpy()
        counts = np.sum(values == sentinel, axis=1, dtype=np.int64)
    else:
        resolved: dict[str, set] = {}
        for col in cols:
            if per_column_sentinels is not None and col in per_column_sentinels:
                resolved[col] = _as_sentinel_set(per_column_sentinels[col])
            elif auto_detect_sentinels:
                detected = detect_column_sentinel(df[col], min_ratio=auto_detect_min_ratio, min_fraction=auto_detect_min_fraction)
                resolved[col] = {detected} if detected is not None else set()
            else:
                resolved[col] = {sentinel}

        # Per-column sentinel sets can't share one vectorized array comparison (each column's sentinel set
        # differs), so this path pays one np.isin per column -- still one numpy pass per column, no Python
        # per-cell loop.
        counts = np.zeros(len(df), dtype=np.int64)
        for col in cols:
            sentinel_set = resolved[col]
            if not sentinel_set:
                continue
            col_values = df[col].to_numpy()
            counts += np.isin(col_values, list(sentinel_set)).astype(np.int64)

    out = df.copy(deep=False)
    out[feature_name] = counts
    return out


__all__ = ["add_sentinel_missing_count_feature", "detect_column_sentinel", "detect_sentinel_values"]
