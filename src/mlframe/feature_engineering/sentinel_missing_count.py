"""``add_sentinel_missing_count_feature``: per-row missing/sentinel-value count, configurable sentinel.

Source: 2nd_porto-seguro-safe-driver-prediction.md -- ``train['missing'] = (train == -1).sum(axis=1)``, a
trivial but recurring winner. mlframe already has a NaN-based row-missing-count generator
(``feature_selection.filters._missingness_fe.missingness_count_fit``/``apply_missingness_count``), but that's
hardwired to pandas' own ``.isna()`` semantics -- it can't count rows of an EXPLICIT SENTINEL value (-1, -999,
"N/A", ...), the actual encoding many real datasets use for missingness instead of true NaN. This is the
genuinely missing piece: a configurable-sentinel row-count feature, distinct from (and complementary to) the
existing NaN-based one.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


def add_sentinel_missing_count_feature(
    df: pd.DataFrame,
    sentinel: Any = -1,
    columns: Optional[Sequence[str]] = None,
    feature_name: str = "sentinel_missing_count",
) -> pd.DataFrame:
    """Append a per-row count of ``sentinel``-valued cells as a new column.

    Parameters
    ----------
    df
        Source frame.
    sentinel
        The value treated as "missing" (e.g. ``-1``, ``-999``, ``"N/A"``) -- NOT necessarily NaN.
    columns
        Columns to scan; defaults to all columns in ``df``.
    feature_name
        Name for the appended count column.

    Returns
    -------
    pd.DataFrame
        ``df`` (shallow copy) plus one new ``int64`` column counting, per row, how many of ``columns`` equal
        ``sentinel``.
    """
    cols = list(columns) if columns is not None else list(df.columns)
    # A single whole-array comparison + axis-1 sum beats a per-column Python loop (each iteration paid
    # pandas Series-extraction overhead -- _ixs/_box_col_values/__finalize__, the same column-access
    # bottleneck class fixed elsewhere this session) -- one vectorized numpy pass instead of len(cols) small
    # ones.
    values = df[cols].to_numpy()
    counts = np.sum(values == sentinel, axis=1, dtype=np.int64)

    out = df.copy(deep=False)
    out[feature_name] = counts
    return out


__all__ = ["add_sentinel_missing_count_feature"]
