"""Shared per-column categorical-level extraction for the ``training/strategies`` tree
strategies (``xgboost.py`` / ``hgb.py``): independently duplicated across those modules,
consolidated here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    import polars as pl


def batched_unique(df: "pl.DataFrame", candidate_cols: List[str]) -> "Dict[str, list]":
    """Extract per-column unique values for all ``candidate_cols`` present in ``df`` in one lazy collect() instead of one collect per column; on failure (bad cast, etc.) falls back to a per-column loop so a single poisoned column doesn't blank out the whole frame's categories."""
    import polars as pl

    cols_present = [c for c in candidate_cols if c in df.columns]
    if not cols_present:
        return {}
    try:
        lf = df.lazy().select([pl.col(c).cast(pl.String).drop_nulls().unique().implode().alias(c) for c in cols_present])
        row = lf.collect()
        return {c: row[c][0].to_list() for c in cols_present}
    except Exception:
        d: Dict[str, list] = {}
        for c in cols_present:
            try:
                d[c] = df[c].drop_nulls().unique().cast(pl.String).to_list()
            except Exception:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
                d[c] = []
        return d
