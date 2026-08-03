"""Shared per-column categorical-level extraction for the ``training/strategies`` tree
strategies (``xgboost.py`` / ``hgb.py``): independently duplicated across those modules,
consolidated here so a fix can't silently drift out of sync across copies.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


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
    except Exception as e:
        logger.debug("batched unique-level collect failed, falling back to per-column loop: %s", e)
        d: Dict[str, list] = {}
        for c in cols_present:
            try:
                d[c] = df[c].drop_nulls().unique().cast(pl.String).to_list()
            except Exception as e2:  # noqa: PERF203 -- per-iteration fault isolation is intentional, not a hoisting candidate
                logger.debug("per-column unique-level extraction failed for %s: %s", c, e2)
                d[c] = []
        return d


def build_polars_enum_map(self, train_df: "pl.DataFrame", val_df: "pl.DataFrame", cat_features: List[str]) -> "Dict[str, pl.Any]":
    """Build per-column ``pl.Enum`` dtypes from the union of train+val unique values. Test data is intentionally
    excluded -- letting test levels widen the Enum would leak label-time information back into the model's
    accepted-category set. Returns ``{col_name: pl.Enum([...])}`` for every string / Categorical / Enum column
    present in ``train_df``. Columns absent from ``val_df`` contribute only their train levels (still safe).
    Bind as ``ClassName.build_polars_enum_map = build_polars_enum_map`` -- ``self`` is unused (kept for the
    tree-strategy method signature the callers already dispatch through)."""
    import polars as pl

    cat_features = cat_features or []
    candidate_cols = [
        name for name, dtype in train_df.schema.items() if dtype in (pl.Utf8, pl.String) or dtype == pl.Categorical or isinstance(dtype, pl.Enum) or name in cat_features
    ]
    candidate_cols = [c for c in candidate_cols if c in train_df.columns]

    # batch per-column unique extraction into one collect() per frame (train + val). The
    # previous loop did ``df[col].unique()`` per cat col -- on c0031 (15 cat cols x 2 frames = 30 collects
    # per build) that cost ~300ms across the suite via PyLazyFrame.collect. Batched via implode() it's 2
    # collects total per call. Falls back to a per-col loop on any error so one bad cast doesn't poison the frame.
    out: Dict[str, pl.Any] = {}

    train_levels = batched_unique(train_df, candidate_cols)
    val_levels = batched_unique(val_df, candidate_cols) if val_df is not None else {}
    for col in candidate_cols:
        levels: set = set()
        levels.update(train_levels.get(col, []))
        levels.update(val_levels.get(col, []))
        out[col] = pl.Enum(sorted(levels))
    return out
