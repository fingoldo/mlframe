"""``drop_raw_after_embedding``: drop a raw high-cardinality categorical once its derived encodings exist.

Source: 1st_talkingdata-adtracking-fraud-detection.md -- "we removed all raw categorical features except app
since we supposed embedding features cover information... jumped public LB from 0.9821 to 0.9828". Once a raw
high-cardinality categorical has been converted into derived features (target/frequency/count encodings,
entity embeddings, SVD/co-occurrence features), keeping the raw column around mostly adds overfitting
surface (a tree model can memorize per-category splits the encoding already summarized) rather than genuine
signal -- this is a small, explicit drop step, not a generic redundancy pruner: it only ever removes the RAW
column, never a derived one, and only once the derived columns it depends on are actually present.
"""
from __future__ import annotations

from typing import Dict, Sequence

import pandas as pd


def drop_raw_after_embedding(df: pd.DataFrame, raw_to_derived: Dict[str, Sequence[str]], min_derived_present: int = 1) -> pd.DataFrame:
    """Drop each raw column in ``raw_to_derived`` once enough of its derived columns are present in ``df``.

    Parameters
    ----------
    df
        Frame containing both raw categorical columns and their derived features.
    raw_to_derived
        Mapping from a raw column name to the derived column names built from it (e.g. an entity-embedding
        or target-encoding step's output columns).
    min_derived_present
        Minimum number of a raw column's derived columns that must already be present in ``df`` before that
        raw column is dropped -- guards against dropping a raw column whose encoding step never ran (e.g. was
        skipped due to low cardinality, or failed upstream), which would silently destroy the only signal
        source for that column.

    Returns
    -------
    pd.DataFrame
        ``df`` (shallow copy) with each qualifying raw column removed. Derived columns are always kept as-is.
    """
    to_drop = []
    for raw_col, derived_cols in raw_to_derived.items():
        if raw_col not in df.columns:
            continue
        n_present = sum(1 for c in derived_cols if c in df.columns)
        if n_present >= min_derived_present:
            to_drop.append(raw_col)

    return df.drop(columns=to_drop)


__all__ = ["drop_raw_after_embedding"]
