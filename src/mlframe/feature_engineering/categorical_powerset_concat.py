"""``categorical_powerset_concat``: composite categorical features for every non-empty subset of a key set.

Source: MLFRAME_IDEAS_production.md -- "Auto-generated combinatorial features across all subsets of
categorical keys (2^n - 1 style)": given categorical key columns [A, B, C], generate a string-concat
composite column for EVERY non-empty subset (A, B, C, A+B, A+C, B+C, A+B+C), not just pairwise interactions
-- ``feature_selection.filters.cat_interactions`` already covers statistically-gated pairwise-then-greedy-
k-way interaction mining, a deliberately pruned search; this is the complementary exhaustive-enumeration
utility for a small key set where the caller wants every combination fed to frequency/count encoding, not a
relevance-filtered subset. Reuses ``concat_categorical_group`` per subset rather than reimplementing the
string-join.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Sequence

import pandas as pd

from mlframe.feature_engineering.categorical_group_concat import concat_categorical_group


def categorical_powerset_concat(df: pd.DataFrame, columns: Sequence[str], separator: str = "_", max_order: int | None = None) -> pd.DataFrame:
    """Append one composite categorical column per non-empty subset of ``columns`` (2^n - 1 total).

    Parameters
    ----------
    df
        Source frame.
    columns
        Categorical key columns to combine. Single-column "subsets" are passed through unchanged (no new
        column, since ``df[col]`` already exists) -- only subsets of size >= 2 produce a new composite column.
    separator
        Joiner between each column's value within a subset, forwarded to ``concat_categorical_group``.
    max_order
        Cap subset size (e.g. ``max_order=2`` limits to pairwise combos only); ``None`` uses the full
        powerset up to ``len(columns)``. Guards against the 2^n blowup for large key sets -- with n=10 keys
        the full powerset is 1023 columns, so callers with many keys should pass an explicit cap.

    Returns
    -------
    pd.DataFrame
        ``df`` (shallow copy) plus one new ``object`` column per subset of size >= 2, named by joining the
        subset's column names with ``separator`` (e.g. ``"A_B"``, ``"A_B_C"``).
    """
    if len(columns) < 2:
        raise ValueError("categorical_powerset_concat: need at least 2 columns to build any composite subset")

    upper = len(columns) if max_order is None else min(max_order, len(columns))
    if upper < 2:
        raise ValueError("categorical_powerset_concat: max_order must be >= 2 to produce any composite column")

    out = df.copy(deep=False)
    subsets: List[Sequence[str]] = [subset for order in range(2, upper + 1) for subset in combinations(columns, order)]

    for subset in subsets:
        feature_name = separator.join(subset)
        out = concat_categorical_group(out, columns=list(subset), separator=separator, feature_name=feature_name)

    return out


__all__ = ["categorical_powerset_concat"]
