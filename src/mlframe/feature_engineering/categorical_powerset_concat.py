"""``categorical_powerset_concat``: composite categorical features for every non-empty subset of a key set.

Source: MLFRAME_IDEAS_production.md -- "Auto-generated combinatorial features across all subsets of
categorical keys (2^n - 1 style)": given categorical key columns [A, B, C], generate a string-concat
composite column for EVERY non-empty subset (A, B, C, A+B, A+C, B+C, A+B+C), not just pairwise interactions
-- ``feature_selection.filters.cat_interactions`` already covers statistically-gated pairwise-then-greedy-
k-way interaction mining, a deliberately pruned search; this is the complementary exhaustive-enumeration
utility for a small key set where the caller wants every combination fed to frequency/count encoding, not a
relevance-filtered subset. Reuses ``concat_categorical_group`` per subset rather than reimplementing the
string-join.

Extension: ``prune_against_target`` -- with many keys, most of the 2^n - 1 composites are noise (the idea's
own critique: "adds bloat and downstream selection burden"). When a target is supplied, each generated
composite (order >= 2) is scored and dropped if it fails a threshold, rather than materializing every subset
unconditionally. The scoring reuses two EXISTING primitives rather than a new statistical test:
``training.feature_handling.ordered_target_encoder.ordered_target_encode_batch`` (leak-free expanding
target-mean encoding, smoothed toward the global prior so a high-cardinality noise composite with few rows
per level doesn't spuriously separate the target; the batch variant shares one sort/prior pass across every
composite column instead of repeating it per column) turns the composite's arbitrary string levels into a
numeric score,
then ``feature_selection.drop_near_noise_univariate_auc`` (the existing near-chance-AUC prescreen) flags
composites whose encoded-score AUC sits within ``min_score`` of chance.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from mlframe.feature_engineering.categorical_group_concat import concat_categorical_group
from mlframe.feature_selection.drop_near_noise_univariate_auc import drop_near_noise_univariate_auc
from mlframe.training.feature_handling.ordered_target_encoder import ordered_target_encode_batch


def categorical_powerset_concat(
    df: pd.DataFrame,
    columns: Sequence[str],
    separator: str = "_",
    max_order: int | None = None,
    prune_against_target: Optional[Tuple[np.ndarray, float]] = None,
    prune_smoothing: float = 5.0,
) -> pd.DataFrame:
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
    prune_against_target
        Optional ``(y, min_score)``. When supplied, every generated composite (order >= 2) is target-encoded
        (leak-free, smoothed) and scored by univariate AUC; a composite is DROPPED when its encoded-score AUC
        sits within ``min_score`` of chance (``abs(auc - 0.5) <= min_score``), i.e. ``min_score`` is the
        minimum ``|AUC - 0.5|`` a composite must clear to be kept. ``None`` (default) keeps every composite,
        matching the original unconditional behaviour. Single-column pass-throughs are never pruned.
    prune_smoothing
        Forwarded as ``smoothing`` to :func:`ordered_target_encode_batch` for the pruning score only -- higher
        values pull small/rare composite levels toward the global target prior, avoiding a spuriously
        separable in-sample encoding for high-cardinality noise composites.

    Returns
    -------
    pd.DataFrame
        ``df`` (shallow copy) plus one new ``object`` column per KEPT subset of size >= 2, named by joining
        the subset's column names with ``separator`` (e.g. ``"A_B"``, ``"A_B_C"``).
    """
    if len(columns) < 2:
        raise ValueError("categorical_powerset_concat: need at least 2 columns to build any composite subset")

    upper = len(columns) if max_order is None else min(max_order, len(columns))
    if upper < 2:
        raise ValueError("categorical_powerset_concat: max_order must be >= 2 to produce any composite column")

    out = df.copy(deep=False)
    subsets: List[Sequence[str]] = [subset for order in range(2, upper + 1) for subset in combinations(columns, order)]

    composite_names: List[str] = []
    for subset in subsets:
        feature_name = separator.join(subset)
        out = concat_categorical_group(out, columns=list(subset), separator=separator, feature_name=feature_name)
        composite_names.append(feature_name)

    if prune_against_target is not None and composite_names:
        y, min_score = prune_against_target
        y_arr = np.asarray(y)
        encoded = pd.DataFrame(
            ordered_target_encode_batch({name: out[name].to_numpy() for name in composite_names}, y_arr, smoothing=prune_smoothing)
        )
        dropped = drop_near_noise_univariate_auc(encoded, y_arr, columns=composite_names, tolerance=min_score)
        if dropped:
            out = out.drop(columns=dropped)

    return out


__all__ = ["categorical_powerset_concat"]
