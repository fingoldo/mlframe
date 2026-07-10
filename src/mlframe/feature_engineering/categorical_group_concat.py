"""``concat_categorical_group``: build a composite categorical column from multiple raw categorical columns.

Source: 2nd_porto-seguro-safe-driver-prediction.md -- building ``new_ind``/``new_reg``/``new_car`` by
string-concatenating groups of raw categorical columns, then computing frequency/count encodings of both raw
and composite categoricals. mlframe already has full frequency/count encoding coverage
(``feature_selection.filters._count_freq_interaction_fe.frequency_encode_fit``/``count_encode_fit``) -- the
genuinely missing piece is the CONCATENATOR itself, a standalone precursor step (feed its output into the
existing frequency encoder) rather than the vectorized ``.str.cat`` pattern currently buried inline inside
``two_step_target_encode.py``.
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd


def concat_categorical_group(df: pd.DataFrame, columns: Sequence[str], separator: str = "_", feature_name: str = "concat_group") -> pd.DataFrame:
    """Append a composite categorical column built by string-concatenating ``columns``.

    Parameters
    ----------
    df
        Source frame.
    columns
        Categorical columns to concatenate, in order.
    separator
        Joiner between each column's value.
    feature_name
        Name for the appended composite column.

    Returns
    -------
    pd.DataFrame
        ``df`` (shallow copy) plus one new ``object`` (string) column: each row's values from ``columns``
        joined by ``separator``. Uses vectorized ``Series.str.cat`` (measured ~13x faster than
        ``df[columns].agg(separator.join, axis=1)`` at scale, per the existing pattern this reuses from
        ``two_step_target_encode.py``).
    """
    if len(columns) < 2:
        raise ValueError("concat_categorical_group: need at least 2 columns to concatenate")

    first, *rest = columns
    composite = df[first].astype(str).str.cat([df[c].astype(str) for c in rest], sep=separator)

    out = df.copy(deep=False)
    out[feature_name] = composite
    return out


__all__ = ["concat_categorical_group"]
