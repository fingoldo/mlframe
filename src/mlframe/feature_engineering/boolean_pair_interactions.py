"""``boolean_pair_interactions``: pairwise AND/OR/XOR expansion for binary/one-hot columns.

Source: 2nd_playground-series-s3e13.md -- combined every pair of one-hot binary features with AND, OR, and
XOR operators, generating 6000+ candidate features, then filtered via forward selection using the actual
competition metric (MAP@3). Analogous to mlframe's existing continuous-valued pairwise recipe generators
(``fe_baselines.trivial_pair_features``, ``polynom_pair_fe``) but for boolean logic specifically -- ``mul``/
``maxab``/``minab`` coincide numerically with AND/OR for strict ``{0, 1}`` inputs, but no existing recipe
detects binary-valued columns and routes them here; XOR in particular has no continuous-arithmetic equivalent
already in the recipe set.

Generates candidates only, wholesale keeping is NOT the intent -- the source itself pruned the combinatorial
explosion via forward selection on the real metric; callers should feed the output through mlframe's existing
MRMR/forward-selection tooling rather than using every generated column.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


def is_binary_column(series: pd.Series) -> bool:
    """True if ``series`` (after dropping NaN) takes only values in ``{0, 1}`` (or boolean dtype)."""
    if series.dtype == bool:
        return True
    values = pd.unique(series.dropna())
    return len(values) > 0 and set(np.asarray(values).tolist()).issubset({0, 1})


def boolean_pair_interactions(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    operators: Sequence[str] = ("and", "or", "xor"),
) -> pd.DataFrame:
    """Generate pairwise AND/OR/XOR columns for every pair of binary columns.

    Parameters
    ----------
    df
        Source frame.
    columns
        Binary/one-hot column names to combine; defaults to auto-detecting every column satisfying
        ``is_binary_column``.
    operators
        Subset of ``{"and", "or", "xor"}``.

    Returns
    -------
    pd.DataFrame
        One column per ``(pair, operator)`` combination, named ``"{col_a}__{op}__{col_b}"``, dtype ``int8``
        (0/1). ``len(columns) choose 2`` pairs x ``len(operators)`` columns total -- combinatorial, meant to
        be pruned by a downstream feature selector, not used wholesale.
    """
    if columns is None:
        columns = [c for c in df.columns if is_binary_column(df[c])]
    if len(columns) < 2:
        return pd.DataFrame(index=df.index)

    invalid_ops = set(operators) - {"and", "or", "xor"}
    if invalid_ops:
        raise ValueError(f"boolean_pair_interactions: unsupported operators {invalid_ops}, expected subset of {{'and', 'or', 'xor'}}")

    out: Dict[str, np.ndarray] = {}
    arrs = {c: df[c].to_numpy(dtype=np.int8) for c in columns}
    for col_a, col_b in combinations(columns, 2):
        a, b = arrs[col_a], arrs[col_b]
        if "and" in operators:
            out[f"{col_a}__and__{col_b}"] = (a & b).astype(np.int8)
        if "or" in operators:
            out[f"{col_a}__or__{col_b}"] = (a | b).astype(np.int8)
        if "xor" in operators:
            out[f"{col_a}__xor__{col_b}"] = (a ^ b).astype(np.int8)

    return pd.DataFrame(out, index=df.index)


__all__ = ["boolean_pair_interactions", "is_binary_column"]
