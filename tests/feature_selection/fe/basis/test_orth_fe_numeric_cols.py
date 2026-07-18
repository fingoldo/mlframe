"""Regression: the orthogonal/polynomial hybrid-FE column filter keeps only numeric (incl. bool) scalar columns.

Pre-fix, the adaptive-arity / three-gate / meta-scorer FE col lists (`_aa_cols` / `_tg_cols` / `_meta_cols`) included raw
categorical / string columns; the orthogonal FE then did `float(...)` on them -> `ValueError: could not convert string
to float: 'B'`, swallowed by MRMR ("continuing without ... columns"), silently dropping those FE passes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl import _orth_fe_numeric_cols


def test_excludes_string_and_cat_keeps_numeric_and_bool():
    """_orth_fe_numeric_cols keeps numeric and bool columns but drops raw string/categorical ones, avoiding the float('B') crash."""
    X = pd.DataFrame(
        {
            "num_f": np.arange(5, dtype=float),
            "num_i": np.arange(5),
            "flag": pd.array([True, False, True, False, True], dtype="bool"),
            "B": ["a", "b", "c", "a", "b"],  # string -> the crash source
            "cat": pd.Categorical(["x", "y", "x", "y", "x"]),  # category -> not numeric
        }
    )
    out = _orth_fe_numeric_cols(X, ["num_f", "num_i", "flag", "B", "cat", "absent"])
    assert out == ["num_f", "num_i", "flag"]


def test_skips_duplicate_named_column():
    """A duplicated column name (X[c] returns a 2-D DataFrame) is skipped as ambiguous rather than crashing."""
    X = pd.DataFrame(np.zeros((4, 3)), columns=["dup", "dup", "ok"])
    out = _orth_fe_numeric_cols(X, ["dup", "ok"])  # X['dup'] -> DataFrame (ndim 2) -> skip as ambiguous
    assert out == ["ok"]
