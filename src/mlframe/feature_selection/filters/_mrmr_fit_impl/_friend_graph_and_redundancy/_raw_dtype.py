"""Dtype checks on raw input columns for the post-selection protection passes."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def raw_column_is_numeric(X: Any, name: str, data: np.ndarray, col_idx: int) -> bool:
    """True when the selected raw column ``name`` is numeric (bool counts as numeric).

    ``data`` is the categorize output: integer bin codes for EVERY column, string and categorical ones included, so its dtype says nothing
    about the raw column. Read the raw frame's dtype when it is available; only without a named raw frame does the code matrix decide.
    """
    if isinstance(X, pd.DataFrame) and name in X.columns:
        col = X[name]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        return bool(pd.api.types.is_numeric_dtype(col.dtype)) and not isinstance(col.dtype, pd.CategoricalDtype)
    if hasattr(X, "schema") and hasattr(X, "columns") and name in list(X.columns):
        return bool(X.schema[name].is_numeric())
    return bool(np.issubdtype(np.asarray(data[:, int(col_idx)]).dtype, np.number))
