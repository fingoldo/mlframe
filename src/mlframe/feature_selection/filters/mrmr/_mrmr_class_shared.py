"""Tiny leaf module for MRMR helpers shared between ``_mrmr_class.py`` and its
``_mrmr_class_fit_helpers.py`` mixin. Both modules need ``_mrmr_y_columns``; putting it here
(rather than in either of the two) breaks the mutual import cycle between them (mixin classes
import each other for composition, and _mrmr_class_fit_helpers previously imported this function
straight from _mrmr_class, which itself imports the mixin).
"""
from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import pandas as pd


def _mrmr_y_columns(y: Any) -> Iterator[tuple[str, np.ndarray]]:
    """Yield (label, y_column_1d) for each output column of a 2D y (pandas / polars DataFrame, or 2D ndarray)."""
    if isinstance(y, pd.DataFrame):
        for col in y.columns:
            yield str(col), y[col].to_numpy()
        return
    if str(type(y).__module__).startswith("polars") and type(y).__name__ == "DataFrame":
        for col in y.columns:
            yield str(col), y[col].to_numpy()
        return
    arr = np.asarray(y)
    for k in range(arr.shape[1]):
        yield f"y{k}", arr[:, k]
