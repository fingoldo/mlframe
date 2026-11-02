"""Append new columns to a pandas frame without copying the columns it already has."""

from __future__ import annotations

import warnings

import pandas as pd


def append_columns(X: pd.DataFrame, new_cols: pd.DataFrame) -> pd.DataFrame:
    """``X`` with the columns of ``new_cols`` added, sharing the existing column buffers with ``X``.

    ``pd.concat([X, new_cols], axis=1)`` copies every block of ``X`` before pandas 3 (a 40 GB copy to add five columns to a 1e8 x 100 float32 frame). A
    shallow copy plus whole-column assignment leaves ``X`` untouched and shares its buffers; ``new_cols`` must share ``X``'s index (the callers build it with
    ``index=X.index``).
    """
    if new_cols.shape[1] == 0:
        return X
    out = X.copy(deep=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)  # many appended columns fragment the frame; the caller consolidates when it needs to
        for name in new_cols.columns:
            out[name] = new_cols[name]
    return out
