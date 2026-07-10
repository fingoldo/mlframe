"""``rolling_target_correlation_tracker``: dynamic, time-varying "most-correlated-with-target" feature.

Source: Ubiquant Market Prediction 2nd place -- "average by time_id for the most correlated feature with the
target on the latest 1000 time_id with more than 31 observations." Feature SELECTION becomes itself a
rolling-window operation: which feature is most predictive can drift over time (a regime shift, a market
condition change), so recomputing "the current best single feature" on a trailing window and emitting its
own value as a dynamic column captures that drift, rather than a static one-time correlation computed once
over the whole training set and then frozen.

Leakage discipline: the correlation used to SELECT the best feature at row ``i`` is computed over a
STRICTLY PAST window (rows ``i-window .. i-1``, via a ``.shift(1)`` before the rolling correlation) -- row
``i``'s own target is never used to decide which feature row ``i`` should read from. The VALUE gathered for
row ``i`` is that (already-selected) feature's own current value, legitimately available at prediction time.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def rolling_target_correlation_tracker(
    X: pd.DataFrame,
    y: np.ndarray,
    window: int,
    min_periods: Optional[int] = None,
    feature_columns: Optional[Sequence[str]] = None,
    column_prefix: str = "dyn_target_corr",
) -> pd.DataFrame:
    """For each row, find the feature with the highest trailing-window |correlation| to the target (using
    only strictly-past rows to select), and emit that feature's own current value as a dynamic column.

    Parameters
    ----------
    X
        Feature frame, ROW-ORDERED (e.g. by time); row ``i``'s neighbors ``i-window..i-1`` are its trailing
        window.
    y
        ``(n,)`` target, same row order as ``X``.
    window
        Trailing window size (in rows) for the rolling correlation.
    min_periods
        Minimum trailing observations before a correlation is computed (default: ``max(10, window // 10)``).
    feature_columns
        Candidate columns (default: every numeric column of ``X``).
    column_prefix
        Output column-name prefix.

    Returns
    -------
    pd.DataFrame
        Three columns: ``{prefix}_value`` (the selected feature's own current value), ``{prefix}_corr`` (that
        feature's trailing |correlation| that earned the selection), ``{prefix}_feature`` (its name, for
        diagnostics) -- all NaN for rows before ``min_periods`` trailing observations are available.
    """
    cols = list(feature_columns) if feature_columns is not None else list(X.select_dtypes(include=[np.number]).columns)
    if not cols:
        raise ValueError("rolling_target_correlation_tracker: no numeric feature_columns found/given.")
    min_periods = min_periods if min_periods is not None else max(10, window // 10)

    y_series = pd.Series(np.asarray(y, dtype=np.float64), index=X.index).shift(1)
    corr = pd.DataFrame({c: X[c].shift(1).rolling(window, min_periods=min_periods).corr(y_series) for c in cols}, index=X.index)
    abs_corr = corr.abs().to_numpy()

    valid_mask = np.isfinite(abs_corr).any(axis=1)
    abs_corr_filled = np.where(np.isfinite(abs_corr), abs_corr, -np.inf)
    best_idx = abs_corr_filled.argmax(axis=1)

    n = X.shape[0]
    row_pos = np.arange(n)
    X_arr = X[cols].to_numpy(dtype=np.float64)
    dynamic_value = np.where(valid_mask, X_arr[row_pos, best_idx], np.nan)
    dynamic_corr = np.where(valid_mask, abs_corr[row_pos, best_idx], np.nan)
    col_names = np.asarray(cols, dtype=object)
    dynamic_feature = np.where(valid_mask, col_names[best_idx], np.array(None, dtype=object))

    return pd.DataFrame(
        {f"{column_prefix}_value": dynamic_value, f"{column_prefix}_corr": dynamic_corr, f"{column_prefix}_feature": dynamic_feature},
        index=X.index,
    )


__all__ = ["rolling_target_correlation_tracker"]
