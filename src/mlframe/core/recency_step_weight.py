"""``recency_step_weight``: a two-level step-function sample_weight boosting the most recent rows.

Source: 9th_optiver-trading-at-the-close.md -- "xgb sample_weight 1.5 weight for latest 45 days data." Unlike
:func:`mlframe.core.recency_weights.recency_weights` (continuous poly/exp/power decay over the WHOLE history,
normalized to sum 1 -- built for weighted feature aggregation), a training ``sample_weight=`` array wants
absolute multipliers with no normalization, and the source's own scheme is a simple two-level step (flat
``base`` weight for older rows, flat ``boost`` weight for rows within the recent cutoff) rather than a smooth
decay -- a genuinely different shape, not a parameterization of the existing schemes.
"""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


def recency_step_weight(
    dates: Union[np.ndarray, pd.Series],
    cutoff_date: Union[np.datetime64, "pd.Timestamp", float, int],
    boost: float = 1.5,
    base: float = 1.0,
) -> np.ndarray:
    """Per-row ``sample_weight``: ``boost`` for rows with ``dates >= cutoff_date``, ``base`` otherwise.

    Parameters
    ----------
    dates
        ``(n,)`` per-row date/time (or any orderable recency key, e.g. an integer day index).
    cutoff_date
        Rows on or after this value get ``boost``; earlier rows get ``base``.
    boost
        Weight for recent rows (the source's own default: ``1.5``).
    base
        Weight for older rows. Default ``1.0``.

    Returns
    -------
    np.ndarray
        ``(n,)`` float64 sample weights, ready for ``sample_weight=`` in XGBoost/LightGBM/sklearn ``fit``.
    """
    dates_arr = np.asarray(dates)
    is_recent = dates_arr >= np.asarray(cutoff_date)
    # bench-attempt-rejected: a fused `base + is_recent * (boost - base)` looked like it should save a pass
    # vs np.where(...).astype(...), but measured SLOWER in isolated A/B (n=10M: ~98ms/call vs ~80ms/call) --
    # np.where's scalar-fill path is apparently better optimized than the bool->float multiply here. Kept
    # np.where; the whole call is proportional elementwise cost either way (~8ns/row), not worth chasing further.
    return np.where(is_recent, float(boost), float(base)).astype(np.float64)


__all__ = ["recency_step_weight"]
