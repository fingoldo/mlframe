"""``recency_step_weight``: a two-level step-function sample_weight boosting the most recent rows.

Source: 9th_optiver-trading-at-the-close.md -- "xgb sample_weight 1.5 weight for latest 45 days data." Unlike
:func:`mlframe.core.recency_weights.recency_weights` (continuous poly/exp/power decay over the WHOLE history,
normalized to sum 1 -- built for weighted feature aggregation), a training ``sample_weight=`` array wants
absolute multipliers with no normalization, and the source's own scheme is a simple two-level step (flat
``base`` weight for older rows, flat ``boost`` weight for rows within the recent cutoff) rather than a smooth
decay -- a genuinely different shape, not a parameterization of the existing schemes.

Extension: real regime drift is rarely a single sharp jump -- relevance often strengthens gradually as rows
approach "now". The single hard cutoff only captures a step function; ``tiers`` adds an N-level step ladder
(multiple cutoffs, each progressively closer to "now" carrying a higher weight) and ``smooth_window`` adds a
continuous linear ramp from ``base`` to ``boost`` around ``cutoff_date``, for cases where even an N-level ladder
is too coarse. Both are opt-in and mutually exclusive with each other; omitting both reproduces the original
two-level step bit-for-bit.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

DateLike = Union[np.datetime64, "pd.Timestamp", float, int]


def recency_step_weight(
    dates: Union[np.ndarray, pd.Series],
    cutoff_date: DateLike,
    boost: float = 1.5,
    base: float = 1.0,
    tiers: Optional[Sequence[Tuple[DateLike, float]]] = None,
    smooth_window: Optional[float] = None,
) -> np.ndarray:
    """Per-row ``sample_weight``: ``boost`` for rows with ``dates >= cutoff_date``, ``base`` otherwise.

    Parameters
    ----------
    dates
        ``(n,)`` per-row date/time (or any orderable recency key, e.g. an integer day index).
    cutoff_date
        Rows on or after this value get ``boost``; earlier rows get ``base``. Also the ramp's right edge
        when ``smooth_window`` is given.
    boost
        Weight for recent rows (the source's own default: ``1.5``).
    base
        Weight for older rows. Default ``1.0``.
    tiers
        Opt-in N-level step ladder: an ``(cutoff, weight)`` sequence, e.g. ``[(d1, 1.2), (d2, 1.8), (d3, 2.5)]``
        with ``d1 < d2 < d3``. Each row gets the weight of the highest-cutoff tier it meets or exceeds
        (``dates >= cutoff``); rows below every tier's cutoff get ``base``. ``boost``/``cutoff_date`` are ignored
        when this is set. Mutually exclusive with ``smooth_window``.
    smooth_window
        Opt-in continuous mode: instead of a hard step at ``cutoff_date``, linearly ramp the weight from
        ``base`` to ``boost`` over ``[cutoff_date - smooth_window, cutoff_date]`` (rows at or after
        ``cutoff_date`` get the full ``boost``, rows at or before the window start get ``base``). Requires
        numeric/timedelta-subtractable ``dates``. Mutually exclusive with ``tiers``.

    Returns
    -------
    np.ndarray
        ``(n,)`` float64 sample weights, ready for ``sample_weight=`` in XGBoost/LightGBM/sklearn ``fit``.
    """
    if tiers is not None and smooth_window is not None:
        raise ValueError("tiers and smooth_window are mutually exclusive")

    dates_arr = np.asarray(dates)

    if tiers is not None:
        # sort ascending by cutoff so later (closer-to-now) tiers overwrite earlier ones for rows that
        # qualify for multiple -- each row ends up with its highest-cutoff-met tier's weight.
        sorted_tiers = sorted(tiers, key=lambda t: t[0])
        weights = np.full(dates_arr.shape, float(base), dtype=np.float64)
        for tier_cutoff, tier_weight in sorted_tiers:
            weights = np.where(dates_arr >= np.asarray(tier_cutoff), float(tier_weight), weights)
        return weights

    if smooth_window is not None:
        if smooth_window <= 0:
            raise ValueError("smooth_window must be > 0")
        cutoff_arr = np.asarray(cutoff_date)
        # position within the ramp, clipped to [0, 1]: 0 at (cutoff - smooth_window) or earlier, 1 at cutoff
        # or later. Subtraction requires numeric or timedelta-compatible dtypes (datetime64 dates - a scalar
        # window float would raise; callers on datetime64 should pass window as an np.timedelta64).
        progress = (dates_arr - (cutoff_arr - smooth_window)) / smooth_window
        progress = np.clip(progress, 0.0, 1.0)
        return (float(base) + progress * (float(boost) - float(base))).astype(np.float64)

    is_recent = dates_arr >= np.asarray(cutoff_date)
    # bench-attempt-rejected: a fused `base + is_recent * (boost - base)` looked like it should save a pass
    # vs np.where(...).astype(...), but measured SLOWER in isolated A/B (n=10M: ~98ms/call vs ~80ms/call) --
    # np.where's scalar-fill path is apparently better optimized than the bool->float multiply here. Kept
    # np.where; the whole call is proportional elementwise cost either way (~8ns/row), not worth chasing further.
    return np.where(is_recent, float(boost), float(base)).astype(np.float64)


__all__ = ["recency_step_weight"]
