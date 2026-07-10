"""``randomize_as_of_lag``: per-row random staleness offset for as-of cutoff-based feature training.

Source: Power Laws Forecasting 1st place -- "for each training sample, a random lag (1..forecast_period_len)
is chosen before computing aggregates, simulating the real test-time situation where the most recent data
isn't available... validation-set features are recomputed with a '_noval' suffix using sequential (not
random) lags matching the true test-time lag structure."

Train/serve freshness mismatch: a naive as-of feature pipeline computes every TRAINING row's history
aggregates up to the freshest cutoff available at label time, but at TRUE inference time the most recent
data is typically stale by some amount (a data pipeline refresh delay, a forecast horizon, ...) -- so the
model is trained on fresher inputs than it will ever see served, and any distributional dependence on
"how fresh is my most recent history point" (rolling-window value distributions shift with staleness, not
just their information content) doesn't get learned. This shifts each TRAINING row's as-of cutoff earlier by
a per-row RANDOM staleness offset (sampled independently per row, matching the true range of serving-time
staleness) before its history aggregates are computed -- VALIDATION/TEST rows keep their true, sequential
(non-randomized) cutoff, matching the real inference-time lag structure exactly.

Hooks into the existing leakage-safe cutoff-driven aggregate builders (:func:`as_of_aggregate.
leakage_safe_aggregate`, :func:`multi_window_aggregate.multi_window_aggregate`) -- this function only
produces the shifted ``as_of`` query frame; the caller passes it to whichever aggregate builder they already
use, unchanged.
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

Lag = Union[float, int, pd.Timedelta]


def randomize_as_of_lag(
    as_of: pd.DataFrame,
    cutoff_col: str,
    max_lag: Lag,
    min_lag: Lag = 0,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Return a COPY of ``as_of`` with ``cutoff_col`` shifted earlier by a per-row random offset drawn
    uniformly from ``[min_lag, max_lag]`` -- simulating the true range of serving-time data staleness.

    Parameters
    ----------
    as_of
        Query frame with a cutoff column (as consumed by ``leakage_safe_aggregate``'s ``as_of`` parameter).
    cutoff_col
        Name of the cutoff column to shift.
    max_lag, min_lag
        Bounds of the per-row random staleness offset, in the SAME units as ``cutoff_col`` (a numeric offset
        for a numeric cutoff column, a ``pd.Timedelta`` for a datetime cutoff column -- mixing a numeric
        offset into a datetime column, or vice versa, raises via pandas' own arithmetic).
    random_state
        Seed for the per-row offset sampling.

    Returns
    -------
    pd.DataFrame
        Copy of ``as_of`` with ``cutoff_col`` shifted; all other columns unchanged. Never mutates the input.
    """
    if cutoff_col not in as_of.columns:
        raise ValueError(f"randomize_as_of_lag: {cutoff_col!r} not in as_of.columns")

    rng = np.random.default_rng(random_state)
    n = len(as_of)
    out = as_of.copy()

    is_datetime = pd.api.types.is_datetime64_any_dtype(out[cutoff_col])
    if is_datetime:
        min_ns = pd.Timedelta(min_lag).value
        max_ns = pd.Timedelta(max_lag).value
        offsets = pd.to_timedelta(rng.uniform(min_ns, max_ns, size=n).astype(np.int64))
    else:
        offsets = rng.uniform(float(min_lag), float(max_lag), size=n)

    out[cutoff_col] = out[cutoff_col] - offsets
    return out


__all__ = ["randomize_as_of_lag"]
