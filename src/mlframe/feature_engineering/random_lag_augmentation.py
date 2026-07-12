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

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

Lag = Union[float, int, pd.Timedelta]


def _lag_to_raw(value: Lag, is_datetime: bool) -> float:
    """Convert a single ``Lag`` value to a raw float in ``cutoff_col``'s native arithmetic unit (ns for a
    datetime column, the value itself otherwise) -- shared conversion so histogram edges and the plain
    ``min_lag``/``max_lag`` bounds are interpreted identically."""
    return float(pd.Timedelta(value).value) if is_datetime else float(value)


def randomize_as_of_lag(
    as_of: pd.DataFrame,
    cutoff_col: str,
    max_lag: Lag,
    min_lag: Lag = 0,
    random_state: Optional[int] = None,
    lag_histogram_edges: Optional[Sequence[Lag]] = None,
    lag_histogram_counts: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Return a COPY of ``as_of`` with ``cutoff_col`` shifted earlier by a per-row random offset --
    simulating the true range of serving-time data staleness.

    By default the offset is drawn uniformly from ``[min_lag, max_lag]``. Real serving-lag distributions are
    often non-uniform (e.g. skewed toward short lags from a fast pipeline refresh, with an occasional long
    tail from retries/backfills) -- passing BOTH ``lag_histogram_edges`` and ``lag_histogram_counts`` (an
    opt-in caller-supplied empirical staleness histogram, e.g. from real production serving logs) instead
    draws each row's bin proportionally to the observed counts, then uniformly within that bin, so the
    per-row training lag distribution matches the true serving-lag shape rather than a flat uniform draw.

    Parameters
    ----------
    as_of
        Query frame with a cutoff column (as consumed by ``leakage_safe_aggregate``'s ``as_of`` parameter).
    cutoff_col
        Name of the cutoff column to shift.
    max_lag, min_lag
        Bounds of the per-row random staleness offset, in the SAME units as ``cutoff_col`` (a numeric offset
        for a numeric cutoff column, a ``pd.Timedelta`` for a datetime cutoff column -- mixing a numeric
        offset into a datetime column, or vice versa, raises via pandas' own arithmetic). Ignored when
        ``lag_histogram_edges``/``lag_histogram_counts`` are supplied.
    random_state
        Seed for the per-row offset sampling.
    lag_histogram_edges
        Opt-in: bin edges (length B+1, strictly increasing, same units as ``min_lag``/``max_lag``) of an
        empirical observed serving-latency histogram. Must be supplied together with
        ``lag_histogram_counts``; supplying only one raises.
    lag_histogram_counts
        Opt-in: observed frequency/weight per bin (length B, all non-negative, sum > 0) of the empirical
        serving-latency histogram -- bin ``i`` spans ``[lag_histogram_edges[i], lag_histogram_edges[i+1])``.

    Returns
    -------
    pd.DataFrame
        Copy of ``as_of`` with ``cutoff_col`` shifted; all other columns unchanged. Never mutates the input.
    """
    if cutoff_col not in as_of.columns:
        raise ValueError(f"randomize_as_of_lag: {cutoff_col!r} not in as_of.columns")
    if (lag_histogram_edges is None) != (lag_histogram_counts is None):
        raise ValueError("randomize_as_of_lag: lag_histogram_edges and lag_histogram_counts must be supplied together")

    rng = np.random.default_rng(random_state)
    n = len(as_of)
    out = as_of.copy()

    is_datetime = pd.api.types.is_datetime64_any_dtype(out[cutoff_col])

    if lag_histogram_edges is not None and lag_histogram_counts is not None:
        edges_arr = np.array([_lag_to_raw(edge, is_datetime) for edge in lag_histogram_edges], dtype=np.float64)
        counts_arr = np.asarray(lag_histogram_counts, dtype=np.float64)
        if len(edges_arr) != len(counts_arr) + 1:
            raise ValueError(f"randomize_as_of_lag: lag_histogram_edges must have len(lag_histogram_counts) + 1 entries, got {len(edges_arr)} and {len(counts_arr)}")
        if np.any(counts_arr < 0) or counts_arr.sum() <= 0:
            raise ValueError("randomize_as_of_lag: lag_histogram_counts must be non-negative with a positive sum")
        if np.any(np.diff(edges_arr) <= 0):
            raise ValueError("randomize_as_of_lag: lag_histogram_edges must be strictly increasing")

        bin_idx = rng.choice(len(counts_arr), size=n, p=counts_arr / counts_arr.sum())
        lo = edges_arr[bin_idx]
        hi = edges_arr[bin_idx + 1]
        within_bin = lo + rng.uniform(0.0, 1.0, size=n) * (hi - lo)
        offsets = pd.to_timedelta(within_bin.astype(np.int64)) if is_datetime else within_bin
    elif is_datetime:
        min_ns = pd.Timedelta(min_lag).value
        max_ns = pd.Timedelta(max_lag).value
        offsets = pd.to_timedelta(rng.uniform(min_ns, max_ns, size=n).astype(np.int64))
    else:
        offsets = rng.uniform(float(min_lag), float(max_lag), size=n)

    out[cutoff_col] = out[cutoff_col] - offsets
    return out


__all__ = ["randomize_as_of_lag"]
