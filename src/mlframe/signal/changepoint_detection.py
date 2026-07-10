"""Regime change-point detection wrapper around ``ruptures``, producing per-row regime segment labels.

Fitting one model over a time series whose underlying behavior has genuinely shifted (a change in operating
regime, market condition, or process parameters) blends pre- and post-shift dynamics into a single blurred
average. An LTFS-finhack team's fix: detect structural breaks with the ``ruptures`` library (PELT algorithm),
filter by minimum segment duration and effect-size significance, and use the resulting segment id both as a
feature and to drive per-regime modeling (independent baselines/models per detected steady state).

Performance note: uses the ``"l2"`` (mean-shift) cost model, not ``"rbf"`` -- rbf precomputes a full pairwise
kernel matrix (O(n^2) memory/time) before PELT even starts, which effectively hangs at real production series
lengths (measured: did not finish at n=50,000, ~2.5 billion kernel entries). ``"l2"`` avoids that blowup and
is the correct model for mean-level regime shifts (what this detector targets), but ``ruptures``' own l2 cost
function calls ``np.var()`` per candidate window rather than a cumulative-sum O(1) update, so it still costs
~29s at n=50,000 (measured, `bench_changepoint_detection.py`) -- fine at typical panel-timeseries lengths
(hundreds to low thousands of points per series) but downsample/bin longer series before detection.
"""
from __future__ import annotations

import numpy as np


def detect_regime_changepoints(
    y: np.ndarray,
    min_segment_length: int = 10,
    penalty: float = 3.0,
    min_effect_size: float = 0.5,
) -> dict:
    """Detect structural breaks in ``y`` via ruptures' PELT algorithm, filtered by duration and effect size.

    Parameters
    ----------
    y
        ``(n,)`` numeric series in chronological order.
    min_segment_length
        Minimum number of rows in a segment for ruptures' PELT search (``min_size``); also used post-hoc to
        drop candidate breakpoints that would create a shorter segment.
    penalty
        PELT's ``pen`` parameter -- higher values detect fewer, more conservative breakpoints.
    min_effect_size
        A detected breakpoint is kept only if the absolute difference between the segment means on either
        side, divided by the pooled standard deviation (Cohen's-d-style effect size), exceeds this threshold
        -- filters out statistically-detected but practically-negligible breaks.

    Returns
    -------
    dict
        ``regime_id`` ``(n,)`` int array (0-indexed, one id per detected steady-state segment),
        ``breakpoints`` (list of row indices where a new regime starts, after filtering),
        ``n_regimes`` (int).
    """
    import ruptures as rpt

    y = np.asarray(y, dtype=np.float64)
    n = y.shape[0]
    if n < 2 * min_segment_length:
        return {"regime_id": np.zeros(n, dtype=np.int64), "breakpoints": [], "n_regimes": 1}

    # "l2" (mean-shift cost) instead of "rbf" -- rbf precomputes a full pairwise kernel matrix (O(n^2)
    # memory/time) before PELT even starts, which effectively hangs at real production series lengths
    # (measured: did not finish at n=50,000, ~2.5 billion kernel entries). l2 is the correct model for
    # mean-level regime shifts and avoids that blowup; see the module docstring's performance note for its
    # own remaining (non-cumsum) per-candidate cost.
    algo = rpt.Pelt(model="l2", min_size=min_segment_length).fit(y.reshape(-1, 1))
    raw_breakpoints = algo.predict(pen=penalty)
    raw_breakpoints = [bp for bp in raw_breakpoints if bp < n]  # ruptures includes n as a terminal marker

    filtered_breakpoints = []
    prev_start = 0
    for bp in raw_breakpoints:
        next_start = raw_breakpoints[raw_breakpoints.index(bp) + 1] if raw_breakpoints.index(bp) + 1 < len(raw_breakpoints) else n
        if bp - prev_start < min_segment_length or next_start - bp < min_segment_length:
            continue
        left_segment = y[prev_start:bp]
        right_segment = y[bp:next_start]
        pooled_std = np.sqrt((np.var(left_segment, ddof=1) + np.var(right_segment, ddof=1)) / 2.0)
        effect_size = abs(np.mean(right_segment) - np.mean(left_segment)) / pooled_std if pooled_std > 0 else np.inf
        if effect_size >= min_effect_size:
            filtered_breakpoints.append(bp)
            prev_start = bp

    regime_id = np.zeros(n, dtype=np.int64)
    for i, bp in enumerate(filtered_breakpoints):
        regime_id[bp:] = i + 1

    return {"regime_id": regime_id, "breakpoints": filtered_breakpoints, "n_regimes": len(filtered_breakpoints) + 1}


__all__ = ["detect_regime_changepoints"]
