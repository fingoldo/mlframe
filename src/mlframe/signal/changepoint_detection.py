"""Regime change-point detection wrapper, producing per-row regime segment labels.

Fitting one model over a time series whose underlying behavior has genuinely shifted (a change in operating
regime, market condition, or process parameters) blends pre- and post-shift dynamics into a single blurred
average. An LTFS-finhack team's fix: detect structural breaks (PELT algorithm), filter by minimum segment
duration and effect-size significance, and use the resulting segment id both as a feature and to drive
per-regime modeling (independent baselines/models per detected steady state).

Backend note: the default backend is an in-house njit PELT specialized for the l2 (mean-shift) cost
(``mlframe.signal._pelt_l2_njit.pelt_l2``), not the ``ruptures`` library. Two reasons: (1) ``ruptures.Pelt``
with its default ``model="rbf"`` precomputes a full pairwise kernel matrix (O(n^2) memory/time) before PELT
even starts, which effectively hangs at real production series lengths (measured: did not finish at
n=50,000, ~2.5 billion kernel entries); ``"l2"`` (mean-shift) is the theoretically correct cost for what this
detector targets anyway. (2) ``ruptures``' own l2 cost function recomputes ``np.var()`` over each candidate
window from scratch (no cumsum reuse), so switching to it alone still cost ~29s at n=50,000. Segment
sum-of-squared-deviations is additive under prefix sums, so PELT's candidate-cost evaluation is O(1) with
cumsum instead -- the whole PELT loop was reimplemented as one njit kernel, verified breakpoint-identical to
``ruptures.Pelt(model="l2", jump=1)`` (ruptures' own default ``jump=5`` subsamples candidate points and is
therefore *less* exact than this kernel, not a stricter reference), and measured ~65x faster end-to-end at n=5,000 (7ms vs. 948ms) and reduces n=50,000 from ~29s (ruptures l2) /
effectively never (ruptures rbf) down to ~447ms (`_benchmarks/bench_changepoint_detection.py`). ``ruptures`` remains available via ``backend="ruptures"`` for
cross-validation or non-l2 cost models it supports that this in-house kernel does not.
"""
from __future__ import annotations

import numpy as np


def detect_regime_changepoints(
    y: np.ndarray,
    min_segment_length: int = 10,
    penalty: float = 3.0,
    min_effect_size: float = 0.5,
    backend: str = "njit",
    return_segment_stats: bool = False,
) -> dict:
    """Detect structural breaks in ``y`` via PELT, filtered by duration and effect size.

    Parameters
    ----------
    y
        ``(n,)`` numeric series in chronological order.
    min_segment_length
        Minimum number of rows in a segment for PELT's search; also used post-hoc to drop candidate
        breakpoints that would create a shorter segment.
    penalty
        PELT's ``pen`` parameter -- higher values detect fewer, more conservative breakpoints.
    min_effect_size
        A detected breakpoint is kept only if the absolute difference between the segment means on either
        side, divided by the pooled standard deviation (Cohen's-d-style effect size), exceeds this threshold
        -- filters out statistically-detected but practically-negligible breaks.
    backend
        ``"njit"`` (default) uses the in-house cumsum-accelerated PELT (l2 cost only, ~100x faster). Pass
        ``"ruptures"`` to use the ``ruptures`` library's l2-cost PELT instead (useful for cross-checking).
    return_segment_stats
        If ``True`` (default ``False``, opt-in, no effect on the other returned keys), also returns
        ``segment_stats``: a list of ``{"start", "end", "count", "mean", "std"}`` dicts, one per detected
        regime in order, so a caller building regime-conditional features doesn't have to manually re-slice
        ``y`` by ``breakpoints``. ``std`` uses ``ddof=1`` (matches the effect-size computation above) and is
        ``0.0`` for single-row segments.

    Returns
    -------
    dict
        ``regime_id`` ``(n,)`` int array (0-indexed, one id per detected steady-state segment),
        ``breakpoints`` (list of row indices where a new regime starts, after filtering),
        ``n_regimes`` (int), and ``segment_stats`` (only when ``return_segment_stats=True``).
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[0]
    if n < 2 * min_segment_length:
        result: dict = {"regime_id": np.zeros(n, dtype=np.int64), "breakpoints": [], "n_regimes": 1}
        if return_segment_stats:
            result["segment_stats"] = [_segment_stats(y, 0, n)] if n > 0 else []
        return result

    if backend == "njit":
        from mlframe.signal._pelt_l2_njit import pelt_l2

        raw_breakpoints = pelt_l2(y, min_segment_length, penalty)
    elif backend == "ruptures":
        import ruptures as rpt

        algo = rpt.Pelt(model="l2", min_size=min_segment_length).fit(y.reshape(-1, 1))
        raw_breakpoints = algo.predict(pen=penalty)
        raw_breakpoints = [bp for bp in raw_breakpoints if bp < n]  # ruptures includes n as a terminal marker
    else:
        raise ValueError(f"Unknown backend: {backend!r}, expected 'njit' or 'ruptures'.")

    filtered_breakpoints = []
    prev_start = 0
    for i, bp in enumerate(raw_breakpoints):
        next_start = raw_breakpoints[i + 1] if i + 1 < len(raw_breakpoints) else n
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

    result = {"regime_id": regime_id, "breakpoints": filtered_breakpoints, "n_regimes": len(filtered_breakpoints) + 1}
    if return_segment_stats:
        bounds = [0] + filtered_breakpoints + [n]
        result["segment_stats"] = [_segment_stats(y, bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    return result


def _segment_stats(y: np.ndarray, start: int, end: int) -> dict:
    """Mean/std/count summary of ``y[start:end]``; ``std`` uses ``ddof=1``, ``0.0`` for a single-row segment."""
    segment = y[start:end]
    return {
        "start": start,
        "end": end,
        "count": int(segment.shape[0]),
        "mean": float(np.mean(segment)),
        "std": float(np.std(segment, ddof=1)) if segment.shape[0] > 1 else 0.0,
    }


__all__ = ["detect_regime_changepoints"]
