"""Target-temporal-audit chart spec builder.

Renders the full temporal-audit diagnostic via the spec vocabulary:
- the kept-bins target-rate line,
- dropped/sparse bins as a faded marker series (so the operator sees
  what was filtered, not a silent gap),
- per-segment mean as a horizontal step overlay,
- Pelt change-points as vertical reference lines,
- ``x_is_time`` so renderers format the time axis (rotate/auto-fmt).

All series share the full (kept + dropped) timeline; the kept line and
the dropped markers carry NaN at the other's positions so each draws only
where it has data (matplotlib + plotly both skip NaN).
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from mlframe.reporting.spec import FigureSpec, LinePanelSpec


def _median_gap(x_axis: np.ndarray) -> Any:
    """Median inter-point gap of a sorted x-axis. Returns a timedelta for datetime x, a float otherwise.

    Used to size the change-point span half-width; falls back to a unit gap on a single-point axis.
    """
    if len(x_axis) < 2:
        arr = np.asarray(x_axis)
        return np.timedelta64(1, "D") if np.issubdtype(arr.dtype, np.datetime64) else 1.0
    diffs = np.diff(np.asarray(x_axis))
    return np.median(diffs)


def build_temporal_audit_spec(
    audit_result: Any,                # TemporalAuditResult instance
    *,
    figsize: Tuple[float, float] = (12.0, 4.5),
) -> FigureSpec:
    """Build a temporal-audit FigureSpec surfacing all audit structure.

    Reads from ``audit_result``:
    - ``bins``: List[TimeBin] with ``bin_start`` + ``target_rate`` + ``kept``
    - ``segments``: per-segment dicts with ``start_idx`` / ``end_idx``
      (into the kept-bin list) + ``mean_rate``
    - ``change_point_indices``: indices into the kept-bin list
    - ``target_name``, ``granularity``, ``target_type``, ``timestamp_col``
    """
    bins = list(getattr(audit_result, "bins", []) or [])
    kept = [b for b in bins if getattr(b, "kept", True)]
    dropped = [b for b in bins if not getattr(b, "kept", True)]

    target_name = getattr(audit_result, "target_name", "target")
    granularity = getattr(audit_result, "granularity", "")
    target_type_str = str(getattr(audit_result, "target_type", ""))
    timestamp_col = getattr(audit_result, "timestamp_col", "Time")
    segments = list(getattr(audit_result, "segments", []) or [])
    change_points = list(getattr(audit_result, "change_point_indices", []) or [])

    if not kept and not dropped:
        # Degenerate audit -- emit a 1-point flat line so renderers don't crash.
        line = LinePanelSpec(
            x=np.array([0.0]), y=np.array([0.0]),
            title=f"Target rate over time: {target_name} (no bins)",
            xlabel="Time", ylabel="mean(y)", line_styles=("-",),
        )
        return FigureSpec(suptitle="", panels=((line,),), figsize=figsize)

    # Full timeline (kept + dropped), sorted by bin_start so the x-axis is monotone.
    all_bins = sorted(bins, key=lambda b: b.bin_start)
    _starts = [b.bin_start for b in all_bins]
    # Coerce timestamp-like bin starts to datetime64[ns] so renderers format the time axis and span arithmetic
    # (cx +- half_w via timedelta) works; numeric / non-datetime starts pass through unchanged.
    if _starts and hasattr(_starts[0], "isoformat"):
        import pandas as _pd
        x_axis = _pd.DatetimeIndex(_starts).to_numpy()
    else:
        x_axis = np.asarray(_starts)
    pos = {id(b): i for i, b in enumerate(all_bins)}

    kept_y = np.full(len(all_bins), np.nan, dtype=np.float64)
    for b in kept:
        kept_y[pos[id(b)]] = float(b.target_rate)

    dropped_y = np.full(len(all_bins), np.nan, dtype=np.float64)
    for b in dropped:
        dropped_y[pos[id(b)]] = float(b.target_rate)

    # Per-segment mean as a step series over the full timeline. Each kept bin inherits its segment's mean_rate; the
    # value holds flat across the segment and steps at boundaries -- the visual "is the rate stable within a regime".
    seg_step = np.full(len(all_bins), np.nan, dtype=np.float64)
    for s in segments:
        start_idx = int(s.get("start_idx", 0))
        end_idx = int(s.get("end_idx", 0))  # exclusive into kept[]
        mean_rate = float(s.get("mean_rate", np.nan))
        for ki in range(start_idx, min(end_idx, len(kept))):
            seg_step[pos[id(kept[ki])]] = mean_rate

    # Change-points (indices into kept[]) -> thin shaded vertical spans at the corresponding timeline positions.
    # vspans (add_vrect / axvspan) rather than vlines: plotly's add_vline does annotation-position arithmetic on the
    # x value, which raises on a datetime axis (Timestamp has no integer add); add_vrect handles datetime on both
    # backends. The span half-width is a small fraction of the median inter-bin gap so it reads as a crisp boundary.
    half_w = _median_gap(x_axis) * 0.15
    vspans: List[Tuple[Any, Any, str, float]] = []
    for ci in change_points:
        if 0 <= ci < len(kept):
            cx = x_axis[pos[id(kept[ci])]]
            vspans.append((cx - half_w, cx + half_w, "red", 0.5))

    ylabel = "P(y=1)" if "binary" in target_type_str.lower() else "mean(y)"
    n_seg = len(segments)
    title = (
        f"Target rate over time: {target_name} "
        f"({target_type_str}, {granularity}-binned; {n_seg} segment(s), "
        f"{len(change_points)} change-point(s), {len(dropped)} sparse bin(s))"
    )

    line = LinePanelSpec(
        x=x_axis,
        y=(kept_y, seg_step, dropped_y),
        series_labels=(target_name, "segment mean", f"sparse (filtered, n={len(dropped)})"),
        title=title,
        xlabel=f"{timestamp_col} ({granularity})",
        ylabel=ylabel,
        line_styles=("lines+markers", "-", "markers"),
        colors=("steelblue", "orange", "gray"),
        vspans=tuple(vspans) or None,
        x_is_time=True,
    )

    return FigureSpec(
        suptitle="",
        panels=((line,),),
        figsize=figsize,
    )


__all__ = ["build_temporal_audit_spec"]
