"""Target-temporal-audit chart spec builder.

Single LinePanelSpec showing P(y=1) (binary) or mean(y) (regression)
over time at the auto-picked granularity, with optional segment
markers (Pelt change-points).
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from mlframe.reporting.spec import FigureSpec, LinePanelSpec


def build_temporal_audit_spec(
    audit_result: Any,                # TemporalAuditResult instance
    *,
    figsize: Tuple[float, float] = (12.0, 4.0),
) -> FigureSpec:
    """Build a temporal-audit FigureSpec.

    Reads from ``audit_result``:
    - ``bins``: List[TimeBin] with ``bin_start`` + ``target_rate`` + ``kept``
    - ``target_name``, ``granularity``, ``segments`` (optional)

    Per-segment shaded background is left to follow-up (renderers don't yet
    have a generic ``vrect`` / ``axvspan`` spec field). For now, segment
    boundaries are folded into the title.
    """
    # Filter out sparse-bin entries (kept=False) the same way the
    # legacy plot_target_over_time path does -- only kept bins are drawn.
    kept = [b for b in getattr(audit_result, "bins", []) if getattr(b, "kept", True)]
    if not kept:
        # Degenerate audit -- emit a 1-point flat line so renderers don't crash.
        time_bins = np.array([0.0])
        rates = np.array([0.0])
    else:
        time_bins = np.asarray([b.bin_start for b in kept])
        rates = np.asarray([float(b.target_rate) for b in kept], dtype=np.float64)

    seg_text = ""
    n_segments = len(getattr(audit_result, "segments", []) or [])
    if n_segments > 1:
        seg_text = f" ({n_segments} segments detected)"

    target_name = getattr(audit_result, "target_name", "target")
    granularity = getattr(audit_result, "granularity", "")
    target_type_str = getattr(audit_result, "target_type", "")
    title = (
        f"Target rate over time: {target_name} "
        f"({target_type_str}, {granularity}-binned){seg_text}"
    )

    line = LinePanelSpec(
        x=time_bins,
        y=rates,
        title=title,
        xlabel="Time",
        ylabel="P(y=1)" if "binary" in str(target_type_str).lower() else "mean(y)",
        line_styles=("-",),
    )

    return FigureSpec(
        suptitle="",
        panels=((line,),),
        figsize=figsize,
    )


__all__ = ["build_temporal_audit_spec"]
