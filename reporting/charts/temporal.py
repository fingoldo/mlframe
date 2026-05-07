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
    - ``time_bins``: array of bin centers (datetime64 or numeric)
    - ``rates``: array of per-bin P(y=1) or mean(y)
    - ``target_name``, ``granularity``, ``segments`` (optional)

    Per-segment shaded background is left to PR2 (renderers don't yet
    have a generic ``vrect`` / ``axvspan`` spec field). For now, segment
    boundaries are folded into the title.
    """
    time_bins = np.asarray(audit_result.time_bins)
    rates = np.asarray(audit_result.rates, dtype=np.float64)

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
