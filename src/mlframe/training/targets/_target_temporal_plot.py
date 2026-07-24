"""``plot_target_over_time`` -- temporal-audit visualisation.

Split out from ``training/target_temporal_audit.py`` to keep that file below the 1k-line monolith
threshold. Behaviour preserved bit-for-bit; the function is re-exported from ``target_temporal_audit``
so existing imports continue to work.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from .target_temporal_audit import TemporalAuditResult


try:
    import polars as pl
except ImportError:
    pl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def plot_target_over_time(
    result: TemporalAuditResult,
    *,
    save_path: str | None = None,
    show: bool = False,
    figsize: tuple[float, float] = (12, 4.5),
    plot_outputs: str | None = None,
    base_path: str | None = None,
) -> Any | None:
    """Render the time-series plot of target rate over bins.

    Mirrors the user's polars-rendered chart (year-month index, line
    plot of target rate). Adds:
    - dropped sparse bins shown as faded markers
    - vertical lines at change-point boundaries
    - per-segment mean as a horizontal step

    Parameters
    ----------
    result : TemporalAuditResult
    save_path : str, optional
        If provided, save the figure here and return the path (closes
        the figure to free memory).
    show : bool
        Call ``plt.show()`` instead of saving.
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure or None
        Returns the Figure when neither save_path nor show is set.
    """
    # Opt-in DSL render path. When ``plot_outputs`` +
    # ``base_path`` are supplied, route through the spec pipeline
    # (matplotlib + plotly via the same DSL). Default behaviour
    # preserved -- legacy callers see no change.
    if plot_outputs and base_path:
        from mlframe.reporting.charts.temporal import build_temporal_audit_spec
        from mlframe.reporting.output import parse_plot_output_dsl
        from mlframe.reporting.renderers import render_and_save
        spec = build_temporal_audit_spec(result, figsize=figsize)
        render_and_save(spec, parse_plot_output_dsl(plot_outputs), base_path)
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping target-over-time plot.")
        return None

    if not result.bins:
        return None

    kept = [b for b in result.bins if b.kept]
    dropped = [b for b in result.bins if not b.kept]

    fig, ax = plt.subplots(figsize=figsize)
    if kept:
        ax.plot(
            [b.bin_start for b in kept],
            [b.target_rate for b in kept],
            marker="o", linestyle="-", linewidth=1.2, markersize=4,
            label=result.target_name,
        )
    if dropped:
        ax.plot(
            [b.bin_start for b in dropped],
            [b.target_rate for b in dropped],
            marker="x", linestyle=":", linewidth=0.6, markersize=4,
            color="gray", alpha=0.4,
            label=f"sparse (filtered, n={len(dropped)})",
        )

    # Change-point lines + per-segment mean.
    for s in result.segments:
        if s["start_idx"] > 0 and s["start_idx"] < len(kept):
            ax.axvline(
                kept[s["start_idx"]].bin_start,
                color="red", linestyle="--", linewidth=0.8, alpha=0.4,
            )
        if s["start_idx"] < len(kept) and s["end_idx"] - 1 < len(kept):
            x_start = kept[s["start_idx"]].bin_start
            x_end = kept[min(s["end_idx"] - 1, len(kept) - 1)].bin_start
            ax.hlines(
                s["mean_rate"], x_start, x_end,
                color="orange", linestyle="-", linewidth=2.0, alpha=0.6,
            )

    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(f"{result.timestamp_col} ({result.granularity})")
    ax.set_ylabel("target rate")
    ax.set_title(f"target_temporal_audit: {result.target_name} " f"({result.granularity}-binned, {len(result.segments)} segments)")
    ax.legend(loc="best", framealpha=0.7)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        result.plot_path = save_path
        return None
    if show:
        plt.show()
        return None
    return fig
