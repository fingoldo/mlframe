"""Shared colormaps + palettes for mlframe reporting.

Single source of truth so matplotlib + plotly renderers produce visually
consistent output across backends. matplotlib uses the colormap name
directly; plotly uses ``plotly.colors.sample_colorscale`` with a
matplotlib-compatible name.

Conventions:
- ``CALIBRATION``: ``RdYlBu`` (blue=high population, red=low) — used in
  the calibration scatter + bin-population histogram so both panels
  read against the same colorbar.
- ``CONFUSION``: ``RdBu_r`` (sequential, red diagonal = confusion good)
- ``HEATMAP_GENERIC``: ``viridis`` for non-confusion heatmaps
- ``BAR_PRIMARY``: ``"steelblue"`` (single-series bar default)
- ``LINE_PALETTE``: discrete categorical palette for overlaid line
  plots (per-class ROC, multi-line NDCG@k etc.)
"""

from __future__ import annotations

from typing import List, Tuple


CALIBRATION = "RdYlBu"
CONFUSION = "RdBu_r"
HEATMAP_GENERIC = "viridis"

BAR_PRIMARY = "steelblue"
PERFECT_FIT_LINE = "green"
NORMAL_OVERLAY = "red"
ZERO_LINE = "green"

# Discrete categorical palette (10 colors). Matches matplotlib tab10
# which plotly aliases via `qualitative.Plotly` -- visually similar
# but not identical; cross-backend renderings won't pixel-match here.
LINE_PALETTE: Tuple[str, ...] = (
    "#1f77b4",  # tab:blue
    "#ff7f0e",  # tab:orange
    "#2ca02c",  # tab:green
    "#d62728",  # tab:red
    "#9467bd",  # tab:purple
    "#8c564b",  # tab:brown
    "#e377c2",  # tab:pink
    "#7f7f7f",  # tab:gray
    "#bcbd22",  # tab:olive
    "#17becf",  # tab:cyan
)


def line_color(idx: int) -> str:
    """Cyclic line color for index ``idx`` (per-class line plots)."""
    return LINE_PALETTE[idx % len(LINE_PALETTE)]


__all__ = [
    "CALIBRATION", "CONFUSION", "HEATMAP_GENERIC",
    "BAR_PRIMARY", "PERFECT_FIT_LINE", "NORMAL_OVERLAY", "ZERO_LINE",
    "LINE_PALETTE", "line_color",
]
