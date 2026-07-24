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

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

CALIBRATION = "RdYlBu"
CONFUSION = "RdBu_r"
# CB-safe heatmap defaults. ``HEATMAP_CMAP`` (viridis) is perceptually uniform and colourblind-safe for
# unsigned magnitude heatmaps (confusion / PSI drift / weak-segment / co-occurrence / per-label threshold /
# 2-D PDP); ``DIVERGING_CMAP`` (RdBu_r) is the signed/centred-at-zero counterpart. ``auto_text_color`` picks
# the cell-text colour by perceived luminance, so the bright high-end of viridis stays readable.
HEATMAP_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"
# Generic-heatmap placeholder kept as the spec-field default; renderers resolve it to ``HEATMAP_CMAP`` so an
# un-overridden HeatmapPanelSpec renders CB-safe on both backends. Builders that want a specific map pass it.
HEATMAP_GENERIC = "Blues"

BAR_PRIMARY = "steelblue"
PERFECT_FIT_LINE = "green"
NORMAL_OVERLAY = "red"
ZERO_LINE = "green"

# Discrete categorical palette. First 10 are matplotlib tab10 (kept stable so existing snapshots don't shift); the next
# 10 extend to the full tab20 set so classes stop colliding before K=20. Plotly aliases tab10 via ``qualitative.Plotly``
# (visually similar, not identical) -- cross-backend renderings won't pixel-match here.
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
    "#aec7e8",  # tab20 light blue
    "#ffbb78",  # tab20 light orange
    "#98df8a",  # tab20 light green
    "#ff9896",  # tab20 light red
    "#c5b0d5",  # tab20 light purple
    "#c49c94",  # tab20 light brown
    "#f7b6d2",  # tab20 light pink
    "#c7c7c7",  # tab20 light gray
    "#dbdb8d",  # tab20 light olive
    "#9edae5",  # tab20 light cyan
)


def line_color(idx: int) -> str:
    """Cyclic line color for index ``idx`` (per-class line plots)."""
    return LINE_PALETTE[idx % len(LINE_PALETTE)]


def resolve_heatmap_cmap(colormap: Optional[str]) -> str:
    """Resolve a HeatmapPanelSpec colormap to the CB-safe default when it was left at the generic placeholder.

    An un-overridden ``HeatmapPanelSpec.colormap`` is ``HEATMAP_GENERIC``; mapping it to ``HEATMAP_CMAP`` here
    means both renderers default heatmaps to perceptually-uniform viridis while an explicit override is honoured.
    """
    if colormap is None or colormap == HEATMAP_GENERIC:
        return HEATMAP_CMAP
    return colormap


# Feature "friend graph" node classes (mlframe.feature_selection friend_graph):
# green = unique informative feature, red = suspected redundant sink/aggregator,
# yellow = middling. Shared here so both renderers and any external dashboard
# read the same legend colors. ``FRIEND_GRAPH_EDGE_CMAP`` shades edges by the
# mutual information they carry.
FRIEND_GRAPH_NODE_COLORS = {
    "green": "#2ca02c",  # tab:green - unique knowledge
    "red": "#d62728",  # tab:red - suspected sink / aggregator
    "yellow": "#e6a700",  # amber - middling
}
FRIEND_GRAPH_EDGE_CMAP = "plasma"


def friend_graph_node_color(klass: str) -> str:
    """Resolve a friend-graph node class ('green'/'red'/'yellow') to a hex color.

    Unknown classes fall back to gray so a mis-labelled node is still visible
    rather than crashing the render."""
    return FRIEND_GRAPH_NODE_COLORS.get(klass, "#7f7f7f")


def auto_text_color(value: float, colormap: str, vmin: float = 0.0, vmax: float = 1.0) -> str:
    """Pick ``"black"`` or ``"white"`` text for an overlay on a
    colormap-shaded background, by perceived-luminance threshold.

    Without this, naive ``white if value > 0.5 else black`` heuristics
    fail on light-end colormaps (viridis -> bright yellow at 0.85,
    RdYlBu -> pale yellow at 0.5, etc.) -- white text on a yellow
    cell is unreadable. We sample the colormap at the normalised value,
    compute the standard sRGB perceived luminance (Rec. 601), and
    return the color that maximises contrast.

    Falls back to ``"black"`` if the colormap can't be loaded
    (matplotlib import error, unknown name).
    """
    try:
        import matplotlib
        cm = matplotlib.colormaps[colormap]
        # Normalise value into [0, 1] for the colormap.
        if vmax > vmin:
            t = (float(value) - vmin) / (vmax - vmin)
        else:
            t = 0.5
        t = max(0.0, min(1.0, t))
        r, g, b, _ = cm(t)
        # Rec. 601 perceived luminance.
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return "black" if luminance > 0.5 else "white"
    except Exception as exc:
        logger.debug("auto_text_color: colormap lookup for %r failed, falling back to 'black': %s", colormap, exc)
        return "black"


__all__ = [
    "CALIBRATION", "CONFUSION", "HEATMAP_GENERIC", "HEATMAP_CMAP", "DIVERGING_CMAP",
    "BAR_PRIMARY", "PERFECT_FIT_LINE", "NORMAL_OVERLAY", "ZERO_LINE",
    "LINE_PALETTE", "line_color", "auto_text_color", "resolve_heatmap_cmap",
    "FRIEND_GRAPH_NODE_COLORS", "FRIEND_GRAPH_EDGE_CMAP", "friend_graph_node_color",
]
