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
# 2026-05-09: was ``"viridis"``. Viridis maps high values to bright
# yellow which (a) clashes visually with the tab10 categorical
# palette used in adjacent bar / line panels of the same figure and
# (b) makes white-text overlays on diagonal cells invisible. ``Blues``
# is single-hue (consistent with tab10[0] for cross-panel coherence)
# and its high-end stays dark enough for white text. ``auto_text_color``
# below handles the contrast computation regardless of colormap.
HEATMAP_GENERIC = "Blues"

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


def auto_text_color(value: float, colormap: str,
                    vmin: float = 0.0, vmax: float = 1.0) -> str:
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
    except Exception:
        return "black"


__all__ = [
    "CALIBRATION", "CONFUSION", "HEATMAP_GENERIC",
    "BAR_PRIMARY", "PERFECT_FIT_LINE", "NORMAL_OVERLAY", "ZERO_LINE",
    "LINE_PALETTE", "line_color", "auto_text_color",
]
