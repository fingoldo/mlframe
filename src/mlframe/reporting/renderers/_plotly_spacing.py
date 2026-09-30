"""Pixel-based subplot spacing for the plotly renderer's grid.

``make_subplots`` takes ``vertical_spacing`` as a FRACTION of the plotting region's height, and builders size a grid at
a fixed height per row (``figsize_for_grid``). A fixed fraction therefore grows the gap with the row count while the
furniture it has to hold (tick labels, x-axis title, the next row's panel title) stays the same size, and every panel
of a tall grid shrinks. The gaps are sized here in pixels from what has to fit in them, then converted to fractions.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, ConfusionMarginsPanelSpec, HeatmapPanelSpec, LinePanelSpec, NetworkPanelSpec, ViolinPanelSpec,
)

from ._shared_helpers import PANEL_TITLE_FONTSIZE, _BAR_LABEL_MAXLEN, measured_text_width_pt

# plotly's default tick-label and axis-title font sizes under the stock template.
_TICK_FONT_PX = 12
_AXIS_TITLE_FONT_PX = 14
# Tick label line plus the tick-to-label pad.
_TICK_BAND_PX = 20
# Numeric y tick labels ("0.8", "1e+03") plus the label-to-axis pad.
_Y_TICK_BAND_PX = 36
# One x-axis title line plus its standoff from the tick labels.
_AXIS_TITLE_BAND_PX = 26
# Vertical room per wrapped line of a panel title (plotly stamps it just above the subplot domain).
_TITLE_LINE_PX = PANEL_TITLE_FONTSIZE + 4
# Clearance between the row above's furniture and the next row's title.
_GAP_PAD_PX = 12
# Violin labels are truncated to this many chars by the renderer before being drawn at -30 degrees.
_VIOLIN_LABEL_MAXLEN = 20
# The opt-in rangeslider strip sits between the axis and its tick labels.
_RANGESLIDER_PX = 60.0
# A plotly date tick is at most "YYYY-MM-DD HH:MM" wide.
_DATE_TICK_SAMPLE = "2024-01-01 00:00"


def _rotated_band_px(labels: Iterable[Any], angle_deg: float, maxlen: int) -> float:
    """Vertical extent of the widest tick label drawn at ``angle_deg``, in px."""
    widest = max((measured_text_width_pt(str(lab)[:maxlen], _TICK_FONT_PX) for lab in labels), default=0.0)
    theta = math.radians(min(abs(float(angle_deg)), 90.0))
    return widest * math.sin(theta) + _TICK_FONT_PX * 1.25 * math.cos(theta) + 6.0


def _tick_band_px(panel: Any) -> float:
    """Room the panel's x tick labels need below its axis, mirroring the rotations the renderer applies."""
    if isinstance(panel, (AnnotationPanelSpec, NetworkPanelSpec)):
        return 0.0
    if isinstance(panel, LinePanelSpec):
        labels = getattr(panel, "x_tick_labels", None)
        if labels:
            return max(_TICK_BAND_PX, _rotated_band_px(labels, 30, _BAR_LABEL_MAXLEN))
        if getattr(panel, "x_is_time", False):
            band = max(_TICK_BAND_PX, _rotated_band_px((_DATE_TICK_SAMPLE,), 30, len(_DATE_TICK_SAMPLE)))
            return band + (_RANGESLIDER_PX if getattr(panel, "rangeslider", False) else 0.0)
        return _TICK_BAND_PX
    if isinstance(panel, BarPanelSpec):
        if getattr(panel, "orientation", "vertical") == "horizontal":
            return _TICK_BAND_PX
        cats = list(panel.categories)
        maxlen = _BAR_LABEL_MAXLEN if panel.label_maxlen is None else int(panel.label_maxlen)
        if panel.xtick_rotation:
            angle = float(panel.xtick_rotation)
        elif len(cats) > 25:
            angle = 45.0
        elif any(len(str(c)) > _BAR_LABEL_MAXLEN for c in cats):
            angle = 30.0
        else:
            angle = 0.0
        return max(_TICK_BAND_PX, _rotated_band_px(cats, angle, maxlen)) if angle else _TICK_BAND_PX
    if isinstance(panel, HeatmapPanelSpec):
        return max(_TICK_BAND_PX, _rotated_band_px(panel.col_labels, 45, _BAR_LABEL_MAXLEN))
    if isinstance(panel, ViolinPanelSpec):
        return max(_TICK_BAND_PX, _rotated_band_px(panel.group_labels, 30, _VIOLIN_LABEL_MAXLEN))
    if isinstance(panel, ConfusionMarginsPanelSpec):
        return _TICK_BAND_PX
    return _TICK_BAND_PX


def panel_bottom_furniture_px(panel: Any) -> float:
    """Pixels the panel draws BELOW its x-axis domain: tick labels plus the x-axis title, if any."""
    if panel is None:
        return 0.0
    band = _tick_band_px(panel)
    xlabel = str(getattr(panel, "xlabel", "") or "")
    if xlabel and not isinstance(panel, AnnotationPanelSpec):
        band += _AXIS_TITLE_BAND_PX + (xlabel.count("\n")) * (_AXIS_TITLE_FONT_PX + 4)
    return band


def row_bottom_furniture_px(row: Sequence[Any]) -> float:
    """The tallest bottom furniture among one grid row's panels."""
    return max((panel_bottom_furniture_px(p) for p in row if p is not None), default=0.0)


def vertical_spacing_fraction(panels: Sequence[Sequence[Any]], subplot_titles: Sequence[str], cols: int, plot_height_px: float) -> float:
    """``make_subplots`` ``vertical_spacing`` for this grid: the widest needed inter-row gap in px, as a fraction.

    ``subplot_titles`` is the flat, row-major, already ``<br>``-wrapped title list passed to ``make_subplots``.
    ``plot_height_px`` is the plotting region's height (the figure height minus the top/bottom margins).
    """
    rows = len(panels)
    if rows <= 1:
        return 0.0
    gap_px = 0.0
    for r in range(rows - 1):
        below_titles = [t for t in subplot_titles[(r + 1) * cols : (r + 2) * cols] if t]
        title_lines = max((t.count("<br>") + 1 for t in below_titles), default=0)
        gap_px = max(gap_px, row_bottom_furniture_px(panels[r]) + title_lines * _TITLE_LINE_PX + _GAP_PAD_PX)
    frac = gap_px / max(float(plot_height_px), 1.0)
    # plotly rejects vertical_spacing > 1/(rows-1); stay under it so each row keeps a sliver of height.
    return min(frac, 0.9 / (rows - 1))


def _uses_secondary_y(panel: Any) -> bool:
    """Whether a line panel draws a right-hand axis (any series flagged ``secondary_y``)."""
    flags = getattr(panel, "secondary_y", None)
    if flags is None:
        return False
    return any(flags) if isinstance(flags, tuple) else bool(flags)


def panel_left_furniture_px(panel: Any) -> float:
    """Pixels the panel draws LEFT of its plot area: y tick labels plus the y-axis title, if any."""
    if panel is None or isinstance(panel, (AnnotationPanelSpec, NetworkPanelSpec)):
        return 0.0
    if isinstance(panel, BarPanelSpec) and getattr(panel, "orientation", "vertical") == "horizontal":
        maxlen = _BAR_LABEL_MAXLEN if panel.label_maxlen is None else int(panel.label_maxlen)
        ticks = max((measured_text_width_pt(str(c)[:maxlen], _TICK_FONT_PX) for c in panel.categories), default=0.0) + 8.0
    elif isinstance(panel, (HeatmapPanelSpec, ConfusionMarginsPanelSpec)):
        ticks = max((measured_text_width_pt(str(c)[:_BAR_LABEL_MAXLEN], _TICK_FONT_PX) for c in panel.row_labels), default=0.0) + 8.0
    else:
        ticks = _Y_TICK_BAND_PX
    ylabel = str(getattr(panel, "ylabel", "") or "")
    return ticks + (_AXIS_TITLE_BAND_PX if ylabel else 0.0)


def panel_right_furniture_px(panel: Any) -> float:
    """Pixels the panel draws RIGHT of its plot area: a secondary y-axis' ticks and title."""
    if isinstance(panel, LinePanelSpec) and _uses_secondary_y(panel):
        return _Y_TICK_BAND_PX + (_AXIS_TITLE_BAND_PX if panel.secondary_ylabel else 0.0)
    return 0.0


def horizontal_gap_px(panels: Sequence[Sequence[Any]], cols: int) -> float:
    """The widest gap any column boundary needs: the left panel's right furniture plus the right panel's left furniture."""
    gap_px = 0.0
    for row in panels:
        for c in range(cols - 1):
            left = row[c] if c < len(row) else None
            right = row[c + 1] if c + 1 < len(row) else None
            if right is None:
                continue
            gap_px = max(gap_px, panel_right_furniture_px(left) + panel_left_furniture_px(right) + _GAP_PAD_PX)
    return gap_px


def horizontal_spacing_fraction(panels: Sequence[Sequence[Any]], cols: int, plot_width_px: float) -> float:
    """Column gap as a fraction of the plot width: 0.08 is the floor, and a boundary whose neighbours carry a secondary y-axis or long category labels needs more.

    Capped at plotly's ``1 / (cols - 1)`` ceiling (half of it, so the panels keep a usable width).
    """
    if cols <= 1:
        return 0.08
    return min(max(0.08, horizontal_gap_px(panels, cols) / max(plot_width_px, 1.0)), 0.5 / (cols - 1))
