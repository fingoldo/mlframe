"""Figure-level layout helpers for ``PlotlyRenderer.render``, carved out to keep ``plotly.py`` under the 1000-LOC house limit.

``render`` calls these in the same order and with the same inputs it used inline, so behaviour is unchanged.
"""

from __future__ import annotations

from typing import Any, List, Optional

from mlframe.reporting.spec import FigureSpec, LinePanelSpec

from ._shared_helpers import PANEL_TITLE_FONTSIZE, panel_title_wrap_chars


def _subplot_specs(spec: FigureSpec, cols: int) -> List[List[Optional[dict]]]:
    """Per-cell ``make_subplots`` spec grid (``None`` for an empty cell, ``secondary_y`` for line panels that request it)."""
    from .plotly import _line_uses_secondary_y

    # Per-panel subplot spec: heatmap needs no shared axes; default ``xy`` works for everything else. A line
    # panel that requests a secondary y-axis must declare ``secondary_y=True`` at subplot-creation time (plotly
    # can't add a right axis after the grid is built), so detect that here.
    sub_specs: List[List[Optional[dict]]] = []
    for _r, row in enumerate(spec.panels):
        row_specs: List[Optional[dict]] = []  # None means "no subplot in this cell"
        for c in range(cols):
            if c >= len(row) or row[c] is None:
                # ``None`` means "no subplot here". An empty dict is NOT that -- plotly reads it as a
                # default ``xy`` cell, so a 2x2 grid with one None produced 4 axes for 3 traces and drew
                # an empty framed panel where matplotlib draws nothing at all.
                row_specs.append(None)
            else:
                cell: dict = {"type": "xy"}
                if isinstance(row[c], LinePanelSpec) and _line_uses_secondary_y(row[c]):
                    cell["secondary_y"] = True
                row_specs.append(cell)
        sub_specs.append(row_specs)
    return sub_specs


def _subplot_titles(spec: FigureSpec, cols: int) -> List[str]:
    """Wrapped subplot titles in row-major order (empty string for an empty cell)."""
    from .plotly import _wrap_text_to_figure

    # Subplot titles are HTML annotations: wrap long panel titles (~46 chars/line, matching matplotlib) so they fold instead of bleeding into the adjacent subplot, and convert ``\n`` -> ``<br>`` (plotly drops a raw newline).
    # Measured against the font rather than counted in characters, for the reason the matplotlib twin
    # documents: a diagnostic title is digits and CamelCase identifiers, not average-width characters.
    _panel_wrap = panel_title_wrap_chars(spec.figsize, cols)
    _panel_w_in = float(spec.figsize[0]) / max(cols, 1)
    subplot_titles: List[str] = []
    for row in spec.panels:
        for c in range(cols):
            if c >= len(row) or row[c] is None:
                subplot_titles.append("")
            else:
                subplot_titles.append(
                    _wrap_text_to_figure(
                        getattr(row[c], "title", "") or "",
                        fontsize=PANEL_TITLE_FONTSIZE, width_in=_panel_w_in, fallback_chars=_panel_wrap,
                    )
                )
    return subplot_titles


def _apply_legend_layout(fig: Any, spec: FigureSpec, static_legend: bool, below_axis_px: int, px_per_inch: float) -> None:
    """Place the figure legend: below the plot for static exports, outside or lower-right when a panel asks for it."""
    if static_legend:
        # Park the legend BELOW the plot area (horizontal, centred) so it never overlaps subplot titles / the suptitle the way a default top-right in-plot legend does on multi-panel figures.
        fig.update_layout(legend=dict(
            font=dict(size=9), itemsizing="constant",
            bgcolor="rgba(255,255,255,0.6)",
            orientation="h", yanchor="top", y=-below_axis_px / max(spec.figsize[1] * px_per_inch, 1.0), xanchor="center", x=0.5,
        ))
    elif any(getattr(pn, "legend_outside", False) for row in spec.panels for pn in row if pn is not None):
        # legend_outside / legend_ncol were matplotlib-only, so the many-series overlays they exist for got an
        # in-axes legend covering the very curves on the HTML backend.
        _ncol = max((int(getattr(pn, "legend_ncol", 1)) for row in spec.panels for pn in row if pn is not None), default=1)
        fig.update_layout(legend=dict(
            font=dict(size=9), itemsizing="constant", bgcolor="rgba(255,255,255,0.6)",
            yanchor="middle", y=0.5, xanchor="left", x=1.02,
            # plotly has no column count; a legend the caller wanted in N columns is at least made to
            # TRACK across instead of running one very tall single file down the side.
            orientation="h" if _ncol > 1 else "v",
        ))
    elif any(getattr(pn, "legend_loc", None) == "lower right" for row in spec.panels for pn in row if pn is not None):
        # Mirrors the matplotlib ``legend_loc="lower right"`` placement (e.g. the calibration reliability
        # scatter, whose most informative points sit near the origin) -- plotly's own default in-axes
        # legend corner is top-right, which would sit on top of that region instead of the empty one.
        fig.update_layout(legend=dict(
            font=dict(size=9), itemsizing="constant", bgcolor="rgba(255,255,255,0.7)",
            yanchor="bottom", y=0.02, xanchor="right", x=0.98,
        ))
