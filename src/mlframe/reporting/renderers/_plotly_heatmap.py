"""Heatmap-family panel renderers for the plotly backend.

Carved out of ``plotly.py`` to keep that module under the house 1000-LOC limit. These three are the largest
self-contained group in the renderer and share no state with the rest of it: ``_heatmap`` draws the grid,
``_confusion_margins`` folds row/column margins into the tick labels and then delegates to it, and
``_colorbar_placement`` pins each colorbar beside its own subplot.

They are defined here as plain functions taking ``self`` and are BOUND onto ``PlotlyRenderer`` at the bottom
of ``plotly.py``, so ``renderer._heatmap(...)`` resolves exactly as before and no call site changes.

Every module-level name these functions reach for is imported explicitly below. A moved function imports
clean but ``NameError``s at first CALL if it references a parent-module global with no matching import here
(name lookup is lazy) -- the ``_go`` lazy plotly import is exactly that shape, so it is imported rather than
assumed.
"""

from __future__ import annotations


import numpy as np

from mlframe.reporting.renderers._shared_helpers import heatmap_value_to_index

from mlframe.reporting.spec import ConfusionMarginsPanelSpec, HeatmapPanelSpec

from mlframe.reporting.colors import TREND_LINE
from ._plotly_color import _mpl_to_plotly_cmap
from ._shared_helpers import _HEATMAP_CELL_TEXT_MAX, _finite_range, _thin_tick_positions

# Share of the subplot cell each marginal strip takes, and the gap between a strip and the grid it annotates.
_MARGIN_STRIP_FRAC = 0.18
_MARGIN_STRIP_GAP = 0.02


def _cell_domains(fig, row: int, col: int):
    """``(x_domain, y_domain, x_axis_name, y_axis_name)`` of one subplot cell, or ``None`` when unreadable."""
    grid = getattr(fig, "_grid_ref", None)
    if not grid:
        return None
    try:
        axis_pair = grid[row - 1][col - 1][0]
        x_name, y_name = axis_pair.layout_keys[0], axis_pair.layout_keys[1]
        x_dom, y_dom = fig.layout[x_name].domain, fig.layout[y_name].domain
    except (IndexError, KeyError, AttributeError, TypeError):
        return None
    if not x_dom or not y_dom:
        return None
    return tuple(x_dom), tuple(y_dom), x_name, y_name


def _next_axis_names(fig):
    """The next free ``('xaxisN', 'yaxisN')`` pair in the layout, for a manually-domained nested axis."""
    used = [k for k in fig.layout if isinstance(k, str) and k.startswith("xaxis")]
    idx = max((int(k[5:]) if k[5:] else 1) for k in used) if used else 1
    return f"xaxis{idx + 1}", f"yaxis{idx + 1}"


def _confusion_margins(self, fig, p: ConfusionMarginsPanelSpec, row: int, col: int) -> None:
    """Confusion matrix with TRUE marginal bar axes (row support on the right, column volume on top).

    matplotlib draws these as real bar axes via a subgridspec; plotly used to fold the same numbers into the tick
    label strings, so one spec produced two visibly different figures and only one of them let a reader compare two
    class supports by length. Plotly has no nested-subplot primitive, but a subplot cell is only a pair of axis
    DOMAINS -- so the cell is split by hand: the heatmap's own axes shrink to the lower-left block and two extra
    axis pairs are created over the remaining strips, each matching the heatmap's categorical axis so the bars stay
    aligned with the rows and columns they measure. When the cell's domains cannot be read (a figure built outside
    ``make_subplots``), the tick-label fallback still applies, so the margins are never silently lost.
    """
    dom = _cell_domains(fig, row, col)
    # Newline, matching matplotlib: the note is a SECOND line of the title, not a clause appended to the
    # first. Joining with a dash pair also put that pair into rendered user-facing text.
    title = p.title if not p.note else f"{p.title}\n{p.note}"
    row_margin = np.asarray(p.row_margin, dtype=np.float64)
    col_margin = np.asarray(p.col_margin, dtype=np.float64)

    if dom is None:
        col_labels = tuple(f"{lab}<br>(vol={int(v)})" for lab, v in zip(p.col_labels, col_margin))
        row_labels_folded = tuple(f"{lab} (n={int(v)})" for lab, v in zip(p.row_labels, row_margin))
        heat = HeatmapPanelSpec(
            matrix=p.matrix, row_labels=row_labels_folded, col_labels=col_labels,
            title=title, xlabel=p.xlabel, ylabel=p.ylabel, colormap=p.colormap,
            cell_text=p.cell_text, text_format=p.text_format, colorbar_label=p.colorbar_label,
        )
        self._heatmap(fig, heat, row, col)
        return

    heat = HeatmapPanelSpec(
        matrix=p.matrix, row_labels=tuple(p.row_labels), col_labels=tuple(p.col_labels),
        title=title, xlabel=p.xlabel, ylabel=p.ylabel, colormap=p.colormap,
        cell_text=p.cell_text, text_format=p.text_format, colorbar_label=p.colorbar_label,
    )
    self._heatmap(fig, heat, row, col)

    # Deferred like the sibling below: plotly.py imports THIS module, so a module-level import would cycle.
    from .plotly import _go

    go = _go()
    (x0, x1), (y0, y1), x_name, y_name = dom
    w, h = float(x1) - float(x0), float(y1) - float(y0)
    grid_x1 = float(x1) - w * _MARGIN_STRIP_FRAC
    grid_y1 = float(y1) - h * _MARGIN_STRIP_FRAC
    # Shrink the heatmap onto the lower-left block; the strips take what is left.
    fig.layout[x_name].domain = (float(x0), grid_x1)
    fig.layout[y_name].domain = (float(y0), grid_y1)

    heat_x, heat_y = x_name.replace("axis", ""), y_name.replace("axis", "")
    bar_row_labels = [str(lab) for lab in p.row_labels]
    bar_col_labels = [str(lab) for lab in p.col_labels]

    # Right strip: one horizontal bar per TRUE class, matching the heatmap's categorical y so the bars line up
    # with the rows they measure.
    rx, ry = _next_axis_names(fig)
    fig.layout[rx] = dict(
        domain=(min(1.0, grid_x1 + w * _MARGIN_STRIP_GAP), float(x1)), anchor=ry.replace("axis", ""),
        title=dict(text=p.row_margin_label, font=dict(size=9)), tickfont=dict(size=8),
        showgrid=False, zeroline=False,
    )
    fig.layout[ry] = dict(
        domain=(float(y0), grid_y1), anchor=rx.replace("axis", ""), matches=heat_y, showticklabels=False, showgrid=False,
    )
    fig.add_trace(go.Bar(
        x=row_margin, y=bar_row_labels, orientation="h", marker=dict(color=TREND_LINE), showlegend=False,
        hovertemplate="%{y}<br>" + str(p.row_margin_label) + "=%{x}<extra></extra>",
        xaxis=rx.replace("axis", ""), yaxis=ry.replace("axis", ""),
    ))

    # Top strip: one vertical bar per PREDICTED class, matching the heatmap's categorical x.
    tx, ty = _next_axis_names(fig)
    fig.layout[tx] = dict(
        domain=(float(x0), grid_x1), anchor=ty.replace("axis", ""), matches=heat_x, showticklabels=False, showgrid=False,
    )
    fig.layout[ty] = dict(
        domain=(min(1.0, grid_y1 + h * _MARGIN_STRIP_GAP), float(y1)), anchor=tx.replace("axis", ""),
        title=dict(text=p.col_margin_label, font=dict(size=9)), tickfont=dict(size=8),
        showgrid=False, zeroline=False,
    )
    fig.add_trace(go.Bar(
        x=bar_col_labels, y=col_margin, marker=dict(color=TREND_LINE), showlegend=False,
        hovertemplate="%{x}<br>" + str(p.col_margin_label) + "=%{y}<extra></extra>",
        xaxis=tx.replace("axis", ""), yaxis=ty.replace("axis", ""),
    ))

def _colorbar_placement(fig, row: int, col: int, label) -> dict:
    """Pin a heatmap's colorbar beside ITS OWN subplot instead of plotly's default paper position.

    ``colorbar.x`` / ``colorbar.len`` default to a single paper-space position, so on a multi-heatmap
    figure every colorbar stacks in the same place -- they overlap into an unreadable pile and none of
    them sits next to the panel it describes. Reading the subplot's own domain and anchoring the bar to
    its right edge keeps each bar with its heatmap regardless of the grid shape.
    """
    placement: dict = {"title": label} if label else {}
    grid = getattr(fig, "_grid_ref", None)
    if not grid:
        return placement
    try:
        axis_pair = grid[row - 1][col - 1][0]
        x_name = axis_pair.layout_keys[0]
        y_name = axis_pair.layout_keys[1]
        x_dom = fig.layout[x_name].domain
        y_dom = fig.layout[y_name].domain
    except (IndexError, KeyError, AttributeError, TypeError):
        return placement
    if not x_dom or not y_dom:
        return placement
    placement.update(
        x=min(1.0, float(x_dom[1]) + 0.01),
        len=max(0.05, float(y_dom[1]) - float(y_dom[0])),
        y=(float(y_dom[0]) + float(y_dom[1])) / 2.0,
        yanchor="middle",
        thickness=12,
    )
    return placement

def _heatmap(self, fig, p: HeatmapPanelSpec, row: int, col: int) -> None:
    """Render a ``go.Heatmap`` panel: draws per-cell text via individual ``add_annotation`` calls (luminance-flipped color, skipped above ``_HEATMAP_CELL_TEXT_MAX`` cells to avoid an unreadable/slow grid), optional iso-value threshold contour overlays, an optional robust trend line, and thins x/y tick labels to a readable subset on dense grids."""
    # Function-local, not top-level: ``plotly.py`` imports THIS module at its bottom to bind these methods,
    # so a module-level ``from .plotly import _go`` would be a hard import cycle. Deferring it to call time
    # breaks the cycle and still reuses the parent's single cached plotly import.
    from .plotly import _go

    go = _go()
    from mlframe.reporting.colors import resolve_heatmap_cmap
    cmap_name = resolve_heatmap_cmap(p.colormap)

    # Name the axes and the value in the tooltip instead of accepting plotly's default
    # "x: 1 / y: 13 / z: 0.684 / trace 804" -- grid indices, an unlabelled number and an internal trace
    # id, none of which a reader can act on. ``xlabel`` / ``ylabel`` / ``colorbar_label`` already carry
    # the human names, so reuse them; ``cell_hovertext`` (when a builder supplies it) adds per-cell
    # support, which is what decides whether a cell is worth believing.
    _zname = p.colorbar_label or "value"
    _xname = p.xlabel or "x"
    _yname = p.ylabel or "y"
    _hover_extra = "<br>%{text}" if p.cell_hovertext is not None else ""
    fig.add_trace(
        go.Heatmap(z=p.matrix.tolist(),
                   x=list(p.col_labels), y=list(p.row_labels),
                   text=p.cell_hovertext.tolist() if p.cell_hovertext is not None else None,
                   # ``<extra></extra>`` suppresses the trace-name box ("trace 804").
                   hovertemplate=(f"{_xname}: %{{x}}<br>{_yname}: %{{y}}<br>{_zname}: %{{z:.4g}}" f"{_hover_extra}<extra></extra>"),
                   colorscale=_mpl_to_plotly_cmap(cmap_name),
                   colorbar=self._colorbar_placement(fig, row, col, p.colorbar_label),
                   showscale=True),
        row=row, col=col,
    )
    # Per-cell text via add_annotation instead of
    # plotly's built-in ``text`` + ``texttemplate`` (which uses
    # one global font color and produces white-on-yellow
    # invisibility on viridis high-end / RdYlBu high-end). Per-cell
    # ``auto_text_color`` flips by perceived luminance.
    # Skip per-cell text on an empty / all-non-finite matrix (nanmin raises / poisons the color scale) or a
    # huge grid where the per-annotation O(cells) plotly layout copy stalls and the text is unreadable soup anyway.
    rng = _finite_range(p.matrix)
    if p.cell_text is not None and rng is not None and p.matrix.size <= _HEATMAP_CELL_TEXT_MAX:
        from mlframe.reporting.colors import auto_text_colors_batch
        mat = p.matrix
        vmin, vmax = rng
        # One vectorized colormap sample for the whole grid instead of one matplotlib call per cell
        # (bit-identical to the per-cell auto_text_color -- verified in bench_auto_text_colors_batch.py).
        text_colors = auto_text_colors_batch(np.where(np.isfinite(mat), mat, vmin), cmap_name, vmin=vmin, vmax=vmax)
        # ``fig.add_annotation`` re-validates the WHOLE growing ``layout.annotations`` tuple on every
        # call (plotly's own O(n) per-mutation cost), so a per-cell loop is O(cells^2) -- measured
        # 534x at 400 cells (bench_annotation.py). The first call is kept as-is to let plotly resolve
        # this subplot's xref/yref (row/col -> axis-reference mapping is plotly-internal, not worth
        # reimplementing); every remaining cell reuses that SAME xref/yref (constant for a fixed
        # row/col) and is appended in ONE batched tuple assignment instead of N individual calls.
        cells = [(i, j) for i in range(mat.shape[0]) for j in range(mat.shape[1])]
        if cells:
            i0, j0 = cells[0]
            fig.add_annotation(
                text=format(p.cell_text[i0, j0], p.text_format),
                x=p.col_labels[j0], y=p.row_labels[i0],
                showarrow=False,
                font=dict(color=text_colors[i0, j0], size=10),
                row=row, col=col,
            )
            last = fig.layout.annotations[-1]
            xref, yref = last.xref, last.yref
            rest = [
                go.layout.Annotation(
                    text=format(p.cell_text[i, j], p.text_format),
                    x=p.col_labels[j], y=p.row_labels[i],
                    xref=xref, yref=yref,
                    showarrow=False,
                    font=dict(color=text_colors[i, j], size=10),
                )
                for i, j in cells[1:]
            ]
            if rest:
                fig.layout.annotations = fig.layout.annotations + tuple(rest)
    # Iso-value contour overlays at named matrix levels (PSI 0.10 / 0.25 triage lines). Drawn as a line-only
    # go.Contour over the categorical axes: plotly maps category positions to 0..n-1, so the numeric contour
    # x/y (the label lists) line up cell-for-cell with the heatmap.
    if p.threshold_contours:
        mat = np.asarray(p.matrix, dtype=float)
        if mat.ndim == 2 and mat.shape[0] >= 2 and mat.shape[1] >= 2:
            lo, hi = float(np.nanmin(mat)), float(np.nanmax(mat))
            for _entry in p.threshold_contours:
                level, color = _entry[0], _entry[1]
                _dash = _entry[2] if len(_entry) > 2 else "solid"
                if not (lo < level < hi):  # contour only exists when the level is crossed
                    continue
                fig.add_trace(
                    go.Contour(z=mat.tolist(), x=list(p.col_labels), y=list(p.row_labels),
                               contours=dict(start=level, end=level, size=1,
                                             coloring="none", showlabels=False),
                               line=dict(color=color, width=1.6, dash=_dash),
                               showscale=False, hoverinfo="skip"),
                    row=row, col=col,
                )
    if p.trend_line is not None and p.trend_xy is not None:
        from mlframe.reporting.renderers._trend import robust_fit_endpoints
        # The heatmap axes are CATEGORY axes (one position per bin label), but robust_fit_endpoints and the
        # y=x diagonal come back in VALUE space. Plotting those directly put the fit thousands of positions
        # off a 10-category axis (observed endpoints ~3113..6533 on a 10-bin grid), so the line was simply
        # not on the chart. matplotlib already maps value -> bin index for exactly this reason; this is the
        # same mapping, against the same (lo, hi) the panel binned on.
        _xv = np.asarray(p.trend_xy[0], dtype=np.float64).ravel()
        _yv = np.asarray(p.trend_xy[1], dtype=np.float64).ravel()
        _fin = np.isfinite(_xv) & np.isfinite(_yv)
        _nb = len(p.col_labels)
        if int(_fin.sum()) >= 2 and _nb >= 2:
            _lo = float(min(_xv[_fin].min(), _yv[_fin].min()))
            _hi = float(max(_xv[_fin].max(), _yv[_fin].max()))
            if _hi > _lo:
                # Fractional bin-index positions, NOT snapped category labels. A plotly category axis accepts a
                # numeric position, and rounding-plus-clamping to the nearest label moved an extrapolated trend
                # endpoint to the axis edge -- which changes the segment's SLOPE, the one thing this panel exists
                # to show. Axis ranges below clip the drawn line the way matplotlib's limits do, without moving
                # the endpoints. The snap also resolved BOTH coordinates against ``col_labels``; the y map now
                # uses the row axis, which is latent for the hexbin builder (it sets both from the same centres)
                # but puts the trend at a nonexistent y category for any spec with asymmetric labels, and plotly
                # appends such a category rather than raising.
                _to_x = heatmap_value_to_index(_lo, _hi, _nb)
                _to_y = heatmap_value_to_index(_lo, _hi, len(p.row_labels))

                fig.add_trace(
                    go.Scatter(x=[p.col_labels[0], p.col_labels[_nb - 1]],
                               y=[p.row_labels[0], p.row_labels[len(p.row_labels) - 1]],
                               mode="lines", line=dict(color="#666666", width=1, dash="dot"),
                               name="y=x", showlegend=True),
                    row=row, col=col,
                )
                ends = robust_fit_endpoints(_xv, _yv, p.trend_line)
                if ends is not None:
                    (tx0, ty0), (tx1, ty1) = ends
                    fig.add_trace(
                        go.Scatter(x=[_to_x(tx0), _to_x(tx1)], y=[_to_y(ty0), _to_y(ty1)],
                                   mode="lines", line=dict(color=TREND_LINE, width=2),
                                   name=f"robust fit ({p.trend_line})", showlegend=True),
                        row=row, col=col,
                    )

    # A density heatmap has ~80 cell labels per axis; one tick each overlaps into soup. Thin to <= _HEATMAP_MAX_TICKS
    # evenly-spaced category ticks (the full grid is still drawn).
    _xt = _thin_tick_positions(len(p.col_labels))
    _yt = _thin_tick_positions(len(p.row_labels))
    fig.update_xaxes(title_text=p.xlabel, row=row, col=col, tickangle=-45, tickmode="array", tickvals=[p.col_labels[i] for i in _xt])
    # Row order must match matplotlib, which switches to origin="lower" for a density panel carrying
    # `trend_xy` (it reads bottom-up, row 0 = lowest value) and keeps the top-down matrix order otherwise.
    # Reversing unconditionally rendered the pred-vs-actual density heatmap VERTICALLY MIRRORED between the
    # two backends -- the same figure, with the trend running the opposite way.
    _reversed = p.trend_xy is None
    _y_kw = {"autorange": "reversed"} if _reversed else {}
    fig.update_yaxes(title_text=p.ylabel, row=row, col=col, tickmode="array", tickvals=[p.row_labels[i] for i in _yt], **_y_kw)
