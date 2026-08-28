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

from mlframe.reporting.spec import ConfusionMarginsPanelSpec, HeatmapPanelSpec

from mlframe.reporting.colors import TREND_LINE
from ._plotly_color import _mpl_to_plotly_cmap
from ._shared_helpers import _HEATMAP_CELL_TEXT_MAX, _finite_range, _thin_tick_positions


def _confusion_margins(self, fig, p: ConfusionMarginsPanelSpec, row: int, col: int) -> None:
    """Render a confusion matrix with row/column margins folded into the axis tick labels (predicted-class columns get volume, true-class rows get support), then delegates the actual grid to ``_heatmap``."""
    # Plotly subplot cells cannot host nested marginal axes the way the matplotlib subgridspec does, so the
    # margins are folded into the axis tick labels: each predicted-class column header carries its volume and
    # each true-class row header its support. The heatmap itself reuses the HeatmapPanelSpec renderer.
    col_labels = tuple(f"{lab}<br>(vol={int(v)})" for lab, v in zip(p.col_labels, np.asarray(p.col_margin)))
    row_labels = tuple(f"{lab} (n={int(v)})" for lab, v in zip(p.row_labels, np.asarray(p.row_margin)))
    # Newline, matching matplotlib: the note is a SECOND line of the title, not a clause appended to the
    # first. Joining with a dash pair also put that pair into rendered user-facing text.
    title = p.title if not p.note else f"{p.title}\n{p.note}"
    heat = HeatmapPanelSpec(
        matrix=p.matrix, row_labels=row_labels, col_labels=col_labels,
        title=title, xlabel=p.xlabel, ylabel=p.ylabel, colormap=p.colormap,
        cell_text=p.cell_text, text_format=p.text_format, colorbar_label=p.colorbar_label,
    )
    self._heatmap(fig, heat, row, col)

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
            for level, color in p.threshold_contours:
                if not (lo < level < hi):  # contour only exists when the level is crossed
                    continue
                fig.add_trace(
                    go.Contour(z=mat.tolist(), x=list(p.col_labels), y=list(p.row_labels),
                               contours=dict(start=level, end=level, size=1,
                                             coloring="none", showlabels=False),
                               line=dict(color=color, width=1.6),
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
                def _to_cat(v: float):
                    """Map a value-space coordinate onto this panel's category axis via its own (lo, hi) range."""
                    _idx = (float(v) - _lo) / (_hi - _lo) * (_nb - 1)
                    return p.col_labels[round(min(max(_idx, 0.0), _nb - 1.0))]

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
                        go.Scatter(x=[_to_cat(tx0), _to_cat(tx1)], y=[_to_cat(ty0), _to_cat(ty1)],
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
