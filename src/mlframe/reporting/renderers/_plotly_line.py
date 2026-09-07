"""The plotly line panel and the marker-label placement it needs.

Carved out of ``plotly.py`` for the same reason the heatmap and network families were: that module is over
the 1000-LOC house limit, and this is the largest self-contained piece of it. The functions are bound back
onto ``PlotlyRenderer`` at the bottom of the parent, so ``PlotlyRenderer._line`` and the ``_render_panel``
dispatch keep resolving unchanged.

Vertical bands, change points and point markers are all drawn in BATCHES: ``add_vrect`` / ``add_trace`` /
``add_annotation`` each re-validate their whole growing collection per call, which made a per-item loop
super-quadratic across the three together (2 spans 82 ms, 300 spans 150.8 s before batching).
"""

from __future__ import annotations

import logging
from contextlib import nullcontext  # noqa: F401 -- kept for parity with the parent's import surface
from typing import Optional

import numpy as np

from mlframe.reporting.spec import LinePanelSpec

from mlframe.reporting.colors import line_color

from ._plotly_color import _rgba
from ._shared_helpers import PX_PER_INCH as _PX_PER_INCH
from ._shared_helpers import _per_series_flags, epoch_ns_ticks, plotly_axis_suffix, rotated_tick_pitch_in, stagger_label_rows, ticks_that_fit

logger = logging.getLogger(__name__)

#: Plotly's own default figure width. Used only when the layout has not been given one.
_DEFAULT_FIGURE_WIDTH_PX = 600


def _figure_width_px(fig) -> float:
    """The figure's width in px, falling back to plotly's default only when none is set.

    Explicit ``is None`` rather than ``fig.layout.width or 600``: a caller that legitimately sets width 0
    would have been silently rewritten to 600, and a zero-width figure is a bug worth surfacing rather than
    papering over with a default that makes the tick budget lie.
    """
    width = fig.layout.width
    return float(_DEFAULT_FIGURE_WIDTH_PX if width is None else width)


def _line(self, fig, p: LinePanelSpec, row: int, col: int) -> None:
    """Render a multi-series line panel: per-series style/color/secondary-y/fill-to-baseline, an optional uncertainty band, vspans/vlines (datetime-safe), and point markers; secondary-y series get their own right-hand axis when any series requests it."""
    # Lazy, matching the heatmap sibling: the parent imports THIS module at its own bottom, so a
    # module-level import of the parent's lazy plotly handle or its label constants would be a hard cycle.
    # By call time the parent is fully loaded.
    from .plotly import _STACKED_LABEL_SHIFT_PX, _go, _marker_symbol, pd_timestamp

    go = _go()

    ys = p.y if isinstance(p.y, tuple) else (p.y,)
    xs_per_series = isinstance(p.x, tuple)
    labels = p.series_labels if p.series_labels is not None else (None,) * len(ys)
    styles = p.line_styles if p.line_styles is not None else ("-",) * len(ys)
    cols = p.colors if p.colors is not None else tuple(line_color(i) for i in range(len(ys)))
    sec = _per_series_flags(p.secondary_y, len(ys))
    fills = _per_series_flags(p.fill_to_baseline, len(ys))
    has_secondary = any(sec)
    # matplotlib linestyle tokens -> plotly dash; "markers" / "lines+markers" select the trace mode.
    _STYLE_MAP = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}

    def _xi(i):
        """Return the x-values for series ``i``: per-series ``p.x[i]`` when the spec carries a tuple of x-arrays, else the single shared ``p.x``."""
        v = p.x[i] if xs_per_series else p.x
        return np.asarray(v) if isinstance(v, np.ndarray) else v

    if p.band is not None:
        x0 = _xi(0)
        lower, upper = np.asarray(p.band[0]), np.asarray(p.band[1])
        band_color = p.band_color if p.band_color is not None else cols[0]
        fig.add_trace(
            go.Scatter(x=np.concatenate([x0, x0[::-1]]),
                       y=np.concatenate([upper, lower[::-1]]),
                       fill="toself", fillcolor=_rgba(band_color, 0.2),
                       line=dict(width=0), hoverinfo="skip",
                       name=p.band_label if p.band_label is not None else "band", showlegend=bool(p.band_label)),
            row=row, col=col,
        )

    for i, y in enumerate(ys):
        token = styles[i % len(styles)]  # nosec B105 - not a credential -- config/format token label or sentinel string constant
        if token == "markers":  # nosec B105 - identifier/config-key name matched by heuristic, not an embedded credential
            mode, dash = "markers", "solid"  # nosec B105 - not a credential -- config/format token label or sentinel string constant
        elif token == "lines+markers":  # nosec B105 - identifier/config-key name matched by heuristic, not an embedded credential
            mode, dash = "lines+markers", "solid"
        else:
            mode, dash = "lines", _STYLE_MAP.get(token, "solid")
        yv = np.asarray(y) if isinstance(y, np.ndarray) else y
        # Area fill under the curve down to the panel baseline. plotly has no "fill to an arbitrary y", and
        # "tonexty" fills to the PREVIOUS TRACE -- so with a non-zero baseline it shaded the gap to whatever
        # series happened to precede this one, a region that encodes nothing, while matplotlib shaded the gap
        # to `fill_baseline`. Lay down an invisible constant-baseline trace first so "tonexty" has the right
        # thing to fill against and both backends shade the same region.
        trace_kw = {}
        if fills[i]:
            if p.fill_baseline == 0.0:
                trace_kw["fill"] = "tozeroy"
            else:
                _bx = _xi(i)
                fig.add_trace(
                    go.Scatter(
                        x=_bx,
                        y=np.full(len(_bx), float(p.fill_baseline)),
                        mode="lines",
                        line=dict(width=0),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=row, col=col, **({"secondary_y": sec[i]} if has_secondary else {}),
                )
                trace_kw["fill"] = "tonexty"
            trace_kw["fillcolor"] = _rgba(cols[i % len(cols)], 0.2)
            if p.step_fill:
                # matplotlib's step="post" steps the FILL EDGE and leaves the line straight; "hv" here stepped
                # the line too, so the same spec drew a staircase on one backend and a polyline on the other.
                # The fill edge is the shared meaning, so the line stays straight and the fill is stepped by
                # emitting the baseline boundary as a step trace.
                trace_kw.setdefault("line_shape", "linear")
        sec_kw = {"secondary_y": sec[i]} if has_secondary else {}
        fig.add_trace(
            go.Scatter(x=_xi(i), y=yv,
                       mode=mode,
                       line=dict(color=cols[i % len(cols)], dash=dash),
                       marker=dict(color=cols[i % len(cols)], size=5),
                       name=labels[i] if i < len(labels) else None,
                       # Per-series, not any(labels) applied identically to every trace: the latter set
                       # showlegend=True on an UNLABELED series whenever ANY other series in the same
                       # panel had a label, rendering a blank/"undefined" legend row for it. matplotlib
                       # doesn't have this problem (ax.get_legend_handles_labels() omits unlabeled
                       # artists automatically).
                       showlegend=bool(labels[i]) if i < len(labels) else False,
                       **trace_kw),
            row=row, col=col, **sec_kw,
        )

    _vspan_rows = self._label_rows(
        [sp[0] for sp in (p.vspans or ())],
        [(sp[4] if len(sp) > 4 else "") for sp in (p.vspans or ())],
        fig, p,
    )
    # Batched. Every one of ``add_vrect`` / ``add_trace`` / ``add_annotation`` re-validates its whole
    # growing collection per call, so a per-item loop is super-quadratic over the three of them
    # together: measured on this panel, 2 spans 82 ms, 20 spans 737 ms, 100 spans 14.3 s and 300 spans
    # 150.8 s. The audit called this latent because today's callers pass one or two bands; a regime
    # chart is exactly the thing that grows with the data, and 100 regimes is not an exotic input.
    _suffix = plotly_axis_suffix(fig, row, col, len(fig._grid_ref[0]) if getattr(fig, "_grid_ref", None) else 1)
    _xref, _yref = f"x{_suffix}", f"y{_suffix}"
    _span_shapes = []
    _span_traces = []
    _span_annotations = []
    for _vspan_i, span in enumerate(p.vspans or ()):
        vx0, vx1, vcolor, valpha = span[0], span[1], span[2], span[3]
        vlabel = span[4] if len(span) > 4 else ""
        _span_shapes.append(
            go.layout.Shape(type="rect", xref=_xref, yref=f"{_yref} domain", x0=vx0, x1=vx1, y0=0, y1=1,
                            fillcolor=_rgba(vcolor, valpha), line=dict(width=0), layer="below")
        )
        if vlabel:
            # No native per-vrect legend in plotly; the invisible scatter proxy carries the label INTO the
            # legend, and the annotation carries it onto the band itself -- which is the only one that survives
            # on a multi-panel interactive figure, where the legend is off (hover identifies the series).
            _span_traces.append(
                go.Scatter(x=[None], y=[None], mode="markers",
                           marker=dict(size=8, color=_rgba(vcolor, max(valpha, 0.3)), symbol="square"),
                           name=vlabel, showlegend=True)
            )
            # Staggered by measured overlap: every vspan label used to be stamped at the same y just
            # above the panel, so two adjacent regimes -- which is what a regime chart is FOR -- printed
            # on top of each other. The colour still ties each label to its band. Hanging DOWNWARD from
            # the top of the plot area, like the vline labels below and for the same reason: stacked
            # upward, the top row lands in the subplot title's strip and prints over it -- seen in the
            # render, with "recovery" written across "regimes and change points".
            _span_annotations.append(
                go.layout.Annotation(x=vx0, y=1.0, xref=_xref, yref=f"{_yref} domain", yanchor="top", xanchor="left",
                                     xshift=3, yshift=-3 - _STACKED_LABEL_SHIFT_PX * _vspan_rows[_vspan_i],
                                     text=vlabel, showarrow=False, font=dict(size=8, color=vcolor))
            )
    if _span_shapes:
        fig.layout.shapes = tuple(fig.layout.shapes) + tuple(_span_shapes)
    if _span_traces:
        fig.add_traces(_span_traces, rows=[row] * len(_span_traces), cols=[col] * len(_span_traces))
    if _span_annotations:
        fig.layout.annotations = tuple(fig.layout.annotations) + tuple(_span_annotations)
    # Same batching as the bands above. ``add_vline`` also does arithmetic on x that raises on a datetime
    # axis, so the line is a shape with x in data coords and y spanning the panel's y-domain, which works
    # on numeric AND datetime axes alike.
    _vline_rows = self._label_rows([v[0] for v in (p.vlines or ())], [v[2] for v in (p.vlines or ())], fig, p)
    _line_shapes = []
    _line_annotations = []
    for _vline_i, (vx, vcolor, vlabel) in enumerate(p.vlines or ()):
        _x = pd_timestamp(vx) if self._is_datetime_like(vx) else vx
        _line_shapes.append(
            go.layout.Shape(type="line", x0=_x, x1=_x, y0=0, y1=1, xref=_xref, yref=f"{_yref} domain", line=dict(color=vcolor, dash="dot", width=1.2))
        )
        if vlabel:
            # Hangs DOWNWARD from the top of the plot area: stacking upward put the first row in the same
            # strip as the subplot title, so clearing the labels of each other collided them with it.
            _line_annotations.append(
                go.layout.Annotation(x=_x, y=1.0, xref=_xref, yref=f"{_yref} domain", yanchor="top",
                                     xanchor="left", xshift=3,
                                     yshift=-3 - _STACKED_LABEL_SHIFT_PX * int(_vline_rows[_vline_i]),
                                     text=vlabel, showarrow=False, font=dict(size=9, color=vcolor))
            )
    if _line_shapes:
        fig.layout.shapes = tuple(fig.layout.shapes) + tuple(_line_shapes)
    if _line_annotations:
        fig.layout.annotations = tuple(fig.layout.annotations) + tuple(_line_annotations)

    _marker_traces = [
        # Marker only, with the label on the legend entry and the hover -- not printed beside the point as
        # well. Carrying it in both places captioned every operating point twice, and the printed copy
        # overhung the panel exactly as it did on the matplotlib twin.
        go.Scatter(x=[mx], y=[my], mode="markers",
                   marker=dict(color=mcolor, size=13, symbol=_marker_symbol(msym),
                               line=dict(color="black", width=0.6)),
                   hovertext=[mlabel or ""], hoverinfo="text" if mlabel else "skip",
                   name=mlabel or None, showlegend=bool(mlabel))
        for mx, my, mlabel, mcolor, msym in p.point_markers or ()
    ]
    if _marker_traces:
        fig.add_traces(_marker_traces, rows=[row] * len(_marker_traces), cols=[col] * len(_marker_traces))

    # ``x_is_time`` with a NUMERIC x means epoch nanoseconds; rotating the labels (all this used to do)
    # leaves them reading "1.62e18". ``epoch_ns_ticks`` no-ops on an already-datetime axis.
    # See the matplotlib twin: builders set ylim deliberately and no line-panel path honoured it.
    _ylim = getattr(p, "ylim", None)
    if _ylim is not None:
        fig.update_yaxes(range=[float(_ylim[0]), float(_ylim[1])], row=row, col=col, secondary_y=False)
    _xkw: dict = dict(title_text=p.xlabel, row=row, col=col, showgrid=p.grid, tickangle=-30 if p.x_is_time else 0)
    # The COUNT comes from the panel's own width, not the helper's fixed six: a -30-degree date label needs
    # real room, and the fixed count crowds a narrow panel exactly as a fixed cap crowded the heatmap ticks.
    _n_dates = 6
    if p.x_is_time:
        _cols = len(fig._grid_ref[0]) if getattr(fig, "_grid_ref", None) else 1
        _panel_w_in = _figure_width_px(fig) / _PX_PER_INCH / max(_cols, 1)
        _n_dates = ticks_that_fit(_panel_w_in, 6, floor=2, pitch_in=rotated_tick_pitch_in(8, 30))
    _tv, _tt = epoch_ns_ticks(_xi(0), n_ticks=_n_dates) if p.x_is_time else (None, None)
    if _tv is not None:
        _xkw.update(tickmode="array", tickvals=_tv, ticktext=_tt)
    fig.update_xaxes(**_xkw)
    fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid, secondary_y=False)
    if has_secondary:
        fig.update_yaxes(title_text=p.secondary_ylabel, row=row, col=col, secondary_y=True, showgrid=False)

def _is_datetime_like(v) -> bool:
    """True if ``v`` is a ``numpy.datetime64`` or a stdlib ``datetime``/``date``; gates the datetime-safe vline path since ``fig.add_vline`` raises ``TypeError`` on datetime x."""
    import datetime as _dt
    if isinstance(v, (np.datetime64,)):
        return True
    if isinstance(v, (_dt.datetime, _dt.date)):
        return True
    return False

def _panel_x_span(cls, p, marker_xs) -> Optional[float]:
    """Width of the panel's x axis in data units: the pinned xlim, else the plotted x, else the markers."""
    if getattr(p, "xlim", None):
        _lim = cls._as_numeric_axis(list(p.xlim))
        if _lim is not None and _lim.size >= 2:
            return max(float(_lim[1]) - float(_lim[0]), 1e-9)
    _x = getattr(p, "x", None)
    if _x is not None:
        _flat = cls._as_numeric_axis(np.asarray(_x[0] if isinstance(_x, tuple) else _x).ravel())
        if _flat is not None:
            _finite = _flat[np.isfinite(_flat)]
            if _finite.size:
                return max(float(_finite.max()) - float(_finite.min()), 1e-9)
    _m = np.asarray(marker_xs, dtype=float)
    return max(float(_m.max()) - float(_m.min()), 1e-9) if _m.size else None

def _as_numeric_axis(values) -> Optional[np.ndarray]:
    """``values`` as plain floats, mapping datetimes onto epoch nanoseconds, or ``None`` if neither works."""
    arr = np.asarray(values)
    if arr.size == 0:
        return None
    try:
        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.astype("datetime64[ns]").astype("int64").astype(float)
        return arr.astype(float)
    except (TypeError, ValueError):
        try:
            import pandas as pd

            return np.asarray(pd.to_datetime(list(values)).astype("int64"), dtype=float)
        except Exception:
            logger.debug("could not read the marker axis for label staggering; falling back to alternating rows", exc_info=True)
            return None

def _label_rows(self, xs, texts, fig, p) -> list:
    """Stacked row per label, measured against the panel's own x range so neighbours never overprint."""
    # Lazy for the same cycle reason as ``_line`` above.
    from .plotly import _STACKED_LABEL_ROWS

    _xs = [x for x in xs if x is not None]
    if not _xs:
        return [0] * len(xs)
    # The AXIS range, not the range of the markers themselves: three change points 0.01 apart on a
    # unit axis are 1% of the panel, and measuring their labels against their own 0.02-wide extent
    # makes every one of them look as though it has the whole panel to itself.
    _num_xs = self._as_numeric_axis(_xs)
    _x_span = self._panel_x_span(p, _num_xs) if _num_xs is not None else None
    if _num_xs is None or _x_span is None:
        # Nothing measurable on this axis; alternate, which is what this did before it could measure.
        return [i % _STACKED_LABEL_ROWS for i in range(len(xs))]
    _cols = len(fig._grid_ref[0]) if getattr(fig, "_grid_ref", None) else 1
    _w_in = _figure_width_px(fig) / _PX_PER_INCH / max(_cols, 1)
    return stagger_label_rows(_num_xs, texts, fontsize=8, x_span=_x_span, width_in=_w_in, max_rows=_STACKED_LABEL_ROWS)
