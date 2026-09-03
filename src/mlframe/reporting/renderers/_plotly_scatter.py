"""``PlotlyRenderer._scatter``, carved out to keep ``plotly.py`` under the 1000-LOC house limit.

Same pattern as ``_plotly_network`` / ``_plotly_heatmap``: the method is defined here as a plain function and
bound back onto the class at the bottom of ``plotly.py``, so ``PlotlyRenderer._scatter`` and the ``_render_panel``
dispatch keep resolving unchanged.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from mlframe.reporting.colors import OVERLAY_LINE, PERFECT_FIT_LINE, TREND_LINE
from mlframe.reporting.spec import ScatterPanelSpec

from ._plotly_color import _axis_ref, _mpl_to_plotly_cmap
from ._shared_helpers import _SCATTER_MAX_POINTS, low_evidence_mask, select_per_point

logger = logging.getLogger(__name__)


def _scatter(self, fig, p: ScatterPanelSpec, row: int, col: int) -> None:
    """Render a scatter panel: downsamples above ``_SCATTER_MAX_POINTS`` (extremes-preserving), converts mpl marker-area sizing to plotly pixel-diameter, switches to WebGL above ``_SCATTER_WEBGL_THRESHOLD`` points (unless error bars are present, which Scattergl doesn't support), and layers optional highlight points, trend line, uncertainty band, overlay line, and a perfect-fit y=x diagonal on top."""
    # Lazy, function-local, matching ``_plotly_network``: the parent module imports this one at its own bottom,
    # so a module-level ``from .plotly import ...`` would be a hard cycle. By call time the parent is loaded.
    from .plotly import _SCATTER_WEBGL_THRESHOLD, _err_to_plotly, _go, _warn_scatter_downsample

    go = _go()

    x = np.asarray(p.x)
    y = np.asarray(p.y)
    n = len(x)

    # Per-point size / color arrays must follow the SAME row subset as x/y when downsampling.
    size_arr = p.point_size if isinstance(p.point_size, np.ndarray) else None
    color_arr = p.point_color if isinstance(p.point_color, np.ndarray) else None

    idx = None  # the downsample selection, so per-point spec arrays can follow the same rows
    if n > _SCATTER_MAX_POINTS:
        _warn_scatter_downsample(n)
        from mlframe.reporting.charts import subsample_preserving_extremes
        idx = subsample_preserving_extremes(x, y, sample_size=_SCATTER_MAX_POINTS)
        x, y = x[idx], y[idx]
        if size_arr is not None and len(size_arr) == n:
            size_arr = size_arr[idx]
        if color_arr is not None and len(color_arr) == n:
            color_arr = color_arr[idx]

    marker: dict = dict(opacity=p.point_alpha)
    # ScatterPanelSpec.point_size follows matplotlib's ``s=`` (area in pt^2); plotly marker.size is the
    # DIAMETER in px. Without conversion large mpl areas blow up to giant circles and the auto-axis range
    # goes haywire. Conversion: plotly_diameter_px = sqrt(mpl_area_pt2) * 1.33.
    if size_arr is not None:
        marker["size"] = np.sqrt(np.maximum(np.asarray(size_arr, dtype=float), 0.0)) * 1.33
    else:
        marker["size"] = float(math.sqrt(max(float(p.point_size), 0.0)) * 1.33)
    if color_arr is not None:
        marker["color"] = np.asarray(color_arr)
        marker["colorscale"] = _mpl_to_plotly_cmap(p.colormap)
        # A diverging map autoscaled to the data puts its neutral midpoint at the middle of the observed
        # range instead of at zero, which silently changes what every colour means.
        if p.color_vmin is not None:
            marker["cmin"] = p.color_vmin
        if p.color_vmax is not None:
            marker["cmax"] = p.color_vmax
        marker["showscale"] = bool(p.colorbar_label)
        if p.colorbar_label:
            marker["colorbar"] = dict(title=p.colorbar_label)
    elif p.point_color is not None:
        marker["color"] = p.point_color

    # inline_labels are (x, y, text) triples placed AT THOSE COORDINATES on matplotlib. Using them as
    # per-point marker text put them on the wrong points, and the len == n gate silently dropped a shorter
    # list entirely -- the common case, since these annotate a handful of bins, not every row. They are
    # emitted as annotations below instead; there is no per-point text path here.
    # add_annotation re-validates the whole growing tuple per call (O(n^2) over a loop), so only the FIRST call
    # is made through it -- to let plotly resolve this subplot's axis refs -- and the rest are built as plain
    # Annotation objects and assigned in one go.
    _labels = tuple(p.inline_labels or ())
    _lab_colors = tuple(p.inline_label_colors or ())

    def _lab_font(i: int):
        """Label font, taking the builder's per-label colour when it supplied one (a label sitting on its own marker)."""
        return dict(size=8, color=_lab_colors[i]) if i < len(_lab_colors) else dict(size=8)

    if _labels:
        _lx, _ly, _ltext = _labels[0]
        fig.add_annotation(x=_lx, y=_ly, text=str(_ltext), showarrow=False, font=_lab_font(0), yshift=8, row=row, col=col)
        if len(_labels) > 1:
            _ref = fig.layout.annotations[-1]
            _rest = tuple(
                go.layout.Annotation(
                    x=lx, y=ly, text=str(txt), showarrow=False, font=_lab_font(_i + 1), yshift=8,
                    xref=_ref.xref, yref=_ref.yref,
                )
                for _i, (lx, ly, txt) in enumerate(_labels[1:])
            )
            fig.layout.annotations = fig.layout.annotations + _rest

    # Per-point error bars (e.g. Wilson CIs on reliability bins). CI panels carry n=bin-count points (no
    # downsample reorder), so the error arrays align with x/y as-passed; only attach when not downsampled.
    error_y = error_x = None
    if n <= _SCATTER_MAX_POINTS:
        error_y = _err_to_plotly(p.y_err)
        error_x = _err_to_plotly(p.x_err)

    # WebGL renders large scatters orders of magnitude faster than SVG-mode go.Scatter; ndarrays pass
    # through to plotly natively (faster + smaller than .tolist()). Scattergl has no error_y/error_x support,
    # so a panel carrying error bars uses SVG-mode go.Scatter (bin counts are small, so no perf concern).
    if error_y is not None or error_x is not None:
        trace_cls = go.Scatter
    else:
        trace_cls = go.Scattergl if n > _SCATTER_WEBGL_THRESHOLD else go.Scatter
    # Points resting on too little data to be an observation: drawn hollow in their own trace, mirroring the
    # matplotlib twin, so the two backends do not disagree about which bins are readable.
    weak = low_evidence_mask(p.low_evidence_indices if len(x) == len(np.asarray(p.x)) else None, len(x))

    def _sel_err(err, mask, muted: bool = False):
        """Narrow a plotly error spec to the masked points, keeping symmetric/asymmetric structure.

        ``muted=True`` also RESTYLES the interval, which the matplotlib twin has always done
        (``ecolor="0.75"``, ``elinewidth=0.6``, ``capsize=0``) to implement the spec's stated intent that a
        low-evidence interval reads as "we know nothing here". Narrowing the arrays alone drew the weak whiskers
        identically to the strong ones, so the distinction ``low_evidence_ci_width`` exists for was lost on the
        HTML backend -- and worse, the weak trace sets ``marker.color`` to a fully transparent
        ``rgba(0,0,0,0)``, which plotly's error bars INHERIT when ``error_y.color`` is unset, so they could
        render invisible rather than merely undifferentiated.
        """
        if err is None:
            return None
        out = dict(err)
        for key in ("array", "arrayminus"):
            if out.get(key) is not None:
                out[key] = np.asarray(out[key])[mask]
        if muted:
            out["color"] = "#c0c0c0"
            out["thickness"] = 0.8
            out["width"] = 0
        return out

    def _sel_marker(mask):
        """Narrow every per-point marker field to the masked points."""
        return {k: select_per_point(v, mask, len(x)) for k, v in marker.items()}

    def _sel_hover(mask=None):
        """The spec's per-point hovertext, narrowed to this trace's own rows.

        Attached HERE, on the trace that knows the row subset, rather than by the post-hoc interactivity pass.
        That pass keys a panel's hovertext by subplot axis and attaches it only when a trace's point count
        exactly equals ``len(sup)``, so every scatter panel that downsampled or split into strong/weak traces
        silently lost it -- and the per-point DENOMINATOR is precisely what the spec field exists to carry
        ("without it a rate computed from 3 rows renders identically to one from 300k"). The fallback was the
        generic axis-name template, so the tooltip still looked plausible and nothing signalled the loss.
        """
        ht = getattr(p, "hovertext", None)
        if not ht:
            return {}
        arr = np.asarray(ht, dtype=object)
        if arr.shape[0] != len(np.asarray(p.x)):
            return {}
        if idx is not None:
            arr = arr[idx]
        if mask is not None:
            arr = arr[mask]
        if arr.shape[0] != (len(x) if mask is None else int(np.count_nonzero(mask))):
            return {}
        return dict(hovertext=[str(v) for v in arr], hoverinfo="text")

    if weak.any() and (~weak).any():
        strong = ~weak
        fig.add_trace(
            trace_cls(x=x[strong], y=y[strong], mode="markers", marker=_sel_marker(strong),
                      error_y=_sel_err(error_y, strong), error_x=_sel_err(error_x, strong),
                      name=p.legend_label or "", showlegend=bool(p.legend_label), **_sel_hover(strong)),
            row=row, col=col,
        )
        _weak_marker = _sel_marker(weak)
        _weak_marker.pop("color", None)
        _weak_marker.pop("colorscale", None)
        _weak_marker.pop("colorbar", None)
        _weak_marker.pop("showscale", None)
        _weak_marker.pop("cmin", None)
        _weak_marker.pop("cmax", None)
        _weak_marker["color"] = "rgba(0,0,0,0)"
        _weak_marker["line"] = dict(color="#8c8c8c", width=1.2)
        fig.add_trace(
            trace_cls(x=x[weak], y=y[weak], mode="markers", marker=_weak_marker,
                      error_y=_sel_err(error_y, weak, muted=True), error_x=_sel_err(error_x, weak, muted=True),
                      name="too few rows to read", showlegend=True, **_sel_hover(weak)),
            row=row, col=col,
        )
    else:
        fig.add_trace(
            trace_cls(x=x, y=y,
                      mode="markers",
                      marker=marker,
                      error_y=error_y, error_x=error_x,
                      name=p.legend_label or "",
                      showlegend=bool(p.legend_label), **_sel_hover()),
            row=row, col=col,
        )

    # Emphasised subset (worst-K errors): resolve indices against the ORIGINAL arrays (pre-downsample).
    if p.highlight_indices is not None:
        hi_idx = np.asarray(p.highlight_indices, dtype=np.int64)
        ox, oy = np.asarray(p.x), np.asarray(p.y)
        hi_idx = hi_idx[(hi_idx >= 0) & (hi_idx < len(ox))]
        if hi_idx.size:
            fig.add_trace(
                go.Scatter(x=ox[hi_idx], y=oy[hi_idx], mode="markers",
                           marker=dict(symbol="circle-open", size=12,
                                       line=dict(color=p.highlight_color, width=2)),
                           name="worst-K", showlegend=True),
                row=row, col=col,
            )

    if p.trend_line is not None and n > 1:
        from mlframe.reporting.renderers._trend import robust_fit_endpoints
        ends = robust_fit_endpoints(np.asarray(p.x), np.asarray(p.y), p.trend_line)
        if ends is not None:
            (tx0, ty0), (tx1, ty1) = ends
            fig.add_trace(
                go.Scatter(x=[tx0, tx1], y=[ty0, ty1], mode="lines",
                           line=dict(color=TREND_LINE, width=2),
                           name=f"robust fit ({p.trend_line})", showlegend=True),
                row=row, col=col,
            )

    if p.overlay_band is not None:
        bx, blo, bhi = (np.asarray(a) for a in p.overlay_band)
        fig.add_trace(
            go.Scatter(x=bx, y=blo, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(x=bx, y=bhi, mode="lines", line=dict(width=0),
                       fill="tonexty", fillcolor="rgba(128,0,128,0.18)",
                       name="curve 95% band", showlegend=True, hoverinfo="skip"),
            row=row, col=col,
        )

    if p.overlay_line is not None:
        ox_grid, oy_grid, olabel = p.overlay_line
        fig.add_trace(
            go.Scatter(x=np.asarray(ox_grid), y=np.asarray(oy_grid), mode="lines", line=dict(color=OVERLAY_LINE, width=2), name=olabel, showlegend=True),
            row=row,
            col=col,
        )

    if p.perfect_fit_line and n > 0:
        # Span the y=x line over the UNION of both axes so it stays the true diagonal even when prediction
        # collapse (constant y) makes the y-range a single point; scaleanchor squares the panel so y=x is 45deg.
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        fig.add_trace(
            go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=PERFECT_FIT_LINE, dash="dash"), name="Perfect fit", showlegend=True),
            row=row,
            col=col,
        )
        y_range = list(p.ylim) if p.ylim is not None else [lo, hi]
        x_range = list(p.xlim) if p.xlim is not None else [lo, hi]
        if p.equal_aspect:
            # Square the panel so y=x is 45deg; probability-vs-probability (calibration) skips this so the panel
            # fills its cell width and aligns with the population histogram below.
            fig.update_yaxes(scaleanchor=_axis_ref(fig, row, col), scaleratio=1.0, row=row, col=col)
        fig.update_yaxes(range=y_range, row=row, col=col)
        fig.update_xaxes(range=x_range, row=row, col=col)
    else:
        if p.equal_aspect:
            fig.update_yaxes(scaleanchor=_axis_ref(fig, row, col), scaleratio=1.0, row=row, col=col)
        if p.xlim is not None:
            fig.update_xaxes(range=list(p.xlim), row=row, col=col)
        if p.ylim is not None:
            fig.update_yaxes(range=list(p.ylim), row=row, col=col)

    fig.update_xaxes(title_text=p.xlabel, row=row, col=col, showgrid=p.grid)
    fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid)


__all__ = ["_scatter"]
