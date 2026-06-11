"""plotly renderer.

Builds a plotly ``go.Figure`` from a ``FigureSpec``. Multi-panel figures
use ``plotly.subplots.make_subplots`` with row_heights / column_widths
matching the matplotlib gridspec.

Save formats:
- ``html``: ``write_html`` (interactive, includes plotly.js)
- ``json``: ``to_json`` (data + layout, embed-friendly)
- ``png/svg/pdf``: ``write_image`` (requires kaleido package; falls back to
  html with WARN if missing)
"""

from __future__ import annotations

import logging
import math
from typing import Any, List

import numpy as np

from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, ConfusionMarginsPanelSpec, FigureSpec,
    HeatmapPanelSpec, HistogramPanelSpec, LinePanelSpec, NetworkPanelSpec,
    ScatterPanelSpec, ViolinPanelSpec,
)

# Kaleido lifecycle + static-image write plumbing lives in the sibling module; re-exported here so
# ``from mlframe.reporting.renderers.plotly import get_kaleido_oneshot_stats`` (and the recovery-test
# imports of ``_restart_kaleido_server`` etc.) keep resolving from the same place.
from ._kaleido import (  # noqa: F401
    _ensure_kaleido_server_started,
    _is_kaleido_persistent_burned,
    _mark_kaleido_persistent_burned,
    _record_kaleido_persistent_failure,
    _restart_kaleido_server,
    get_kaleido_oneshot_stats,
    record_kaleido_oneshot_call,
    reset_kaleido_oneshot_stats,
    write_image_via_kaleido,
)
from ._plotly_interactivity import apply_interactivity, html_config

logger = logging.getLogger(__name__)

# Renderer-level safety nets for specs carrying raw large-n data. Builders are expected to
# pre-sample / pre-bin, but the renderer is public API: above these thresholds a raw spec would
# embed n values into the HTML (37 MB / 73 MB per panel at 2M, browser-freezing).
_HIST_PREBIN_THRESHOLD = 50_000
_SCATTER_MAX_POINTS = 50_000
# WebGL traces render large scatters orders of magnitude faster than SVG-mode go.Scatter.
_SCATTER_WEBGL_THRESHOLD = 10_000
_SCATTER_DOWNSAMPLE_WARNED = False
# Above this many heatmap cells, per-cell text is unreadable soup AND the plotly add_annotation loop (one layout
# copy per cell) stalls; skip the text past it (matches the matplotlib renderer cap).
_HEATMAP_CELL_TEXT_MAX = 400


def _finite_range(mat):
    """``(vmin, vmax)`` over finite entries, or ``None`` when the matrix is empty / all non-finite."""
    a = np.asarray(mat, dtype=float)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    return float(finite.min()), float(finite.max())


def _warn_scatter_downsample(n: int) -> None:
    global _SCATTER_DOWNSAMPLE_WARNED
    if not _SCATTER_DOWNSAMPLE_WARNED:
        logger.warning(
            "[plotly-render] scatter panel carries %d raw points; downsampled to %d "
            "(extremes preserved) to keep the figure responsive. Pre-sample at the spec "
            "builder to silence this. Fires once per process.",
            n, _SCATTER_MAX_POINTS,
        )
        _SCATTER_DOWNSAMPLE_WARNED = True


def _per_series_flags(flag, n: int):
    """Normalize a per-series bool flag (single bool / tuple / None) into a length-n bool list."""
    if flag is None:
        return [False] * n
    if isinstance(flag, (tuple, list, np.ndarray)):
        seq = list(flag)
        return [bool(seq[i]) if i < len(seq) else False for i in range(n)]
    return [bool(flag)] * n


def _line_uses_secondary_y(p) -> bool:
    n = len(p.y) if isinstance(p.y, tuple) else 1
    return any(_per_series_flags(p.secondary_y, n))


def _err_to_plotly(err):
    """Spec error-bar field -> plotly ``error_y`` / ``error_x`` dict (data mode, asymmetric where a pair is given)."""
    if err is None:
        return None
    if isinstance(err, tuple):
        return dict(type="data", symmetric=False,
                    array=np.asarray(err[1], dtype=float), arrayminus=np.asarray(err[0], dtype=float),
                    visible=True)
    return dict(type="data", symmetric=True, array=np.asarray(err, dtype=float), visible=True)


class PlotlyRenderer:
    backend = "plotly"

    def render(self, spec: FigureSpec, *, static_legend: bool = False) -> Any:
        """Build a plotly figure from the spec.

        ``static_legend`` enables a figure-level legend. The interactive HTML output identifies series via
        hover tooltips, so legends stay off there; a STATIC export (png/svg/pdf) has no hover, so when the
        save-format set includes one the caller passes ``static_legend=True`` to make the export readable.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        rows = len(spec.panels)
        cols = max((len(r) for r in spec.panels), default=0)
        if rows == 0 or cols == 0:
            raise ValueError("FigureSpec has no panels")

        # Per-panel subplot spec: heatmap needs no shared axes; default ``xy`` works for everything else. A line
        # panel that requests a secondary y-axis must declare ``secondary_y=True`` at subplot-creation time (plotly
        # can't add a right axis after the grid is built), so detect that here.
        sub_specs: List[List[dict]] = []
        for r, row in enumerate(spec.panels):
            row_specs: List[dict] = []
            for c in range(cols):
                if c >= len(row) or row[c] is None:
                    row_specs.append({})  # empty cell
                else:
                    cell = {"type": "xy"}
                    if isinstance(row[c], LinePanelSpec) and _line_uses_secondary_y(row[c]):
                        cell["secondary_y"] = True
                    row_specs.append(cell)
            sub_specs.append(row_specs)

        # Build subplot titles list (row-major). Plotly's subplot_titles
        # are HTML annotations — newlines need ``<br>`` instead of ``\n``,
        # otherwise the linebreak character is dropped and 2-line panel
        # titles render as one long string that bleeds into adjacent
        # subplots horizontally (visible bug 2026-05-09 on regression
        # chart with hypothesis + heteroscedasticity multiline titles).
        subplot_titles = []
        for row in spec.panels:
            for c in range(cols):
                if c >= len(row) or row[c] is None:
                    subplot_titles.append("")
                else:
                    raw_title = getattr(row[c], "title", "") or ""
                    subplot_titles.append(raw_title.replace("\n", "<br>"))

        subplots_kwargs = dict(
            rows=rows, cols=cols,
            specs=sub_specs,
            subplot_titles=subplot_titles,
            shared_xaxes=spec.sharex,
            shared_yaxes=spec.sharey,
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )
        if spec.row_height_ratios is not None:
            total = sum(spec.row_height_ratios)
            subplots_kwargs["row_heights"] = [r / total for r in spec.row_height_ratios]
        if spec.col_width_ratios is not None:
            total = sum(spec.col_width_ratios)
            subplots_kwargs["column_widths"] = [c / total for c in spec.col_width_ratios]

        fig = make_subplots(**subplots_kwargs)

        # 2026-05-09: shrink subplot-title font from plotly default (16)
        # to match matplotlib's ~11. With three columns at the typical
        # (18in, 4in) figsize, default-16 titles overflow horizontally
        # into adjacent subplots (e.g. regression chart left-panel title
        # "MAE=... R2=..." bleeds into middle-panel "Residuals (skew=...)").
        for ann in fig.layout.annotations:
            ann.font = dict(size=11)

        for r, row in enumerate(spec.panels, start=1):
            for c in range(1, cols + 1):
                if c - 1 >= len(row) or row[c - 1] is None:
                    continue
                self._render_panel(fig, row[c - 1], r, c)

        # Figure-level layout.
        if spec.suptitle:
            fig.update_layout(title=dict(text=spec.suptitle,
                                         font=dict(size=spec.suptitle_fontsize)))
        fig.update_layout(
            width=int(spec.figsize[0] * 80),   # ~80px per matplotlib inch
            height=int(spec.figsize[1] * 80),
            margin=dict(l=60, r=40, t=80 if spec.suptitle else 50, b=50),
            # Interactive HTML identifies series via hover tooltips, so the legend defaults off there to avoid
            # the multi-subplot soup (precision/recall/F1 mixed with reliability lines). A static export has no
            # hover; the caller flips ``static_legend`` when a png/svg/pdf is in the save set so it stays readable.
            showlegend=static_legend,
        )
        if static_legend:
            fig.update_layout(legend=dict(font=dict(size=9), itemsizing="constant",
                                          bgcolor="rgba(255,255,255,0.6)"))
        apply_interactivity(fig, spec, static_legend=static_legend)
        return fig

    def save(self, fig: Any, path: str, fmt: str) -> None:
        fmt = fmt.lower()
        if fmt == "html":
            fig.write_html(path, include_plotlyjs="cdn", auto_open=False, config=html_config())
        elif fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                f.write(fig.to_json())
        elif fmt in ("png", "svg", "pdf"):
            write_image_via_kaleido(fig, path, fmt)
        else:
            raise ValueError(
                f"plotly doesn't support format {fmt!r}; "
                "supported: html/png/svg/pdf/json"
            )

    def show(self, fig: Any) -> None:
        try:
            fig.show()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Per-panel dispatch
    # ------------------------------------------------------------------

    def _render_panel(self, fig, panel, row: int, col: int) -> None:
        if isinstance(panel, ScatterPanelSpec):
            self._scatter(fig, panel, row, col)
        elif isinstance(panel, HistogramPanelSpec):
            self._histogram(fig, panel, row, col)
        elif isinstance(panel, HeatmapPanelSpec):
            self._heatmap(fig, panel, row, col)
        elif isinstance(panel, ConfusionMarginsPanelSpec):
            self._confusion_margins(fig, panel, row, col)
        elif isinstance(panel, BarPanelSpec):
            self._bar(fig, panel, row, col)
        elif isinstance(panel, LinePanelSpec):
            self._line(fig, panel, row, col)
        elif isinstance(panel, ViolinPanelSpec):
            self._violin(fig, panel, row, col)
        elif isinstance(panel, NetworkPanelSpec):
            self._network(fig, panel, row, col)
        elif isinstance(panel, AnnotationPanelSpec):
            self._annotation(fig, panel, row, col)
        else:
            raise TypeError(f"unknown panel type: {type(panel).__name__}")

    def _annotation(self, fig, p: AnnotationPanelSpec, row: int, col: int) -> None:
        fig.add_annotation(text=p.text.replace("\n", "<br>"), x=0.5, y=0.5,
                           xref="x domain", yref="y domain", showarrow=False,
                           font=dict(size=p.fontsize), align="center",
                           row=row, col=col)
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    def _scatter(self, fig, p: ScatterPanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go

        x = np.asarray(p.x)
        y = np.asarray(p.y)
        n = len(x)

        # Per-point size / color arrays must follow the SAME row subset as x/y when downsampling.
        size_arr = p.point_size if isinstance(p.point_size, np.ndarray) else None
        color_arr = p.point_color if isinstance(p.point_color, np.ndarray) else None

        if n > _SCATTER_MAX_POINTS:
            _warn_scatter_downsample(n)
            from mlframe.reporting.charts._sampling import subsample_preserving_extremes
            idx = subsample_preserving_extremes(x, y, sample_size=_SCATTER_MAX_POINTS)
            x, y = x[idx], y[idx]
            if size_arr is not None and len(size_arr) == n:
                size_arr = size_arr[idx]
            if color_arr is not None and len(color_arr) == n:
                color_arr = color_arr[idx]

        marker = dict(opacity=p.point_alpha)
        # ScatterPanelSpec.point_size follows matplotlib's ``s=`` (area in pt^2); plotly marker.size is the
        # DIAMETER in px. Without conversion large mpl areas blow up to giant circles and the auto-axis range
        # goes haywire. Conversion: plotly_diameter_px = sqrt(mpl_area_pt2) * 1.33.
        if size_arr is not None:
            marker["size"] = (np.sqrt(np.maximum(np.asarray(size_arr, dtype=float), 0.0)) * 1.33).tolist()
        else:
            marker["size"] = float(math.sqrt(max(float(p.point_size), 0.0)) * 1.33)
        if color_arr is not None:
            marker["color"] = np.asarray(color_arr)
            marker["colorscale"] = _mpl_to_plotly_cmap(p.colormap)
            marker["showscale"] = bool(p.colorbar_label)
            if p.colorbar_label:
                marker["colorbar"] = dict(title=p.colorbar_label)
        elif p.point_color is not None:
            marker["color"] = p.point_color

        # Hover labels for inline_labels (population annotations). Only valid when no downsample reordered rows.
        text = None
        if p.inline_labels and len(p.inline_labels) == n and n <= _SCATTER_MAX_POINTS:
            text = [t[2] for t in p.inline_labels]

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
        fig.add_trace(
            trace_cls(x=x, y=y,
                      mode="markers+text" if text else "markers",
                      marker=marker,
                      error_y=error_y, error_x=error_x,
                      text=text,
                      textposition="top center" if text else None,
                      textfont=dict(size=8),
                      name=p.legend_label or "",
                      showlegend=bool(p.legend_label)),
            row=row, col=col,
        )

        # Emphasised subset (worst-K errors): resolve indices against the ORIGINAL arrays (pre-downsample).
        if p.highlight_indices is not None:
            hi = np.asarray(p.highlight_indices, dtype=np.int64)
            ox, oy = np.asarray(p.x), np.asarray(p.y)
            hi = hi[(hi >= 0) & (hi < len(ox))]
            if hi.size:
                fig.add_trace(
                    go.Scatter(x=ox[hi], y=oy[hi], mode="markers",
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
                               line=dict(color="darkorange", width=2),
                               name=f"robust fit ({p.trend_line})", showlegend=True),
                    row=row, col=col,
                )

        if p.overlay_band is not None:
            bx, blo, bhi = (np.asarray(a) for a in p.overlay_band)
            fig.add_trace(
                go.Scatter(x=bx, y=blo, mode="lines", line=dict(width=0),
                           showlegend=False, hoverinfo="skip"),
                row=row, col=col,
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
                go.Scatter(x=np.asarray(ox_grid), y=np.asarray(oy_grid), mode="lines",
                           line=dict(color="purple", width=2),
                           name=olabel, showlegend=True),
                row=row, col=col,
            )

        if p.perfect_fit_line and n > 0:
            # Span the y=x line over the UNION of both axes so it stays the true diagonal even when prediction
            # collapse (constant y) makes the y-range a single point; scaleanchor squares the panel so y=x is 45deg.
            lo = float(min(np.min(x), np.min(y)))
            hi = float(max(np.max(x), np.max(y)))
            fig.add_trace(
                go.Scatter(x=[lo, hi], y=[lo, hi],
                           mode="lines",
                           line=dict(color="green", dash="dash"),
                           name="Perfect fit", showlegend=True),
                row=row, col=col,
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
            if p.xlim is not None:
                fig.update_xaxes(range=list(p.xlim), row=row, col=col)
            if p.ylim is not None:
                fig.update_yaxes(range=list(p.ylim), row=row, col=col)

        fig.update_xaxes(title_text=p.xlabel, row=row, col=col, showgrid=p.grid)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid)

    def _histogram(self, fig, p: HistogramPanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go

        # ``overlay_x_lo/hi`` anchors the Normal-overlay grid. When we pre-bin (here or upstream) they come from
        # the bin EDGES, avoiding two extra full-n min/max passes over raw values (PERF-18).
        overlay_x_lo = overlay_x_hi = None
        bin_centers = p.bin_centers
        heights = None
        if bin_centers is None and len(np.asarray(p.values)) > _HIST_PREBIN_THRESHOLD:
            # Raw spec with n above the embed-hazard ceiling: bin once with numpy instead of shipping n values
            # into the HTML (37 MB / browser-freezing at 2M).
            from mlframe.reporting.charts._sampling import prebin_histogram
            heights, centers, width0 = prebin_histogram(np.asarray(p.values), p.bins, p.density)
            if heights is not None:
                bin_centers = centers

        if bin_centers is not None:
            if heights is None:
                heights = np.asarray(p.values)
                width = float(p.bin_width or (
                    (bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 1.0))
            else:
                width = float(width0)
            colors_kw = dict(color=p.color)
            if p.bar_colors is not None:
                _h_min = float(np.min(p.bar_colors))
                _h_max = float(np.max(p.bar_colors))
                if _h_max <= _h_min:
                    _h_max = _h_min + 1.0
                colors_kw = dict(
                    color=np.asarray(p.bar_colors),
                    colorscale=_mpl_to_plotly_cmap(p.colormap),
                )
            fig.add_trace(
                go.Bar(x=np.asarray(bin_centers), y=np.asarray(heights),
                       width=width,
                       marker=dict(line=dict(color="white", width=0.5), **colors_kw),
                       showlegend=False),
                row=row, col=col,
            )
            if len(bin_centers) > 0:
                overlay_x_lo = float(bin_centers[0] - width / 2.0)
                overlay_x_hi = float(bin_centers[-1] + width / 2.0)
        else:
            fig.add_trace(
                go.Histogram(x=np.asarray(p.values),
                             nbinsx=p.bins,
                             histnorm="probability density" if p.density else "",
                             marker=dict(color=p.color, line=dict(color="white", width=0.4)),
                             opacity=0.6, showlegend=False),
                row=row, col=col,
            )

        if p.overlay_normal is not None:
            mu, sigma = p.overlay_normal
            if sigma > 0:
                if overlay_x_lo is None:
                    vals = np.asarray(p.values)
                    overlay_x_lo, overlay_x_hi = float(np.min(vals)), float(np.max(vals))
                x_grid = np.linspace(overlay_x_lo, overlay_x_hi, 200)
                normal_pdf = (
                    1 / (sigma * np.sqrt(2 * np.pi))
                    * np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)
                )
                label = p.overlay_label or f"Normal(mu={mu:.2g}, sigma={sigma:.2g})"
                fig.add_trace(
                    go.Scatter(x=x_grid, y=normal_pdf,
                               mode="lines",
                               line=dict(color="red", dash="dash", width=1.4),
                               name=label, showlegend=True),
                    row=row, col=col,
                )

        if p.xlim is not None:
            fig.update_xaxes(range=list(p.xlim), row=row, col=col)
        fig.update_xaxes(title_text=p.xlabel, row=row, col=col, showgrid=p.grid)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid,
                         type="log" if p.yscale == "log" else "linear")

    def _confusion_margins(self, fig, p: ConfusionMarginsPanelSpec, row: int, col: int) -> None:
        # Plotly subplot cells cannot host nested marginal axes the way the matplotlib subgridspec does, so the
        # margins are folded into the axis tick labels: each predicted-class column header carries its volume and
        # each true-class row header its support. The heatmap itself reuses the HeatmapPanelSpec renderer.
        col_labels = tuple(f"{lab}<br>(vol={int(v)})" for lab, v in zip(p.col_labels, np.asarray(p.col_margin)))
        row_labels = tuple(f"{lab} (n={int(v)})" for lab, v in zip(p.row_labels, np.asarray(p.row_margin)))
        title = p.title if not p.note else f"{p.title} -- {p.note}"
        heat = HeatmapPanelSpec(
            matrix=p.matrix, row_labels=row_labels, col_labels=col_labels,
            title=title, xlabel=p.xlabel, ylabel=p.ylabel, colormap=p.colormap,
            cell_text=p.cell_text, text_format=p.text_format, colorbar_label=p.colorbar_label,
        )
        self._heatmap(fig, heat, row, col)

    def _heatmap(self, fig, p: HeatmapPanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go
        from mlframe.reporting.colors import resolve_heatmap_cmap
        cmap_name = resolve_heatmap_cmap(p.colormap)

        fig.add_trace(
            go.Heatmap(z=p.matrix.tolist(),
                       x=list(p.col_labels), y=list(p.row_labels),
                       colorscale=_mpl_to_plotly_cmap(cmap_name),
                       colorbar=dict(title=p.colorbar_label) if p.colorbar_label else None,
                       showscale=True),
            row=row, col=col,
        )
        # 2026-05-09: per-cell text via add_annotation instead of
        # plotly's built-in ``text`` + ``texttemplate`` (which uses
        # one global font color and produces white-on-yellow
        # invisibility on viridis high-end / RdYlBu high-end). Per-cell
        # ``auto_text_color`` flips by perceived luminance.
        # Skip per-cell text on an empty / all-non-finite matrix (nanmin raises / poisons the color scale) or a
        # huge grid where the per-annotation O(cells) plotly layout copy stalls and the text is unreadable soup anyway.
        rng = _finite_range(p.matrix)
        if p.cell_text is not None and rng is not None and p.matrix.size <= _HEATMAP_CELL_TEXT_MAX:
            from mlframe.reporting.colors import auto_text_color
            mat = p.matrix
            vmin, vmax = rng
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    cell = float(mat[i, j])
                    text_color = auto_text_color(cell if np.isfinite(cell) else vmin, cmap_name, vmin=vmin, vmax=vmax)
                    fig.add_annotation(
                        text=format(p.cell_text[i, j], p.text_format),
                        x=p.col_labels[j], y=p.row_labels[i],
                        showarrow=False,
                        font=dict(color=text_color, size=10),
                        row=row, col=col,
                    )
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
            ends = robust_fit_endpoints(p.trend_xy[0], p.trend_xy[1], p.trend_line)
            if ends is not None:
                (tx0, ty0), (tx1, ty1) = ends
                fig.add_trace(
                    go.Scatter(x=[tx0, tx1], y=[ty0, ty1], mode="lines",
                               line=dict(color="darkorange", width=2),
                               name=f"robust fit ({p.trend_line})", showlegend=True),
                    row=row, col=col,
                )

        fig.update_xaxes(title_text=p.xlabel, row=row, col=col,
                         tickangle=-45)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col,
                         autorange="reversed")  # match matplotlib top-to-bottom row order

    def _bar(self, fig, p: BarPanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go

        from mlframe.reporting.colors import line_color
        horizontal = p.orientation == "horizontal"
        cats = list(p.categories)

        def _add_bar(values, color, label, show):
            if horizontal:
                # Categories on y, values on x; reverse so the first category sits on top (worst-first reads down).
                fig.add_trace(
                    go.Bar(y=cats, x=np.asarray(values).tolist(), orientation="h",
                           name=label, showlegend=show, marker=dict(color=color)),
                    row=row, col=col,
                )
            else:
                fig.add_trace(
                    go.Bar(x=cats, y=np.asarray(values).tolist(),
                           name=label, showlegend=show, marker=dict(color=color)),
                    row=row, col=col,
                )

        if isinstance(p.values, tuple):
            for i, series in enumerate(p.values):
                lbl = p.series_labels[i] if p.series_labels else f"series {i}"
                # plotly's default qualitative palette clashes with matplotlib's tab10 in the same figure; fall
                # back to ``line_color(i)`` (tab10) when the spec doesn't pin colors for cross-backend parity.
                color = p.colors[i] if (p.colors is not None and i < len(p.colors)) else line_color(i)
                _add_bar(series, color, lbl, True)
            fig.update_layout(barmode="group")
        else:
            _add_bar(p.values, p.colors[0] if p.colors else "steelblue", "", False)

        # Reference line perpendicular to the bars (global metric). vline for horizontal bars (value axis is x),
        # hline for vertical bars (value axis is y).
        if p.hline is not None:
            hval, hcolor, hlabel = p.hline
            line_kw = dict(line=dict(color=hcolor, dash="dash", width=1.3),
                           annotation_text=hlabel or None, annotation_position="top right",
                           row=row, col=col)
            if horizontal:
                fig.add_vline(x=hval, **line_kw)
            else:
                fig.add_hline(y=hval, **line_kw)

        if horizontal:
            fig.update_yaxes(autorange="reversed", row=row, col=col)
            fig.update_xaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid)
            fig.update_yaxes(title_text=p.xlabel, row=row, col=col)
        else:
            fig.update_xaxes(title_text=p.xlabel, row=row, col=col,
                             tickangle=-p.xtick_rotation if p.xtick_rotation else 0)
            fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid)

    def _line(self, fig, p: LinePanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go
        from mlframe.reporting.colors import line_color

        ys = p.y if isinstance(p.y, tuple) else (p.y,)
        xs_per_series = isinstance(p.x, tuple)
        labels = p.series_labels or (None,) * len(ys)
        styles = p.line_styles or ("-",) * len(ys)
        cols = p.colors or tuple(line_color(i) for i in range(len(ys)))
        sec = _per_series_flags(p.secondary_y, len(ys))
        fills = _per_series_flags(p.fill_to_baseline, len(ys))
        has_secondary = any(sec)
        # matplotlib linestyle tokens -> plotly dash; "markers" / "lines+markers" select the trace mode.
        _STYLE_MAP = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}

        def _xi(i):
            v = p.x[i] if xs_per_series else p.x
            return np.asarray(v) if isinstance(v, np.ndarray) else v

        if p.band is not None:
            x0 = _xi(0)
            lower, upper = np.asarray(p.band[0]), np.asarray(p.band[1])
            band_color = p.band_color or cols[0]
            fig.add_trace(
                go.Scatter(x=np.concatenate([x0, x0[::-1]]),
                           y=np.concatenate([upper, lower[::-1]]),
                           fill="toself", fillcolor=_rgba(band_color, 0.2),
                           line=dict(width=0), hoverinfo="skip",
                           name=p.band_label or "band", showlegend=bool(p.band_label)),
                row=row, col=col,
            )

        for i, y in enumerate(ys):
            token = styles[i % len(styles)]
            if token == "markers":
                mode, dash = "markers", "solid"
            elif token == "lines+markers":
                mode, dash = "lines+markers", "solid"
            else:
                mode, dash = "lines", _STYLE_MAP.get(token, "solid")
            yv = np.asarray(y) if isinstance(y, np.ndarray) else y
            # Area fill under the curve to the panel baseline; "tozeroy" when baseline is 0, else explicit.
            trace_kw = {}
            if fills[i]:
                trace_kw["fill"] = "tozeroy" if p.fill_baseline == 0.0 else "tonexty"
                trace_kw["fillcolor"] = _rgba(cols[i % len(cols)], 0.2)
                if p.step_fill:
                    trace_kw["line_shape"] = "hv"
            sec_kw = {"secondary_y": sec[i]} if has_secondary else {}
            fig.add_trace(
                go.Scatter(x=_xi(i), y=yv,
                           mode=mode,
                           line=dict(color=cols[i % len(cols)], dash=dash),
                           marker=dict(color=cols[i % len(cols)], size=5),
                           name=labels[i] if i < len(labels) else None,
                           showlegend=any(labels),
                           **trace_kw),
                row=row, col=col, **sec_kw,
            )

        for span in (p.vspans or ()):
            vx0, vx1, vcolor, valpha = span[0], span[1], span[2], span[3]
            vlabel = span[4] if len(span) > 4 else ""
            fig.add_vrect(x0=vx0, x1=vx1, fillcolor=_rgba(vcolor, valpha),
                          line_width=0, layer="below", row=row, col=col)
            if vlabel:
                # No native per-vrect legend in plotly; add an invisible scatter proxy so the regime label shows.
                fig.add_trace(
                    go.Scatter(x=[None], y=[None], mode="markers",
                               marker=dict(size=8, color=_rgba(vcolor, max(valpha, 0.3)), symbol="square"),
                               name=vlabel, showlegend=True),
                    row=row, col=col,
                )
        for vx, vcolor, vlabel in (p.vlines or ()):
            # add_vline does arithmetic on x that raises on a datetime axis; a line-shape with the x in data coords
            # and y spanning the panel's y-domain works on numeric AND datetime axes alike (G4).
            self._add_vline_datetime_safe(fig, vx, vcolor, vlabel, row, col)

        _MARKER_MAP = {"*": "star", "D": "diamond", "o": "circle", "s": "square", "^": "triangle-up"}
        for mx, my, mlabel, mcolor, msym in (p.point_markers or ()):
            fig.add_trace(
                go.Scatter(x=[mx], y=[my], mode="markers+text",
                           marker=dict(color=mcolor, size=13, symbol=_MARKER_MAP.get(msym, "star"),
                                       line=dict(color="black", width=0.6)),
                           text=[mlabel or ""], textposition="bottom right", textfont=dict(size=8),
                           name=mlabel or None, showlegend=bool(mlabel)),
                row=row, col=col,
            )

        fig.update_xaxes(title_text=p.xlabel, row=row, col=col, showgrid=p.grid,
                         tickangle=-30 if p.x_is_time else 0)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid,
                         secondary_y=False)
        if has_secondary:
            fig.update_yaxes(title_text=p.secondary_ylabel, row=row, col=col,
                             secondary_y=True, showgrid=False)

    @staticmethod
    def _is_datetime_like(v) -> bool:
        import datetime as _dt
        if isinstance(v, (np.datetime64,)):
            return True
        if isinstance(v, (_dt.datetime, _dt.date)):
            return True
        return False

    def _add_vline_datetime_safe(self, fig, vx, vcolor, vlabel, row: int, col: int) -> None:
        """Vertical reference line that works on numeric AND datetime x-axes.

        ``fig.add_vline`` computes ``x1 - x0`` internally, which raises ``TypeError`` on datetime x. For datetime
        markers we instead add a line shape with the x in data coords and y spanning the subplot's y-domain (the
        temporal change-point markers that previously fell back to vspans now render as true vlines)."""
        if self._is_datetime_like(vx):
            import pandas as pd
            x_coord = pd.Timestamp(vx) if not isinstance(vx, np.datetime64) else pd.Timestamp(vx)
            fig.add_shape(type="line", x0=x_coord, x1=x_coord, y0=0, y1=1,
                          yref="y domain", xref="x",
                          line=dict(color=vcolor, dash="dot", width=1.2),
                          row=row, col=col)
            if vlabel:
                fig.add_annotation(x=x_coord, y=1, yref="y domain", yanchor="bottom",
                                   text=vlabel, showarrow=False, font=dict(size=9),
                                   row=row, col=col)
        else:
            fig.add_vline(x=vx, line=dict(color=vcolor, dash="dot", width=1.2),
                          annotation_text=vlabel or None, annotation_position="top",
                          row=row, col=col)

    def _violin(self, fig, p: ViolinPanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go
        from mlframe.reporting.colors import line_color

        for i, group in enumerate(p.groups):
            # tab10 cycle for cross-backend parity (plotly default
            # 'Plotly' qualitative is over-saturated next to mpl bars).
            color = line_color(i)
            fig.add_trace(
                go.Violin(y=np.asarray(group).tolist(),
                          name=p.group_labels[i],
                          box_visible=p.show_box,
                          meanline_visible=False,
                          line_color=color,
                          fillcolor=color,
                          opacity=0.6,
                          showlegend=False),
                row=row, col=col,
            )
        fig.update_xaxes(title_text=p.xlabel, row=row, col=col,
                         tickangle=-30)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid)

    # Cap on directed-edge arrow annotations. Each arrow is one layout
    # annotation; beyond this the topology still renders (lines + nodes) but
    # arrowheads are skipped so a large opt-in graph doesn't bloat the layout.
    _NETWORK_MAX_ARROWS = 500

    def _network(self, fig, p: NetworkPanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go

        node_x = np.asarray(p.node_x, dtype=float)
        node_y = np.asarray(p.node_y, dtype=float)
        e_src = np.asarray(p.edge_src, dtype=np.int64)
        e_dst = np.asarray(p.edge_dst, dtype=np.int64)
        weights = np.asarray(p.edge_weight, dtype=float)

        if e_src.size:
            wmin, wmax = float(weights.min()), float(weights.max())
            wspan = (wmax - wmin) or 1.0
            lo, hi = p.edge_width_range
            colorscale = _mpl_to_plotly_cmap(p.colormap)
            # Bin edges by MI into a handful of width/color buckets: one Scattergl
            # line trace per non-empty bucket keeps trace count O(bins) regardless
            # of edge count (a single line trace can't vary width/color per segment).
            n_bins = min(8, max(1, e_src.size))
            bin_idx = np.minimum(((weights - wmin) / wspan * n_bins).astype(int), n_bins - 1)
            from plotly.colors import sample_colorscale
            for b in range(n_bins):
                mask = bin_idx == b
                if not mask.any():
                    continue
                frac = (b + 0.5) / n_bins
                width = lo + frac * (hi - lo)
                color = sample_colorscale(colorscale, [frac])[0]
                xs: List = []
                ys: List = []
                for a, d in zip(e_src[mask], e_dst[mask]):
                    xs.extend([node_x[a], node_x[d], None])
                    ys.extend([node_y[a], node_y[d], None])
                fig.add_trace(
                    go.Scattergl(x=xs, y=ys, mode="lines",
                                 line=dict(width=width, color=color),
                                 hoverinfo="skip", showlegend=False),
                    row=row, col=col,
                )

            # Invisible marker trace at edge midpoints carries the continuous MI
            # colorbar and a per-edge hover readout without cluttering the plot.
            mid_x = (node_x[e_src] + node_x[e_dst]) / 2.0
            mid_y = (node_y[e_src] + node_y[e_dst]) / 2.0
            fig.add_trace(
                go.Scattergl(
                    x=mid_x.tolist(), y=mid_y.tolist(), mode="markers",
                    marker=dict(size=0.1, color=weights.tolist(), colorscale=colorscale,
                                showscale=True,
                                colorbar=dict(title=p.colorbar_label) if p.colorbar_label else None),
                    text=[f"MI={w:.4f}" for w in weights],
                    hoverinfo="text", showlegend=False),
                row=row, col=col,
            )

            # Directed-edge arrowheads via data-space annotations. Axis refs are
            # derived from the subplot grid so multi-panel figures stay correct;
            # any failure falls back to no arrows (lines already convey topology).
            directed = p.edge_directed
            if np.isscalar(directed):
                directed = np.full(e_src.shape, bool(directed))
            else:
                directed = np.asarray(directed, dtype=bool)
            if directed.any() and int(directed.sum()) <= self._NETWORK_MAX_ARROWS:
                try:
                    n_cols = len(fig._grid_ref[0])
                    idx = (row - 1) * n_cols + col
                    suffix = "" if idx == 1 else str(idx)
                    xref, yref = f"x{suffix}", f"y{suffix}"
                    for a, d, dirn in zip(e_src, e_dst, directed):
                        if dirn:
                            fig.add_annotation(
                                x=node_x[d], y=node_y[d], ax=node_x[a], ay=node_y[a],
                                xref=xref, yref=yref, axref=xref, ayref=yref,
                                showarrow=True, arrowhead=2, arrowsize=1.2,
                                arrowwidth=1.0, arrowcolor="rgba(80,80,80,0.6)",
                                standoff=6, startstandoff=6,
                            )
                except Exception:
                    logger.debug("network arrows skipped (subplot axis-ref resolution failed)",
                                 exc_info=True)

        # Nodes: one marker trace. size follows matplotlib ``scatter(s=)`` area
        # semantics; convert to plotly's pixel diameter (sqrt(area) * 1.33).
        sizes = np.sqrt(np.maximum(np.asarray(p.node_size, dtype=float), 0.0)) * 1.33
        hovertext = list(p.node_hovertext) if p.node_hovertext else list(p.node_label)
        fig.add_trace(
            go.Scattergl(
                x=node_x.tolist(), y=node_y.tolist(),
                mode="markers+text",
                marker=dict(size=sizes.tolist(), color=list(p.node_color),
                            line=dict(width=0.5, color="black")),
                text=list(p.node_label), textposition="top center", textfont=dict(size=8),
                hovertext=hovertext, hoverinfo="text", showlegend=False),
            row=row, col=col,
        )

        fig.update_xaxes(title_text=p.xlabel, row=row, col=col,
                         showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col,
                         showgrid=False, zeroline=False, showticklabels=False)


# Map matplotlib colormap BASE names to plotly's named colorscales so the plotly backend renders the same scale as matplotlib.
# Keys are lowercased; the resolver lowercases its input and strips a trailing ``_r`` (reversed), re-appending it to the plotly name.
_MPL_TO_PLOTLY = {
    "rdylbu": "RdYlBu",
    "rdylgn": "RdYlGn",
    "rdbu": "RdBu",
    "reds": "Reds",
    "blues": "Blues",
    "greens": "Greens",
    "viridis": "Viridis",
    "plasma": "Plasma",
    "magma": "Magma",
    "inferno": "Inferno",
    "coolwarm": "RdBu_r",  # mpl coolwarm has no plotly twin; RdBu_r is the closest diverging blue-low/red-high match
}


def _axis_ref(fig, row: int, col: int) -> str:
    """x-axis reference string for the subplot at (row, col), e.g. ``"x"`` / ``"x4"`` — for scaleanchor."""
    try:
        n_cols = len(fig._grid_ref[0])
        idx = (row - 1) * n_cols + col
    except Exception:
        idx = 1
    return "x" if idx == 1 else f"x{idx}"


def _rgba(color: str, alpha: float) -> str:
    """Best-effort named/hex color -> rgba() string with the given alpha; leaves rgb()/rgba() as-is."""
    c = str(color)
    if c.startswith("rgba(") or c.startswith("rgb("):
        return c
    try:
        import matplotlib.colors as mcolors
        r, g, b = mcolors.to_rgb(c)
        return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha})"
    except Exception:
        return c


def _mpl_to_plotly_cmap(name: str) -> str:
    """Map a matplotlib colormap name to a plotly colorscale name.

    Case-insensitive (matplotlib resolves names case-insensitively); a trailing ``_r`` is stripped, the base looked up, and ``_r``
    re-appended so the plotly scale is reversed the same way matplotlib reverses. Falls back to 'Viridis' for genuinely-unknown
    names with a WARN — the goal is zero warnings for cmaps the charts actually request.
    """
    key = str(name).lower()
    reversed_suffix = ""
    if key.endswith("_r"):
        key = key[:-2]
        reversed_suffix = "_r"
    base = _MPL_TO_PLOTLY.get(key)
    if base is not None:
        # XOR the request's reversal with any reversal baked into the mapped name (e.g. coolwarm -> RdBu_r).
        want_reversed = bool(reversed_suffix) ^ base.endswith("_r")
        plain = base[:-2] if base.endswith("_r") else base
        return plain + "_r" if want_reversed else plain
    logger.warning(
        "Unknown colormap %r; falling back to plotly 'Viridis'. "
        "Add the mapping to mlframe.reporting.renderers.plotly._MPL_TO_PLOTLY "
        "to silence this warning.",
        name,
    )
    return "Viridis"


__all__ = ["PlotlyRenderer"]
