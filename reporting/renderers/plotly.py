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
from typing import Any, List

import numpy as np

from mlframe.reporting.spec import (
    BarPanelSpec, FigureSpec, HeatmapPanelSpec, HistogramPanelSpec,
    LinePanelSpec, ScatterPanelSpec, ViolinPanelSpec,
)

logger = logging.getLogger(__name__)


# Process-singleton: track whether the persistent kaleido sync server
# is up so we don't pay the ~10-15s Chromium-spawn cost on every PNG /
# SVG / PDF write. Verified empirically (2026-05-08) that kaleido 1.x
# default ``fig.write_image()`` calls spawn a fresh Chromium process
# per call (~13s each); persistent server reuses one process and drops
# subsequent calls to ~0.13s. On c0114 (lgb+xgb multiclass, 100k rows,
# 32 PNG calls), this saved 32 * ~13s = ~7 minutes of wall time.
_KALEIDO_SERVER_STARTED = False


def _ensure_kaleido_server_started() -> bool:
    """Start the kaleido sync server (idempotent). Returns True if the
    server is up after the call, False if kaleido isn't installed.
    """
    global _KALEIDO_SERVER_STARTED
    if _KALEIDO_SERVER_STARTED:
        return True
    try:
        import kaleido  # noqa: F401
    except ImportError:
        return False
    try:
        # silence_warnings=True so the "already started" message doesn't
        # spam logs if some other caller already started the server.
        kaleido.start_sync_server(silence_warnings=True)
        _KALEIDO_SERVER_STARTED = True
        # Register cleanup so the Chromium subprocess gets a clean exit
        # rather than the "Resorting to unclean kill browser" warning
        # at interpreter shutdown.
        import atexit
        def _stop():
            try:
                kaleido.stop_sync_server(silence_warnings=True)
            except Exception:
                pass
        atexit.register(_stop)
        return True
    except Exception as e:
        logger.warning(
            "Failed to start kaleido sync server (%s); will fall back to "
            "the slower per-call oneshot path.", e,
        )
        return False


class PlotlyRenderer:
    backend = "plotly"

    def render(self, spec: FigureSpec) -> Any:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        rows = len(spec.panels)
        cols = max((len(r) for r in spec.panels), default=0)
        if rows == 0 or cols == 0:
            raise ValueError("FigureSpec has no panels")

        # Per-panel subplot spec: heatmap needs no shared axes; default
        # ``xy`` works for everything else.
        sub_specs: List[List[dict]] = []
        for r, row in enumerate(spec.panels):
            row_specs: List[dict] = []
            for c in range(cols):
                if c >= len(row) or row[c] is None:
                    row_specs.append({})  # empty cell
                else:
                    row_specs.append({"type": "xy"})
            sub_specs.append(row_specs)

        # Build subplot titles list (row-major).
        subplot_titles = []
        for row in spec.panels:
            for c in range(cols):
                if c >= len(row) or row[c] is None:
                    subplot_titles.append("")
                else:
                    subplot_titles.append(getattr(row[c], "title", ""))

        subplots_kwargs = dict(
            rows=rows, cols=cols,
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
            showlegend=False,  # per-panel legend handled inline
        )
        return fig

    def save(self, fig: Any, path: str, fmt: str) -> None:
        fmt = fmt.lower()
        if fmt == "html":
            fig.write_html(path, include_plotlyjs="cdn", auto_open=False)
        elif fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                f.write(fig.to_json())
        elif fmt in ("png", "svg", "pdf"):
            try:
                # Persistent-server fast path: write_fig_sync reuses one
                # Chromium subprocess across all calls in the process,
                # dropping per-call cost from ~13s (oneshot) to ~0.13s.
                # Falls through to plotly's default write_image when the
                # sync server can't be started (kaleido missing, etc.).
                if _ensure_kaleido_server_started():
                    import kaleido as _kal
                    # kaleido infers format from path extension; pass
                    # ``opts`` only when the path-extension doesn't
                    # match the requested fmt (defensive).
                    _kal.write_fig_sync(fig, path, opts={"format": fmt})
                else:
                    fig.write_image(path, format=fmt)
            except (ImportError, ValueError) as e:
                # kaleido missing or write_image refused.
                logger.warning(
                    "plotly write_image(%s) failed (%s); falling back to .html. "
                    "Install kaleido to enable static image export: "
                    "pip install -U kaleido",
                    fmt, e,
                )
                # Same path stem, swap extension to .html.
                from os.path import splitext
                root, _ = splitext(path)
                fig.write_html(root + ".html", include_plotlyjs="cdn", auto_open=False)
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
        elif isinstance(panel, BarPanelSpec):
            self._bar(fig, panel, row, col)
        elif isinstance(panel, LinePanelSpec):
            self._line(fig, panel, row, col)
        elif isinstance(panel, ViolinPanelSpec):
            self._violin(fig, panel, row, col)
        else:
            raise TypeError(f"unknown panel type: {type(panel).__name__}")

    def _scatter(self, fig, p: ScatterPanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go

        marker = dict(opacity=p.point_alpha)
        if isinstance(p.point_size, np.ndarray):
            marker["size"] = p.point_size.tolist()
        else:
            marker["size"] = float(p.point_size)
        if isinstance(p.point_color, np.ndarray):
            marker["color"] = p.point_color.tolist()
            marker["colorscale"] = _mpl_to_plotly_cmap(p.colormap)
            marker["showscale"] = bool(p.colorbar_label)
            if p.colorbar_label:
                marker["colorbar"] = dict(title=p.colorbar_label)
        elif p.point_color is not None:
            marker["color"] = p.point_color

        # Hover labels for inline_labels (population annotations).
        text = None
        if p.inline_labels and len(p.inline_labels) == len(p.x):
            text = [t[2] for t in p.inline_labels]

        fig.add_trace(
            go.Scatter(x=p.x.tolist() if isinstance(p.x, np.ndarray) else p.x,
                       y=p.y.tolist() if isinstance(p.y, np.ndarray) else p.y,
                       mode="markers+text" if text else "markers",
                       marker=marker,
                       text=text,
                       textposition="top center" if text else None,
                       textfont=dict(size=8),
                       name=p.legend_label or "",
                       showlegend=bool(p.legend_label)),
            row=row, col=col,
        )

        if p.perfect_fit_line and len(p.x) > 0:
            xmin, xmax = float(np.min(p.x)), float(np.max(p.x))
            fig.add_trace(
                go.Scatter(x=[xmin, xmax], y=[xmin, xmax],
                           mode="lines",
                           line=dict(color="green", dash="dash"),
                           name="Perfect fit", showlegend=True),
                row=row, col=col,
            )

        fig.update_xaxes(title_text=p.xlabel, row=row, col=col, showgrid=p.grid)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid)

    def _histogram(self, fig, p: HistogramPanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go

        if p.bin_centers is not None:
            # Pre-binned: bar trace at centers.
            heights = np.asarray(p.values).tolist()
            width = p.bin_width or (
                (p.bin_centers[1] - p.bin_centers[0]) if len(p.bin_centers) > 1 else 1.0
            )
            colors_kw = dict(color=p.color)
            if p.bar_colors is not None:
                _h_min = float(np.min(p.bar_colors))
                _h_max = float(np.max(p.bar_colors))
                if _h_max <= _h_min:
                    _h_max = _h_min + 1.0
                colors_kw = dict(
                    color=p.bar_colors.tolist(),
                    colorscale=_mpl_to_plotly_cmap(p.colormap),
                )
            fig.add_trace(
                go.Bar(x=p.bin_centers.tolist(), y=heights,
                       width=width,
                       marker=dict(line=dict(color="white", width=0.5), **colors_kw),
                       showlegend=False),
                row=row, col=col,
            )
        else:
            fig.add_trace(
                go.Histogram(x=np.asarray(p.values).tolist(),
                             nbinsx=p.bins,
                             histnorm="probability density" if p.density else "",
                             marker=dict(color=p.color, line=dict(color="white", width=0.4)),
                             opacity=0.6, showlegend=False),
                row=row, col=col,
            )

        if p.overlay_normal is not None:
            mu, sigma = p.overlay_normal
            if sigma > 0:
                vals = np.asarray(p.values)
                x_grid = np.linspace(float(np.min(vals)), float(np.max(vals)), 200)
                normal_pdf = (
                    1 / (sigma * np.sqrt(2 * np.pi))
                    * np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)
                )
                label = p.overlay_label or f"Normal(mu={mu:.2g}, sigma={sigma:.2g})"
                fig.add_trace(
                    go.Scatter(x=x_grid.tolist(), y=normal_pdf.tolist(),
                               mode="lines",
                               line=dict(color="red", dash="dash", width=1.4),
                               name=label, showlegend=True),
                    row=row, col=col,
                )

        fig.update_xaxes(title_text=p.xlabel, row=row, col=col, showgrid=p.grid)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid,
                         type="log" if p.yscale == "log" else "linear")

    def _heatmap(self, fig, p: HeatmapPanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go

        text_kw = {}
        if p.cell_text is not None:
            text_kw["text"] = [[format(v, p.text_format) for v in r] for r in p.cell_text]
            text_kw["texttemplate"] = "%{text}"
            text_kw["textfont"] = dict(size=10)

        fig.add_trace(
            go.Heatmap(z=p.matrix.tolist(),
                       x=list(p.col_labels), y=list(p.row_labels),
                       colorscale=_mpl_to_plotly_cmap(p.colormap),
                       colorbar=dict(title=p.colorbar_label) if p.colorbar_label else None,
                       showscale=True,
                       **text_kw),
            row=row, col=col,
        )
        fig.update_xaxes(title_text=p.xlabel, row=row, col=col,
                         tickangle=-45)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col,
                         autorange="reversed")  # match matplotlib top-to-bottom row order

    def _bar(self, fig, p: BarPanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go

        if isinstance(p.values, tuple):
            for i, series in enumerate(p.values):
                lbl = p.series_labels[i] if p.series_labels else f"series {i}"
                col_kw = {}
                if p.colors is not None and i < len(p.colors):
                    col_kw["marker"] = dict(color=p.colors[i])
                fig.add_trace(
                    go.Bar(x=list(p.categories), y=series.tolist(),
                           name=lbl, showlegend=True, **col_kw),
                    row=row, col=col,
                )
            fig.update_layout(barmode="group")
        else:
            col_kw = dict(marker=dict(color=p.colors[0] if p.colors else "steelblue"))
            fig.add_trace(
                go.Bar(x=list(p.categories),
                       y=np.asarray(p.values).tolist(),
                       showlegend=False, **col_kw),
                row=row, col=col,
            )
        fig.update_xaxes(title_text=p.xlabel, row=row, col=col,
                         tickangle=-p.xtick_rotation if p.xtick_rotation else 0)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid)

    def _line(self, fig, p: LinePanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go
        from mlframe.reporting.colors import line_color

        ys = p.y if isinstance(p.y, tuple) else (p.y,)
        labels = p.series_labels or (None,) * len(ys)
        styles = p.line_styles or ("-",) * len(ys)
        cols = p.colors or tuple(line_color(i) for i in range(len(ys)))
        # matplotlib '--' / ':' / '-' -> plotly 'dash' / 'dot' / 'solid'.
        _STYLE_MAP = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}
        for i, y in enumerate(ys):
            mpl_style = styles[i % len(styles)]
            dash = _STYLE_MAP.get(mpl_style, "solid")
            fig.add_trace(
                go.Scatter(x=p.x.tolist() if isinstance(p.x, np.ndarray) else p.x,
                           y=y.tolist() if isinstance(y, np.ndarray) else y,
                           mode="lines",
                           line=dict(color=cols[i % len(cols)], dash=dash),
                           name=labels[i] if i < len(labels) else None,
                           showlegend=any(labels)),
                row=row, col=col,
            )
        fig.update_xaxes(title_text=p.xlabel, row=row, col=col, showgrid=p.grid)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid)

    def _violin(self, fig, p: ViolinPanelSpec, row: int, col: int) -> None:
        import plotly.graph_objects as go

        for i, group in enumerate(p.groups):
            fig.add_trace(
                go.Violin(y=np.asarray(group).tolist(),
                          name=p.group_labels[i],
                          box_visible=p.show_box,
                          meanline_visible=False,
                          showlegend=False),
                row=row, col=col,
            )
        fig.update_xaxes(title_text=p.xlabel, row=row, col=col,
                         tickangle=-30)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid)


# Map matplotlib colormap names to plotly's named colorscales (best-effort).
# plotly supports many of the standard mpl names directly via lowercase.
_MPL_TO_PLOTLY = {
    "RdYlBu": "RdYlBu",
    "RdYlBu_r": "RdYlBu_r",
    "RdBu": "RdBu",
    "RdBu_r": "RdBu_r",
    "viridis": "Viridis",
    "plasma": "Plasma",
    "magma": "Magma",
    "inferno": "Inferno",
    "Blues": "Blues",
    "Greens": "Greens",
}


def _mpl_to_plotly_cmap(name: str) -> str:
    """Map a matplotlib colormap name to a plotly colorscale name.

    Falls back to 'Viridis' for unknown names with a one-time WARN.
    """
    if name in _MPL_TO_PLOTLY:
        return _MPL_TO_PLOTLY[name]
    logger.warning(
        "Unknown colormap %r; falling back to plotly 'Viridis'. "
        "Add the mapping to mlframe.reporting.renderers.plotly._MPL_TO_PLOTLY "
        "to silence this warning.",
        name,
    )
    return "Viridis"


__all__ = ["PlotlyRenderer"]
