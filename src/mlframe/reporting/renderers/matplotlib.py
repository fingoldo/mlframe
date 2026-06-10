"""matplotlib renderer.

Builds a ``matplotlib.figure.Figure`` from a ``FigureSpec``, then dispatches
panel-level rendering by isinstance. Uses the Agg-backed figure path
(``Figure(layout="constrained")`` + ``FigureCanvasAgg``) for save-only
calls so we don't init a GUI backend on headless / parallel runs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, FigureSpec, HeatmapPanelSpec,
    HistogramPanelSpec, LinePanelSpec, NetworkPanelSpec, ScatterPanelSpec,
    ViolinPanelSpec,
)

logger = logging.getLogger(__name__)

# Above this many raw scatter points, cap (downsample preserving extremes) and rasterize so the saved vector
# file (pdf/svg) doesn't embed millions of DOM nodes (3.2s + bloat at 2M).
_SCATTER_MAX_POINTS = 50_000
# Pre-bin a raw histogram above this n with np.histogram + ax.bar instead of letting ax.hist re-scan full n.
_HIST_PREBIN_THRESHOLD = 50_000


class MatplotlibRenderer:
    backend = "matplotlib"

    def render(self, spec: FigureSpec) -> Any:
        # 2026-05-11: REMOVED ``matplotlib.use("Agg", force=False)``
        # here. The renderer creates its own ``FigureCanvasAgg(fig)``
        # explicitly below, so the global-backend mutation is
        # redundant -- AND it broke inline FI display in Jupyter
        # because once Agg is locked in globally, downstream
        # ``plt.show()`` calls in feature_importance.py print
        # "Matplotlib is currently using agg, which is a non-GUI
        # backend, so cannot show the figure." Per the
        # CLAUDE.md rule, the renderer must NOT pollute global state
        # to make its own save path easier when downstream consumers
        # rely on that state for inline rendering.
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        rows = len(spec.panels)
        cols = max((len(r) for r in spec.panels), default=0)
        if rows == 0 or cols == 0:
            raise ValueError("FigureSpec has no panels")

        # Force constrained_layout whenever a suptitle is present: with the
        # default (None) layout engine the suptitle stamps at y=0.98 figure
        # coords while ax.set_title sits at ax-top which lands at the same
        # band on figsize=(15, 4-5) — visible collision in saved PNGs.
        # constrained_layout reserves space for the suptitle. The ~800 ms
        # cost only fires when caller actually asked for a suptitle.
        layout = "constrained" if (spec.constrained_layout or spec.suptitle) else None
        fig_kwargs = {"figsize": spec.figsize, "layout": layout}
        if spec.dpi is not None:
            fig_kwargs["dpi"] = spec.dpi
        fig = Figure(**fig_kwargs)
        FigureCanvasAgg(fig)

        gs_kwargs = {}
        if spec.row_height_ratios is not None:
            gs_kwargs["height_ratios"] = list(spec.row_height_ratios)
        if spec.col_width_ratios is not None:
            gs_kwargs["width_ratios"] = list(spec.col_width_ratios)
        gs = fig.add_gridspec(rows, cols, **gs_kwargs)

        for r, row in enumerate(spec.panels):
            for c, panel in enumerate(row):
                if panel is None:
                    continue
                ax = fig.add_subplot(gs[r, c])
                self._render_panel(ax, panel, fig)

        if spec.suptitle:
            fig.suptitle(spec.suptitle, fontsize=spec.suptitle_fontsize)
        return fig

    def save(self, fig: Any, path: str, fmt: str) -> None:
        fmt = fmt.lower()
        if fmt not in ("png", "pdf", "svg", "jpg", "jpeg"):
            raise ValueError(
                f"matplotlib doesn't support format {fmt!r}; "
                "supported: png/pdf/svg/jpg"
            )
        # bbox_inches="tight" + small pad guarantees suptitle, ytick labels
        # and any annotations outside the axes box land inside the saved
        # PNG. Without this the renderer crops at the figure box and long
        # ytick labels (FI plots) / suptitles get clipped.
        fig.savefig(path, format=fmt, bbox_inches="tight", pad_inches=0.15)

    def show(self, fig: Any) -> None:
        # The renderer builds figures via ``Figure()`` + ``FigureCanvasAgg`` (never through pyplot), so they
        # have no pyplot manager and no ``.number`` -- ``plt.figure(fig.number)`` would raise. In an IPython
        # kernel the right call is ``IPython.display.display(fig)``, which renders inline without pyplot. Outside
        # a kernel, attach to a pyplot manager and show a window (best-effort; headless/no-display is a no-op).
        import sys
        if "IPython" in sys.modules:
            try:
                ip = sys.modules["IPython"].get_ipython()
            except Exception:
                ip = None
            if ip is not None:
                from IPython.display import display
                display(fig)
                return
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            # Only pop a window when matplotlib is in interactive mode (a REPL with plt.ion()). In a plain
            # script / test the backend may still be a blocking GUI backend (Tk), and plt.show() would hang on
            # the mainloop -- so a non-interactive context is a no-op.
            if not matplotlib.is_interactive():
                return
            manager = plt.figure().canvas.manager
            manager.canvas.figure = fig
            fig.set_canvas(manager.canvas)
            plt.show()
        except Exception as e:
            logger.debug("MatplotlibRenderer.show() no-op (no interactive display): %s: %s",
                         type(e).__name__, e)

    # ------------------------------------------------------------------
    # Per-panel dispatch
    # ------------------------------------------------------------------

    def _render_panel(self, ax, panel, fig) -> None:
        if isinstance(panel, ScatterPanelSpec):
            self._scatter(ax, panel, fig)
        elif isinstance(panel, HistogramPanelSpec):
            self._histogram(ax, panel)
        elif isinstance(panel, HeatmapPanelSpec):
            self._heatmap(ax, panel, fig)
        elif isinstance(panel, BarPanelSpec):
            self._bar(ax, panel)
        elif isinstance(panel, LinePanelSpec):
            self._line(ax, panel, fig)
        elif isinstance(panel, ViolinPanelSpec):
            self._violin(ax, panel)
        elif isinstance(panel, NetworkPanelSpec):
            self._network(ax, panel, fig)
        elif isinstance(panel, AnnotationPanelSpec):
            self._annotation(ax, panel)
        else:
            raise TypeError(f"unknown panel type: {type(panel).__name__}")

    def _annotation(self, ax, p: AnnotationPanelSpec) -> None:
        ax.text(0.5, 0.5, p.text, ha="center", va="center", fontsize=p.fontsize,
                transform=ax.transAxes, wrap=True)
        ax.set_title(p.title)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _scatter(self, ax, p: ScatterPanelSpec, fig) -> None:
        import matplotlib
        x = np.asarray(p.x)
        y = np.asarray(p.y)
        n = len(x)
        size_arr = p.point_size if isinstance(p.point_size, np.ndarray) else None
        color_arr = p.point_color if isinstance(p.point_color, np.ndarray) else None

        rasterized = False
        if n > _SCATTER_MAX_POINTS:
            from mlframe.reporting.charts._sampling import subsample_preserving_extremes
            idx = subsample_preserving_extremes(x, y, sample_size=_SCATTER_MAX_POINTS)
            x, y = x[idx], y[idx]
            if size_arr is not None and len(size_arr) == n:
                size_arr = size_arr[idx]
            if color_arr is not None and len(color_arr) == n:
                color_arr = color_arr[idx]
            rasterized = True  # capped scatter still rasterized so a vector export stays small.

        kw = {"alpha": p.point_alpha, "rasterized": rasterized}
        kw["s"] = size_arr if size_arr is not None else float(p.point_size)
        if color_arr is not None:
            kw["c"] = color_arr
            kw["cmap"] = matplotlib.colormaps[p.colormap]
        elif p.point_color is not None:
            kw["color"] = p.point_color
        sc = ax.scatter(x, y, **kw)

        if p.perfect_fit_line and n > 0:
            # Span y=x over the UNION of both axes (so it stays the diagonal even when prediction collapse makes
            # y constant) and square the panel so y=x is a true 45-degree line.
            lo = float(min(np.min(x), np.min(y)))
            hi = float(max(np.max(x), np.max(y)))
            ax.plot([lo, hi], [lo, hi], "g--", label="Perfect fit")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", "datalim")

        if p.inline_labels:
            for lx, ly, txt in p.inline_labels:
                ax.text(lx, ly, txt, fontsize=8, ha="right", va="bottom")

        if p.colorbar_label and color_arr is not None:
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(p.colorbar_label)

        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        ax.set_title(p.title)
        if p.legend_label or p.perfect_fit_line:
            ax.legend(loc="best", fontsize=8, framealpha=0.7)
        if p.grid:
            ax.grid(True, alpha=0.3)

    def _histogram(self, ax, p: HistogramPanelSpec) -> None:
        import matplotlib
        overlay_x_lo = overlay_x_hi = None
        bin_centers = p.bin_centers
        heights = None
        width = None
        if bin_centers is None and len(np.asarray(p.values)) > _HIST_PREBIN_THRESHOLD:
            # Above the hazard ceiling, bin once with numpy instead of letting ax.hist re-scan the full n array.
            from mlframe.reporting.charts._sampling import prebin_histogram
            heights, bin_centers, width = prebin_histogram(np.asarray(p.values), p.bins, p.density)

        if bin_centers is not None:
            if heights is None:
                heights = np.asarray(p.values)
                width = float(p.bin_width or (bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 1.0)
            colors_kw = {"color": p.color}
            if p.bar_colors is not None:
                cm = matplotlib.colormaps[p.colormap]
                _h_min = float(np.min(p.bar_colors))
                _h_max = float(np.max(p.bar_colors))
                if _h_max <= _h_min:
                    _h_max = _h_min + 1.0
                colors_kw = {"color": cm((np.asarray(p.bar_colors) - _h_min) / (_h_max - _h_min))}
            ax.bar(bin_centers, heights, width=width, align="center",
                   edgecolor="white", linewidth=0.5, **colors_kw)
            if len(bin_centers) > 0:
                overlay_x_lo = float(bin_centers[0] - width / 2.0)
                overlay_x_hi = float(bin_centers[-1] + width / 2.0)
        else:
            ax.hist(p.values, bins=p.bins, alpha=0.6, color=p.color,
                    edgecolor="white", linewidth=0.4, density=p.density)

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
                ax.plot(x_grid, normal_pdf, "r--", linewidth=1.4, label=label)
                ax.legend(loc="best", fontsize=8, framealpha=0.7)

        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        ax.set_title(p.title)
        ax.set_yscale(p.yscale)
        if p.grid:
            ax.grid(True, alpha=0.3)

    def _heatmap(self, ax, p: HeatmapPanelSpec, fig) -> None:
        import matplotlib
        cm = matplotlib.colormaps[p.colormap]
        im = ax.imshow(p.matrix, cmap=cm, aspect="auto")
        ax.set_xticks(range(len(p.col_labels)))
        ax.set_xticklabels(p.col_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(p.row_labels)))
        ax.set_yticklabels(p.row_labels, fontsize=8)
        if p.cell_text is not None:
            from mlframe.reporting.colors import auto_text_color
            # Compute global vmin / vmax so each cell's text color reflects
            # its position in the actual color range — naive
            # ``< 0.5`` threshold fails when the matrix range is e.g.
            # [0.3, 0.85] (all values map to the high-luminance end of
            # the colormap and white text becomes invisible).
            mat = p.matrix
            vmin = float(np.nanmin(mat))
            vmax = float(np.nanmax(mat))
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    text_color = auto_text_color(
                        float(mat[i, j]), p.colormap, vmin=vmin, vmax=vmax,
                    )
                    ax.text(j, i, format(p.cell_text[i, j], p.text_format),
                            ha="center", va="center", fontsize=7,
                            color=text_color)
        cbar = fig.colorbar(im, ax=ax)
        if p.colorbar_label:
            cbar.set_label(p.colorbar_label)
        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        ax.set_title(p.title)

    def _bar(self, ax, p: BarPanelSpec) -> None:
        x = np.arange(len(p.categories))
        if isinstance(p.values, tuple):
            # Grouped bars.
            n_series = len(p.values)
            width = 0.8 / n_series
            for i, series in enumerate(p.values):
                offset = (i - (n_series - 1) / 2) * width
                kw = {}
                if p.colors is not None and i < len(p.colors):
                    kw["color"] = p.colors[i]
                lbl = p.series_labels[i] if p.series_labels else None
                ax.bar(x + offset, series, width=width, label=lbl, **kw)
            if p.series_labels:
                ax.legend(loc="best", fontsize=8, framealpha=0.7)
        else:
            kw = {"color": p.colors[0] if p.colors else "steelblue"}
            ax.bar(x, p.values, **kw)
        ax.set_xticks(x)
        ax.set_xticklabels(p.categories, rotation=p.xtick_rotation,
                           ha="right" if p.xtick_rotation else "center", fontsize=8)
        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        ax.set_title(p.title)
        if p.grid:
            ax.grid(True, alpha=0.3, axis="y")

    def _line(self, ax, p: LinePanelSpec, fig=None) -> None:
        from mlframe.reporting.colors import line_color

        ys = p.y if isinstance(p.y, tuple) else (p.y,)
        labels = p.series_labels or (None,) * len(ys)
        styles = p.line_styles or ("-",) * len(ys)
        cols = p.colors or tuple(line_color(i) for i in range(len(ys)))

        if p.band is not None:
            lower, upper = np.asarray(p.band[0]), np.asarray(p.band[1])
            band_color = p.band_color or cols[0]
            ax.fill_between(p.x, lower, upper, color=band_color, alpha=0.2,
                            label=p.band_label, zorder=0)

        for i, y in enumerate(ys):
            token = styles[i % len(styles)]
            color = cols[i % len(cols)]
            label = labels[i] if i < len(labels) else None
            if token == "markers":
                ax.plot(p.x, y, linestyle="none", marker="o", markersize=4, color=color, label=label)
            elif token == "lines+markers":
                ax.plot(p.x, y, linestyle="-", marker="o", markersize=4, color=color, label=label)
            else:
                ax.plot(p.x, y, token, color=color, label=label)

        for vx0, vx1, vcolor, valpha in (p.vspans or ()):
            ax.axvspan(vx0, vx1, color=vcolor, alpha=valpha, zorder=0)
        for vx, vcolor, vlabel in (p.vlines or ()):
            ax.axvline(vx, color=vcolor, linestyle=":", linewidth=1.2,
                       label=vlabel or None)

        if any(labels) or p.band_label or any((p.vlines or ())):
            ax.legend(loc="best", fontsize=8, framealpha=0.7)
        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        ax.set_title(p.title)
        if p.grid:
            ax.grid(True, alpha=0.3)
        if p.x_is_time and fig is not None:
            fig.autofmt_xdate()

    def _violin(self, ax, p: ViolinPanelSpec) -> None:
        ax.violinplot(p.groups, showmeans=False, showextrema=False,
                      showmedians=p.show_box)
        ax.set_xticks(range(1, len(p.group_labels) + 1))
        ax.set_xticklabels(p.group_labels, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        ax.set_title(p.title)
        if p.grid:
            ax.grid(True, alpha=0.3, axis="y")

    def _network(self, ax, p: NetworkPanelSpec, fig) -> None:
        import matplotlib
        from matplotlib.cm import ScalarMappable
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D

        nx_pos = np.column_stack([np.asarray(p.node_x, dtype=float),
                                  np.asarray(p.node_y, dtype=float)])
        e_src = np.asarray(p.edge_src, dtype=np.int64)
        e_dst = np.asarray(p.edge_dst, dtype=np.int64)
        weights = np.asarray(p.edge_weight, dtype=float)

        # Edges as a single LineCollection: O(E) artists collapse to one draw
        # call, so thousands of edges stay cheap. Width + color both encode MI.
        if e_src.size:
            segments = [[tuple(nx_pos[a]), tuple(nx_pos[b])] for a, b in zip(e_src, e_dst)]
            wmin, wmax = float(weights.min()), float(weights.max())
            norm = Normalize(vmin=wmin, vmax=wmax if wmax > wmin else wmin + 1e-9)
            cmap = matplotlib.colormaps[p.colormap]
            lo, hi = p.edge_width_range
            if wmax > wmin:
                lws = lo + (weights - wmin) / (wmax - wmin) * (hi - lo)
            else:
                lws = np.full_like(weights, (lo + hi) / 2.0)
            lc = LineCollection(segments, linewidths=lws.tolist(),
                                colors=cmap(norm(weights)), alpha=0.8, zorder=1)
            ax.add_collection(lc)

            # Arrows for directed edges. Drawn per-edge (annotate has no batch
            # form); the friend-graph max_nodes guard keeps edge counts modest.
            directed = p.edge_directed
            if np.isscalar(directed):
                directed = np.full(e_src.shape, bool(directed))
            else:
                directed = np.asarray(directed, dtype=bool)
            for a, b, d in zip(e_src, e_dst, directed):
                if d:
                    ax.annotate("", xy=tuple(nx_pos[b]), xytext=tuple(nx_pos[a]),
                                arrowprops=dict(arrowstyle="-|>", color="0.35",
                                                alpha=0.6, shrinkA=8, shrinkB=8),
                                zorder=2)

            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            if p.colorbar_label:
                cbar.set_label(p.colorbar_label)

        ax.scatter(nx_pos[:, 0], nx_pos[:, 1], s=np.asarray(p.node_size, dtype=float),
                   c=list(p.node_color), edgecolors="black", linewidths=0.5, zorder=3)
        for (x, y), label in zip(nx_pos, p.node_label):
            ax.annotate(label, (x, y), fontsize=7, ha="center", va="center", zorder=4)

        if p.node_legend:
            handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=col,
                              markersize=8, label=lbl) for lbl, col in p.node_legend]
            ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.7)

        ax.set_title(p.title)
        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(0.12)


__all__ = ["MatplotlibRenderer"]
