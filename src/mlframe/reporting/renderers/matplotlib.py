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
    AnnotationPanelSpec, BarPanelSpec, ConfusionMarginsPanelSpec, FigureSpec,
    HeatmapPanelSpec, HistogramPanelSpec, LinePanelSpec, NetworkPanelSpec,
    ScatterPanelSpec, ViolinPanelSpec,
)

from ._shared_helpers import _finite_range, _thin_tick_positions

logger = logging.getLogger(__name__)

# Panel-title rendering: wrap long titles onto multiple lines (so a verbose diagnostic title doesn't
# overflow a narrow panel) and cap the font so it can't dwarf the panel. ~46 chars/line is a good fit
# for the default ~6-inch panel width.
_TITLE_WRAP_CHARS = 46
_TITLE_FONTSIZE = 10


def _set_panel_title(ax, title) -> None:
    if not title:
        return
    import textwrap
    s = str(title)
    if len(s) > _TITLE_WRAP_CHARS:
        s = "\n".join(textwrap.wrap(s, width=_TITLE_WRAP_CHARS, break_long_words=False))
    ax.set_title(s, fontsize=_TITLE_FONTSIZE)


# Above this many raw scatter points, cap (downsample preserving extremes) and rasterize so the saved vector
# file (pdf/svg) doesn't embed millions of DOM nodes (3.2s + bloat at 2M).
_SCATTER_MAX_POINTS = 50_000
# Pre-bin a raw histogram above this n with np.histogram + ax.bar instead of letting ax.hist re-scan full n.
_HIST_PREBIN_THRESHOLD = 50_000
# Above this many heatmap cells the per-cell text turns to unreadable soup; skip it (also keeps the plotly
# per-annotation O(cells) loop from stalling on a degenerate huge-K grid).
_HEATMAP_CELL_TEXT_MAX = 400


def _err_to_mpl(err):
    """Spec error-bar field -> matplotlib ``errorbar`` yerr/xerr arg.

    A single array is symmetric; a (lower, upper) pair is asymmetric and matplotlib wants a (2, N) array of the
    DISTANCES from the point (Wilson CIs are asymmetric, so the spec carries absolute distances per side)."""
    if err is None:
        return None
    if isinstance(err, tuple):
        return np.vstack([np.asarray(err[0], dtype=float), np.asarray(err[1], dtype=float)])
    return np.asarray(err, dtype=float)


def _per_series_flags(flag, n: int):
    """Normalize a per-series bool flag (single bool / tuple / None) into a length-n bool list."""
    if flag is None:
        return [False] * n
    if isinstance(flag, (tuple, list, np.ndarray)):
        seq = list(flag)
        return [bool(seq[i]) if i < len(seq) else False for i in range(n)]
    return [bool(flag)] * n


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
        layout = "constrained" if (spec.constrained_layout or spec.suptitle or spec.caption) else None
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

        col_axes: dict[int, list] = {}
        axes_grid: list[list] = []
        for r, row in enumerate(spec.panels):
            row_axes = []
            for c, panel in enumerate(row):
                if panel is None:
                    row_axes.append(None)
                    continue
                ax = fig.add_subplot(gs[r, c])
                row_axes.append(ax)
                col_axes.setdefault(c, []).append(ax)
            axes_grid.append(row_axes)

        # A colorbar attached to a single axes shrinks only that axes; when a shared-x panel (calibration histogram)
        # sits below the scatter, anchor the bar across the whole column so both data axes keep the same width.
        for r, row in enumerate(spec.panels):
            for c, panel in enumerate(row):
                if panel is None:
                    continue
                ax = axes_grid[r][c]
                cbar_axes = col_axes[c] if (spec.sharex and len(col_axes[c]) > 1) else ax
                self._render_panel(ax, panel, fig, cbar_axes=cbar_axes)

        if spec.suptitle:
            # Wrap a long suptitle so it doesn't overflow the figure width (a verbose model identity
            # like "TEST LGBMRegressor TVT basic_new ... trained on 4.1M rows @iter=298 training curves"
            # ran off the right edge). Wrap each author-supplied line independently so explicit newlines
            # are preserved; ~90 chars suits the full figure width (vs ~46 for a single panel).
            import textwrap
            _sup_lines = []
            for _ln in str(spec.suptitle).split("\n"):
                _sup_lines.extend(textwrap.wrap(_ln, width=90, break_long_words=False) or [""])
            fig.suptitle("\n".join(_sup_lines), fontsize=spec.suptitle_fontsize)
        if spec.caption:
            # How-to-read footnote, small + dim, in a reserved bottom band so it never overlaps the x-axis label.
            # constrained_layout (forced on above when a caption is present) is told to leave the band free via rect.
            import textwrap
            _cap_lines = textwrap.wrap(str(spec.caption), width=110, break_long_words=False) or [""]
            _cap = "\n".join(_cap_lines)
            _h_px = fig.get_size_inches()[1] * (fig.get_dpi() or 100.0)
            _band = min(0.30, (len(_cap_lines) * 11.0 + 12.0) / _h_px)  # bottom fraction reserved for the caption
            _eng = fig.get_layout_engine()
            if _eng is not None:
                try:
                    _eng.set(rect=(0.0, _band, 1.0, 1.0))
                except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                    logger.debug("suppressed in matplotlib.py:163: %s", e)
                    pass
            fig.text(0.5, _band * 0.5, _cap, ha="center", va="center", fontsize=7, color="0.35")
        return fig

    def save(self, fig: Any, path: str, fmt: str) -> None:
        fmt = fmt.lower()
        if fmt not in ("png", "pdf", "svg", "jpg", "jpeg"):
            raise ValueError(f"matplotlib doesn't support format {fmt!r}; " "supported: png/pdf/svg/jpg")
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
            logger.debug("MatplotlibRenderer.show() no-op (no interactive display): %s: %s", type(e).__name__, e)

    # ------------------------------------------------------------------
    # Per-panel dispatch
    # ------------------------------------------------------------------

    def _render_panel(self, ax, panel, fig, cbar_axes=None) -> None:
        if isinstance(panel, ScatterPanelSpec):
            self._scatter(ax, panel, fig, cbar_axes=cbar_axes)
        elif isinstance(panel, HistogramPanelSpec):
            self._histogram(ax, panel)
        elif isinstance(panel, HeatmapPanelSpec):
            self._heatmap(ax, panel, fig)
        elif isinstance(panel, ConfusionMarginsPanelSpec):
            self._confusion_margins(ax, panel, fig)
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
        ax.text(0.5, 0.5, p.text, ha="center", va="center", fontsize=p.fontsize, transform=ax.transAxes, wrap=True)
        _set_panel_title(ax, p.title)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _scatter(self, ax, p: ScatterPanelSpec, fig, cbar_axes=None) -> None:
        import matplotlib
        x = np.asarray(p.x)
        y = np.asarray(p.y)
        n = len(x)
        size_arr = p.point_size if isinstance(p.point_size, np.ndarray) else None
        color_arr = p.point_color if isinstance(p.point_color, np.ndarray) else None

        rasterized = False
        if n > _SCATTER_MAX_POINTS:
            from mlframe.reporting.charts import subsample_preserving_extremes
            idx = subsample_preserving_extremes(x, y, sample_size=_SCATTER_MAX_POINTS)
            x, y = x[idx], y[idx]
            if size_arr is not None and len(size_arr) == n:
                size_arr = size_arr[idx]
            if color_arr is not None and len(color_arr) == n:
                color_arr = color_arr[idx]
            rasterized = True  # capped scatter still rasterized so a vector export stays small.

        # Per-point error bars (e.g. Wilson CIs on reliability bins). Drawn before the scatter so the markers
        # sit on top. Subsample never reorders for these CI panels (n is bin-count, well under the cap), so the
        # error arrays align with x/y as-passed.
        if p.y_err is not None or p.x_err is not None:
            yerr = _err_to_mpl(p.y_err)
            xerr = _err_to_mpl(p.x_err)
            ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt="none", ecolor="gray", elinewidth=1.0, capsize=3, alpha=0.7, zorder=1)

        kw = {"alpha": p.point_alpha, "rasterized": rasterized}
        kw["s"] = size_arr if size_arr is not None else float(p.point_size)
        if color_arr is not None:
            kw["c"] = color_arr
            kw["cmap"] = matplotlib.colormaps[p.colormap]
        elif p.point_color is not None:
            kw["color"] = p.point_color
        sc = ax.scatter(x, y, **kw)

        # Emphasised subset (worst-K errors): drawn on top, larger + colored. Indices are positions into the
        # ORIGINAL arrays, so resolve against the pre-subsample data (``p.x`` / ``p.y``), not the capped ``x``/``y``.
        if p.highlight_indices is not None:
            hi = np.asarray(p.highlight_indices, dtype=np.int64)
            ox, oy = np.asarray(p.x), np.asarray(p.y)
            hi = hi[(hi >= 0) & (hi < len(ox))]
            if hi.size:
                base_s = float(p.point_size) if size_arr is None else float(np.median(np.asarray(p.point_size)))
                ax.scatter(ox[hi], oy[hi], s=base_s * 4.0, facecolors="none", edgecolors=p.highlight_color, linewidths=1.5, zorder=5, label="worst-K")

        if p.trend_line is not None and n > 1:
            from mlframe.reporting.renderers._trend import robust_fit_endpoints
            ends = robust_fit_endpoints(np.asarray(p.x), np.asarray(p.y), p.trend_line)
            if ends is not None:
                (tx0, ty0), (tx1, ty1) = ends
                ax.plot([tx0, tx1], [ty0, ty1], color="darkorange", linestyle="-", linewidth=1.6, zorder=4, label=f"robust fit ({p.trend_line})")

        if p.overlay_band is not None:
            bx, blo, bhi = (np.asarray(a) for a in p.overlay_band)
            ax.fill_between(bx, blo, bhi, color="purple", alpha=0.18, zorder=3, linewidth=0, label="curve 95% band")

        if p.overlay_line is not None:
            ox_grid, oy_grid, olabel = p.overlay_line
            ax.plot(np.asarray(ox_grid), np.asarray(oy_grid), color="purple", linestyle="-", linewidth=1.8, zorder=4, label=olabel)

        if p.perfect_fit_line and n > 0:
            # Span y=x over the UNION of both axes (so it stays the diagonal even when prediction collapse makes
            # y constant) and square the panel so y=x is a true 45-degree line.
            lo = float(min(np.min(x), np.min(y)))
            hi = float(max(np.max(x), np.max(y)))
            ax.plot([lo, hi], [lo, hi], "g--", label="Perfect fit")
            if not p.equal_aspect:
                # Probability-vs-probability (calibration): the diagonal spans corner-to-corner at any aspect, so let
                # the panel fill its cell width and align with the histogram below; xlim/ylim are applied just after.
                pass
            elif p.xlim is not None or p.ylim is not None:
                # Explicit limits given: "datalim" would discard set_xlim to satisfy equal aspect (large bubble
                # markers then drive x far past the data); "box" keeps the fixed limits and squares via the box.
                ax.set_aspect("equal", "box")
            else:
                # Equal lo..hi limits on both axes already make the diagonal a true 45-degree line; square via the box
                # ("box" respects the fixed limits). "datalim" would instead adjust the limits to satisfy the aspect and
                # log "Ignoring fixed x limits to fulfill fixed data aspect" on every scatter panel.
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_aspect("equal", "box")
        if p.xlim is not None:
            ax.set_xlim(*p.xlim)
        if p.ylim is not None:
            ax.set_ylim(*p.ylim)

        if p.inline_labels:
            for lx, ly, txt in p.inline_labels:
                ax.text(lx, ly, txt, fontsize=8, ha="right", va="bottom")

        if p.colorbar_label and color_arr is not None:
            cbar = fig.colorbar(sc, ax=(cbar_axes if cbar_axes is not None else ax))
            cbar.set_label(p.colorbar_label)

        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        _set_panel_title(ax, p.title)
        if p.legend_label or p.perfect_fit_line or p.trend_line or p.overlay_line is not None or p.overlay_band is not None or p.highlight_indices is not None:
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
            from mlframe.reporting.charts import prebin_histogram
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
            ax.bar(bin_centers, heights, width=width, align="center", edgecolor="white", linewidth=0.5, **colors_kw)
            if len(bin_centers) > 0:
                overlay_x_lo = float(bin_centers[0] - width / 2.0)
                overlay_x_hi = float(bin_centers[-1] + width / 2.0)
        else:
            # ax.hist autodetects its range from the data and raises on empty / all-non-finite input; drop
            # non-finite first and fall back to an empty axes when nothing is left to bin.
            vals = np.asarray(p.values, dtype=float).ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size:
                ax.hist(vals, bins=p.bins, alpha=0.6, color=p.color, edgecolor="white", linewidth=0.4, density=p.density)
            else:
                ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes, fontsize=9)

        if p.overlay_normal is not None:
            mu, sigma = p.overlay_normal
            if sigma > 0:
                if overlay_x_lo is None:
                    vals = np.asarray(p.values)
                    overlay_x_lo, overlay_x_hi = float(np.min(vals)), float(np.max(vals))
                x_grid = np.linspace(overlay_x_lo, overlay_x_hi, 200)
                normal_pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)
                label = p.overlay_label or f"Normal(mu={mu:.2g}, sigma={sigma:.2g})"
                ax.plot(x_grid, normal_pdf, "r--", linewidth=1.4, label=label)
                ax.legend(loc="best", fontsize=8, framealpha=0.7)

        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        _set_panel_title(ax, p.title)
        ax.set_yscale(p.yscale)
        if p.xlim is not None:
            ax.set_xlim(*p.xlim)
        if p.grid:
            ax.grid(True, alpha=0.3)

    def _heatmap(self, ax, p: HeatmapPanelSpec, fig) -> None:
        import matplotlib
        from mlframe.reporting.colors import resolve_heatmap_cmap
        cmap_name = resolve_heatmap_cmap(p.colormap)
        cm = matplotlib.colormaps[cmap_name]
        # A density panel carrying ``trend_xy`` (the regression pred-vs-true heatmap) reads "bottom-up"
        # (row 0 = lowest value), so it needs origin="lower"; other heatmaps (confusion / drift) keep the
        # default top-down matrix orientation.
        _heatmap_origin = "lower" if getattr(p, "trend_xy", None) is not None else "upper"
        im = ax.imshow(p.matrix, cmap=cm, aspect="auto", origin=_heatmap_origin)
        _xt = _thin_tick_positions(len(p.col_labels))
        ax.set_xticks(_xt)
        ax.set_xticklabels([p.col_labels[i] for i in _xt], rotation=45, ha="right", fontsize=8)
        _yt = _thin_tick_positions(len(p.row_labels))
        ax.set_yticks(_yt)
        ax.set_yticklabels([p.row_labels[i] for i in _yt], fontsize=8)
        rng = _finite_range(p.matrix)
        if p.cell_text is not None and rng is not None and p.matrix.size <= _HEATMAP_CELL_TEXT_MAX:
            from mlframe.reporting.colors import auto_text_color
            # Compute global vmin / vmax so each cell's text color reflects
            # its position in the actual color range — naive
            # ``< 0.5`` threshold fails when the matrix range is e.g.
            # [0.3, 0.85] (all values map to the high-luminance end of
            # the colormap and white text becomes invisible).
            mat = p.matrix
            vmin, vmax = rng
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    cell = float(mat[i, j])
                    text_color = auto_text_color(cell if np.isfinite(cell) else vmin, cmap_name, vmin=vmin, vmax=vmax)
                    ax.text(j, i, format(p.cell_text[i, j], p.text_format), ha="center", va="center", fontsize=7, color=text_color)
        # Iso-value contour overlays at named matrix levels (PSI 0.10 / 0.25 triage lines on the drift heatmap).
        # Contour coords are the imshow cell-center grid (0..ncols-1, 0..nrows-1) so lines land between cells.
        if p.threshold_contours:
            mat = np.asarray(p.matrix, dtype=float)
            if mat.ndim == 2 and mat.shape[0] >= 2 and mat.shape[1] >= 2:
                gx, gy = np.meshgrid(np.arange(mat.shape[1]), np.arange(mat.shape[0]))
                for level, color in p.threshold_contours:
                    lo, hi = float(np.nanmin(mat)), float(np.nanmax(mat))
                    if lo < level < hi:  # contour only exists when the level is crossed
                        ax.contour(gx, gy, mat, levels=[level], colors=[color], linewidths=1.4)
        if p.trend_line is not None and p.trend_xy is not None:
            from mlframe.reporting.renderers._trend import robust_fit_endpoints
            # The imshow axes live in BIN-INDEX space (0..nbins-1); robust_fit_endpoints + the y=x
            # diagonal are in VALUE space. Map value -> index using the SAME (lo, hi) the panel binned on
            # (lo = min over both arrays, hi = max), else the line is plotted at value coords (~1e4) on a
            # 0..79 axis, auto-expanding the axis and squishing the density into a corner.
            _xv = np.asarray(p.trend_xy[0], dtype=np.float64).ravel()
            _yv = np.asarray(p.trend_xy[1], dtype=np.float64).ravel()
            _fin = np.isfinite(_xv) & np.isfinite(_yv)
            _nb = len(p.col_labels)
            if int(_fin.sum()) >= 2 and _nb >= 2:
                _lo = float(min(_xv[_fin].min(), _yv[_fin].min()))
                _hi = float(max(_xv[_fin].max(), _yv[_fin].max()))
                if _hi > _lo:
                    def _to_idx(v: float) -> float:
                        return (float(v) - _lo) / (_hi - _lo) * (_nb - 1)
                    # y=x reference in index space (origin="lower" -> bottom-left to top-right).
                    ax.plot([0, _nb - 1], [0, _nb - 1], color="0.4", linestyle=":", linewidth=1.0, label="y=x")
                    ends = robust_fit_endpoints(_xv, _yv, p.trend_line)
                    if ends is not None:
                        (tx0, ty0), (tx1, ty1) = ends
                        ax.plot(
                            [_to_idx(tx0), _to_idx(tx1)], [_to_idx(ty0), _to_idx(ty1)],
                            color="darkorange", linestyle="-", linewidth=1.6,
                            label=f"robust fit ({p.trend_line})",
                        )
                    ax.set_xlim(-0.5, _nb - 0.5)
                    ax.set_ylim(-0.5, _nb - 0.5)
                    ax.legend(loc="best", fontsize=8, framealpha=0.7)
        cbar = fig.colorbar(im, ax=ax)
        if p.colorbar_label:
            cbar.set_label(p.colorbar_label)
        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        _set_panel_title(ax, p.title)

    def _confusion_margins(self, ax, p: ConfusionMarginsPanelSpec, fig) -> None:
        import matplotlib
        from mlframe.reporting.colors import resolve_heatmap_cmap
        # The single panel cell hosts a 2x2 small-multiple: top bar (predicted volume), heatmap + right bar (true
        # support). Subdividing the cell's own subplotspec keeps the layout grid-driven and aligned with siblings;
        # the passed ``ax`` is the placeholder we replace with the sub-axes.
        cmap_name = resolve_heatmap_cmap(p.colormap)
        cm = matplotlib.colormaps[cmap_name]
        K = p.matrix.shape[0]
        ax.set_axis_off()
        gs = ax.get_subplotspec().subgridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 5], wspace=0.05, hspace=0.05)
        ax_top = fig.add_subplot(gs[0, 0])
        ax_hm = fig.add_subplot(gs[1, 0])
        ax_right = fig.add_subplot(gs[1, 1])

        im = ax_hm.imshow(p.matrix, cmap=cm, aspect="auto")
        ax_hm.set_xticks(range(len(p.col_labels)))
        ax_hm.set_xticklabels(p.col_labels, rotation=45, ha="right", fontsize=8)
        ax_hm.set_yticks(range(len(p.row_labels)))
        ax_hm.set_yticklabels(p.row_labels, fontsize=8)
        ax_hm.set_xlabel(p.xlabel)
        ax_hm.set_ylabel(p.ylabel)
        rng = _finite_range(p.matrix)
        if p.cell_text is not None and rng is not None and p.matrix.size <= _HEATMAP_CELL_TEXT_MAX:
            from mlframe.reporting.colors import auto_text_color
            vmin, vmax = rng
            for i in range(K):
                for j in range(p.matrix.shape[1]):
                    cell = float(p.matrix[i, j])
                    tc = auto_text_color(cell if np.isfinite(cell) else vmin, cmap_name, vmin=vmin, vmax=vmax)
                    ax_hm.text(j, i, format(p.cell_text[i, j], p.text_format), ha="center", va="center", fontsize=7, color=tc)

        pos = np.arange(K)
        # Top bar: predicted-class volume, aligned to the heatmap columns (shared x, ticks hidden -- the heatmap owns them).
        ax_top.bar(pos, np.asarray(p.col_margin, dtype=float), color="#4c72b0", width=0.8)
        ax_top.set_xlim(-0.5, K - 0.5)
        ax_top.set_xticks([])
        ax_top.tick_params(axis="y", labelsize=7)
        ax_top.set_ylabel(p.col_margin_label, fontsize=7)
        # Right bar: per-true-class support, aligned to the heatmap rows (imshow y runs top->bottom, so invert).
        ax_right.barh(pos, np.asarray(p.row_margin, dtype=float), color="#55a868", height=0.8)
        ax_right.set_ylim(-0.5, K - 0.5)
        ax_right.invert_yaxis()
        ax_right.set_yticks([])
        ax_right.tick_params(axis="x", labelsize=7, rotation=45)
        ax_right.set_xlabel(p.row_margin_label, fontsize=7)

        cbar = fig.colorbar(im, ax=ax_right, fraction=0.25, pad=0.35)
        if p.colorbar_label:
            cbar.set_label(p.colorbar_label, fontsize=8)
        title = p.title if not p.note else f"{p.title}\n{p.note}"
        ax_top.set_title(title, fontsize=10)

    def _bar(self, ax, p: BarPanelSpec) -> None:
        horizontal = p.orientation == "horizontal"
        pos = np.arange(len(p.categories))
        if isinstance(p.values, tuple):
            # Grouped bars.
            n_series = len(p.values)
            thickness = 0.8 / n_series
            for i, series in enumerate(p.values):
                offset = (i - (n_series - 1) / 2) * thickness
                kw = {}
                if p.colors is not None and i < len(p.colors):
                    kw["color"] = p.colors[i]
                lbl = p.series_labels[i] if p.series_labels else None
                if horizontal:
                    ax.barh(pos + offset, series, height=thickness, label=lbl, **kw)
                else:
                    ax.bar(pos + offset, series, width=thickness, label=lbl, **kw)
            if p.series_labels:
                ax.legend(loc="best", fontsize=8, framealpha=0.7)
        else:
            kw = {"color": p.colors[0] if p.colors else "steelblue"}
            if horizontal:
                ax.barh(pos, p.values, **kw)
            else:
                ax.bar(pos, p.values, **kw)

        # Reference line perpendicular to the bars (global metric across a per-segment bar). axvline for
        # horizontal bars (value axis is x), axhline for vertical bars (value axis is y).
        if p.hline is not None:
            hval, hcolor, hlabel = p.hline
            if horizontal:
                ax.axvline(hval, color=hcolor, linestyle="--", linewidth=1.3, label=hlabel or None)
            else:
                ax.axhline(hval, color=hcolor, linestyle="--", linewidth=1.3, label=hlabel or None)
            if hlabel:
                ax.legend(loc="best", fontsize=8, framealpha=0.7)

        if horizontal:
            ax.set_yticks(pos)
            ax.set_yticklabels(p.categories, fontsize=8)
            ax.invert_yaxis()  # first category on top -> worst-first ranking reads top-down
        else:
            # Thin the x-tick labels when there are many categories so they don't overlap into an
            # unreadable smear (e.g. a 50-lag residual-ACF bar chart). Keep ~20 evenly-spaced labels;
            # the bars themselves stay 1-per-category, only the LABELS are subsampled.
            n_cat = len(p.categories)
            if n_cat > 25:
                step = int(np.ceil(n_cat / 20))
                sel = np.arange(0, n_cat, step)
                ax.set_xticks(pos[sel])
                ax.set_xticklabels(
                    [p.categories[i] for i in sel],
                    rotation=p.xtick_rotation or 0,
                    ha="right" if p.xtick_rotation else "center", fontsize=8,
                )
            else:
                ax.set_xticks(pos)
                ax.set_xticklabels(p.categories, rotation=p.xtick_rotation, ha="right" if p.xtick_rotation else "center", fontsize=8)
        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        _set_panel_title(ax, p.title)
        if p.grid:
            ax.grid(True, alpha=0.3, axis="x" if horizontal else "y")

    def _line(self, ax, p: LinePanelSpec, fig=None) -> None:
        from mlframe.reporting.colors import line_color

        ys = p.y if isinstance(p.y, tuple) else (p.y,)
        # Per-series x: a tuple of x arrays parallel to ``y`` (ROC overlays with different fpr grids); else shared.
        xs_per_series = isinstance(p.x, tuple)
        labels = p.series_labels or (None,) * len(ys)
        styles = p.line_styles or ("-",) * len(ys)
        cols = p.colors or tuple(line_color(i) for i in range(len(ys)))
        sec = _per_series_flags(p.secondary_y, len(ys))
        fills = _per_series_flags(p.fill_to_baseline, len(ys))

        def _xi(i):
            return p.x[i] if xs_per_series else p.x

        # Lazily create the twin axis only when a series actually needs it.
        ax2 = ax.twinx() if any(sec) else None
        proxies = []  # legend proxies for labeled vspans

        if p.band is not None:
            lower, upper = np.asarray(p.band[0]), np.asarray(p.band[1])
            band_color = p.band_color or cols[0]
            ax.fill_between(_xi(0) if not xs_per_series else p.x[0], lower, upper, color=band_color, alpha=0.2, label=p.band_label, zorder=0)

        for i, y in enumerate(ys):
            token = styles[i % len(styles)]
            color = cols[i % len(cols)]
            label = labels[i] if i < len(labels) else None
            target = ax2 if (ax2 is not None and sec[i]) else ax
            xi = _xi(i)  # nosec B105 - not a credential -- config/format token label or sentinel string constant
            if token == "markers":  # nosec B105 - identifier/config-key name matched by heuristic, not an embedded credential
                target.plot(xi, y, linestyle="none", marker="o", markersize=4, color=color, label=label)  # nosec B105 - not a credential -- config/format token label or sentinel string constant
            elif token == "lines+markers":  # nosec B105 - identifier/config-key name matched by heuristic, not an embedded credential
                target.plot(xi, y, linestyle="-", marker="o", markersize=4, color=color, label=label)
            else:
                target.plot(xi, y, token, color=color, label=label)
            if fills[i]:
                step = "post" if p.step_fill else None
                target.fill_between(xi, p.fill_baseline, y, color=color, alpha=0.2, step=step, zorder=0)

        for span in p.vspans or ():
            vx0, vx1, vcolor, valpha = span[0], span[1], span[2], span[3]
            vlabel = span[4] if len(span) > 4 else ""
            ax.axvspan(vx0, vx1, color=vcolor, alpha=valpha, zorder=0)
            if vlabel:
                from matplotlib.patches import Patch
                proxies.append(Patch(facecolor=vcolor, alpha=valpha, label=vlabel))
        for vx, vcolor, vlabel in p.vlines or ():
            ax.axvline(vx, color=vcolor, linestyle=":", linewidth=1.2, label=vlabel or None)

        for mx, my, mlabel, mcolor, msym in (p.point_markers or ()):
            ax.plot([mx], [my], marker=msym or "*", markersize=13, color=mcolor,
                    markeredgecolor="black", markeredgewidth=0.6, linestyle="none",
                    label=mlabel or None, zorder=6)
            if mlabel:
                ax.annotate(mlabel, (mx, my), textcoords="offset points", xytext=(8, -10), fontsize=7, color=mcolor, zorder=6)

        if ax2 is not None:
            ax2.set_ylabel(p.secondary_ylabel)
        if any(labels) or p.band_label or any((p.vlines or ())) or any((p.point_markers or ())) or proxies or (ax2 is not None and any(sec)):
            handles, leg_labels = ax.get_legend_handles_labels()
            if ax2 is not None:
                h2, l2 = ax2.get_legend_handles_labels()
                handles += h2
                leg_labels += l2
            handles += proxies
            leg_labels += [pr.get_label() for pr in proxies]
            if handles:
                if getattr(p, "legend_outside", False):
                    ax.legend(handles, leg_labels, loc="center left", bbox_to_anchor=(1.02, 0.5),
                              fontsize=7, framealpha=0.7, ncol=max(1, int(getattr(p, "legend_ncol", 1))))
                else:
                    ax.legend(handles, leg_labels, loc="best", fontsize=8, framealpha=0.7, ncol=max(1, int(getattr(p, "legend_ncol", 1))))
        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        _set_panel_title(ax, p.title)
        if p.grid:
            ax.grid(True, alpha=0.3)
        if p.x_is_time and fig is not None:
            fig.autofmt_xdate()

    def _violin(self, ax, p: ViolinPanelSpec) -> None:
        ax.violinplot(p.groups, showmeans=False, showextrema=False, showmedians=p.show_box)
        ax.set_xticks(range(1, len(p.group_labels) + 1))
        ax.set_xticklabels(p.group_labels, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        _set_panel_title(ax, p.title)
        if p.grid:
            ax.grid(True, alpha=0.3, axis="y")

    def _network(self, ax, p: NetworkPanelSpec, fig) -> None:
        import matplotlib
        from matplotlib.cm import ScalarMappable
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize
        from matplotlib.lines import Line2D

        nx_pos = np.column_stack([np.asarray(p.node_x, dtype=float), np.asarray(p.node_y, dtype=float)])
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
            lc = LineCollection(segments, linewidths=lws.tolist(), colors=cmap(norm(weights)), alpha=0.8, zorder=1)
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

        ax.scatter(nx_pos[:, 0], nx_pos[:, 1], s=np.asarray(p.node_size, dtype=float), c=list(p.node_color), edgecolors="black", linewidths=0.5, zorder=3)
        for (x, y), label in zip(nx_pos, p.node_label):
            ax.annotate(label, (x, y), fontsize=7, ha="center", va="center", zorder=4)

        if p.node_legend:
            handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=8, label=lbl) for lbl, col in p.node_legend]
            ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.7)

        _set_panel_title(ax, p.title)
        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(0.12)


__all__ = ["MatplotlibRenderer"]
