"""matplotlib renderer.

Builds a ``matplotlib.figure.Figure`` from a ``FigureSpec``, then dispatches
panel-level rendering by isinstance. Uses the Agg-backed figure path
(``Figure(layout="constrained")`` + ``FigureCanvasAgg``) for save-only
calls so we don't init a GUI backend on headless / parallel runs.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar, Optional

from mlframe._output_paths import ensure_parent_dir
import numpy as np

from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, ConfusionMarginsPanelSpec, FigureSpec,
    HeatmapPanelSpec, HistogramPanelSpec, LinePanelSpec, NetworkPanelSpec,
    ScatterPanelSpec, ViolinPanelSpec,
)

from ._shared_helpers import (  # noqa: F401 -- _HEATMAP_MAX_TICKS re-exported for callers importing the tick-thinning constant from this module
    _HEATMAP_CELL_TEXT_MAX, _HEATMAP_MAX_TICKS, _HIST_PREBIN_THRESHOLD, _SCATTER_MAX_POINTS, heatmap_value_to_index,
    CAPTION_FONTSIZE, CAPTION_WRAP_CHARS, PANEL_TITLE_FONTSIZE, SUPTITLE_WRAP_CHARS,
    _finite_range, _per_series_flags, _thin_tick_positions, epoch_ns_ticks,
    _TITLE_REF_WIDTH_IN, histogram_bar_extent, label_width_pitch_in, log_axis_dropped_note, network_label_indices, low_evidence_mask, non_colliding_label_indices,
    panel_title_wrap_chars,
    rotated_tick_pitch_in, select_per_point, ticks_that_fit, truncate_bar_label,
    wrap_annotation_text, wrap_text_to_width, wrap_title_lines,
)

from mlframe.reporting.colors import BAR_PRIMARY, CONFUSION_COL_MARGIN, CONFUSION_ROW_MARGIN, TREND_LINE, resolve_heatmap_cmap
logger = logging.getLogger(__name__)

# Panel-title font cap so a verbose diagnostic title can't dwarf the panel. The chars-per-line budget is
# width-scaled and shared with the plotly renderer (``_shared_helpers.panel_title_wrap_chars``).
_TITLE_FONTSIZE = PANEL_TITLE_FONTSIZE
# Heatmap / violin tick labels, and the size the tick-spacing budget is computed at -- one value so the two
# cannot drift into a budget that assumes smaller glyphs than the ones drawn.
_HEATMAP_TICK_FONTSIZE = 8
# Directed-edge arrowhead ceiling, shared with the plotly renderer so one spec draws the same arrows on
# both backends. Imported from the sibling that owns it rather than restated.
from ._plotly_network import _NETWORK_MAX_ARROWS


def _needs_layout_engine(spec) -> bool:
    """Whether any panel carries axis furniture that needs space reserved for it before drawing."""
    for row in spec.panels:
        for panel in row:
            if panel is None:
                continue
            if getattr(panel, "xtick_rotation", None):
                return True
            if getattr(panel, "orientation", "vertical") == "horizontal":
                return True
            if getattr(panel, "colorbar_label", None):
                return True
            if isinstance(panel, (HeatmapPanelSpec, ConfusionMarginsPanelSpec, ViolinPanelSpec)):
                return True
    return False


def _measured_axis_in(ax, *, horizontal: bool) -> Optional[float]:
    """Length in inches of the axis the category labels run along, or ``None`` when it cannot be measured."""
    try:
        _pos = ax.get_position()
        _fig_w, _fig_h = (float(v) for v in ax.figure.get_size_inches())
        return float(_pos.height) * _fig_h if horizontal else float(_pos.width) * _fig_w
    except Exception:
        logger.debug("could not measure the category axis for tick budgeting; falling back to the fixed cap", exc_info=True)
        return None


def _bar_colors(colors, values):
    """Colour argument for a single-series bar call: the whole per-bar sequence when it matches, else one colour."""
    if not colors:
        return BAR_PRIMARY
    try:
        if len(colors) == len(values) and len(colors) > 1:
            return list(colors)
    except TypeError:
        pass
    return colors[0]


# A bar chart whose categories are DATA-DERIVED (query-size bins, deciles that collapse, a group column with
# two levels) can land on one or two bars. Matplotlib sizes bars in category units and autoscales the axis to
# the categories present, so a lone 0.8-wide bar covers 80% of the panel and reads as a filled background
# rather than as a measurement. Padding the category axis out to a few slots restores a bar's proportions.
_MIN_CATEGORY_SLOTS = 4


def _pad_sparse_category_axis(ax, n_categories: int, *, horizontal: bool) -> None:
    """Widen the category axis when too few categories would let the bars swell to fill the panel."""
    if not 0 < n_categories < _MIN_CATEGORY_SLOTS:
        return
    centre = (n_categories - 1) / 2.0
    lo, hi = centre - _MIN_CATEGORY_SLOTS / 2.0, centre + _MIN_CATEGORY_SLOTS / 2.0
    if horizontal:
        # The y axis is already inverted here, so the limits go back in descending order to keep it that way.
        ax.set_ylim(hi, lo)
    else:
        ax.set_xlim(lo, hi)


# An in-axes legend is only free when it fits in the corner it lands in. Six entries of
# ``label_5_long_descriptive_name (ECE=0.203, n=3,000)`` produce a box WIDER than the panel, sitting on top of
# the curves it names -- the reader loses the data to read the key. ``legend_outside`` already exists for this
# and has to be asked for per panel, so the panels that grew long labels never got it. Decide from the labels
# and the panel's real width instead: the axes position is fixed by the gridspec before any draw, so this needs
# no renderer pass (measuring after ``draw()`` would fight constrained layout, which has not run yet).
_VIOLIN_LABEL_MAXLEN = 20  # rotated 30 deg under the axis, so a long name projects most of its length horizontally
_NETWORK_LABEL_MAXLEN = 24  # shorter than the bar-axis cap: these sit ON the graph, not along an axis
_LEGEND_MAX_PANEL_FRACTION = 0.62  # widest entry may claim at most this share of the panel
_LEGEND_MAX_INSIDE_ENTRIES = 8  # beyond this the stack is taller than most panels regardless of width
_LEGEND_CHAR_WIDTH_RATIO = 0.6  # mean glyph advance as a fraction of font size, DejaVu Sans at these sizes
_LEGEND_HANDLE_INCHES = 0.45  # marker/line sample plus padding, left of the text


def _legend_overflows(ax, labels, *, fontsize: int = 8) -> bool:
    """True when this legend cannot sit inside ``ax`` without covering a meaningful part of it."""
    texts = [str(lbl) for lbl in labels if lbl]
    if not texts:
        return False
    if len(texts) > _LEGEND_MAX_INSIDE_ENTRIES:
        return True
    fig_w_in = float(ax.figure.get_size_inches()[0])
    panel_w_in = max(0.5, float(ax.get_position().width) * fig_w_in)
    widest_in = (max(len(t) for t in texts) * _LEGEND_CHAR_WIDTH_RATIO * fontsize) / 72.0 + _LEGEND_HANDLE_INCHES
    return widest_in > _LEGEND_MAX_PANEL_FRACTION * panel_w_in


def _panel_is_rightmost(ax) -> bool:
    """True when nothing in the figure is drawn to the right of ``ax``."""
    x1 = float(ax.get_position().x1)
    return all(float(other.get_position().x1) <= x1 + 1e-3 for other in ax.figure.axes)


def _place_legend(ax, handles=None, labels=None, *, force_outside: bool = False, ncol: int = 1, has_secondary_y: bool = False):
    """Draw the legend inside the panel, or beside it when it would not fit and there is room beside it.

    "Outside" is only an improvement where the space actually exists. Anchoring past the right edge of an INNER
    grid column parks the legend in the narrow gutter between panels, where it overlaps its neighbour; doing it
    on a panel with a right-hand axis lands it straight on that axis's label. In both cases an in-axes legend at
    a smaller size, over a more opaque box, is the lesser harm -- so the automatic promotion is gated on the
    panel being the rightmost one and having no secondary axis. An explicit ``legend_outside`` from the builder
    is still honoured unconditionally: the builder knows its own layout.
    """
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    overflows = _legend_overflows(ax, labels)
    if force_outside or (overflows and _panel_is_rightmost(ax) and not has_secondary_y):
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, framealpha=0.7, ncol=max(1, int(ncol)))
    else:
        ax.legend(handles, labels, loc="best", fontsize=7 if overflows else 8, framealpha=0.85 if overflows else 0.7, ncol=max(1, int(ncol)))


def _set_panel_title(ax, title) -> None:
    """Set an axes title, wrapped to the panel's REAL width by measuring the font, and capped in size.

    Three behaviours the flat ``textwrap.wrap(s, _TITLE_WRAP_CHARS)`` form got wrong, in the order they were
    found:

    * ``textwrap.wrap`` treats ``\\n`` as ordinary whitespace, so any explicit line break the CALLER put in
      the title was silently collapsed and re-flowed. Each line is now wrapped independently, so deliberate
      breaks survive.
    * ``_TITLE_WRAP_CHARS`` is calibrated for a ~6-inch panel but was applied at any panel width, so a wide
      panel folded its title into a narrow ragged column with most of the width left empty.
    * Scaling that budget with panel width fixed the second point but kept the assumption underneath it --
      that every character is as wide as the calibration's average. A diagnostic title is mostly digits,
      percent signs and CamelCase identifiers, none of which are that average, so the line still broke in the
      wrong place. The budget is now the measured width of the actual glyphs.
    """
    if not title:
        return
    # ``ax.get_position().width`` is the axes' width as a FRACTION of the figure, so multiplying by the
    # figure width yields this panel's real width in inches -- the quantity being filled.
    try:
        panel_w = float(ax.get_position().width) * float(ax.figure.get_size_inches()[0])
    except Exception:
        logger.debug("could not measure panel width for title wrapping; falling back to the unscaled budget", exc_info=True)
        panel_w = None
    fallback = panel_title_wrap_chars((panel_w, 0), 1) if panel_w else panel_title_wrap_chars(None, 1)
    # Explicit rather than ``panel_w or REF``: a measured width of exactly 0 is a degenerate axes, not a
    # missing measurement, and the two deserve the same fallback for DIFFERENT reasons -- said once, here.
    width_in = panel_w if (panel_w is not None and panel_w > 0.0) else _TITLE_REF_WIDTH_IN
    lines = wrap_text_to_width(title, fontsize=_TITLE_FONTSIZE, width_in=width_in, fallback_chars=fallback)
    ax.set_title("\n".join(lines), fontsize=_TITLE_FONTSIZE)


# Wrap budgets, mirroring the plotly renderer: ~90 chars for the full-figure suptitle, ~110 for the
# wider caption band beneath it.
_SUPTITLE_WRAP_CHARS = SUPTITLE_WRAP_CHARS
_CAPTION_WRAP_CHARS = CAPTION_WRAP_CHARS
# Caption point size, shared by the renderer and the width measurement that wraps it.
_CAPTION_FONTSIZE = CAPTION_FONTSIZE
# A point within this fraction of an axis edge gets its inline label flipped to the other side, so the text
# stays inside the panel instead of being clipped mid-word.
_EDGE_LABEL_FLIP_FRACTION = 0.08
# Bar-category labels are budgeted from the axis's MEASURED length (``ticks_that_fit`` plus the pitch one
# label needs), not from a flat "past 25 categories keep 20" -- that cancelled out the height the
# slice-finder builders buy per bar.
# _BAR_LABEL_MAXLEN / truncate_bar_label come from ._shared_helpers (one definition, both backends).


# Above this many raw scatter points, cap (downsample preserving extremes) and rasterize so the saved vector
# file (pdf/svg) doesn't embed millions of DOM nodes (3.2s + bloat at 2M).
# Pre-bin a raw histogram above this n with np.histogram + ax.bar instead of letting ax.hist re-scan full n.
# Above this many heatmap cells the per-cell text turns to unreadable soup; skip it (also keeps the plotly
# per-annotation O(cells) loop from stalling on a degenerate huge-K grid).


def _err_to_mpl(err):
    """Spec error-bar field -> matplotlib ``errorbar`` yerr/xerr arg.

    A single array is symmetric; a (lower, upper) pair is asymmetric and matplotlib wants a (2, N) array of the
    DISTANCES from the point (Wilson CIs are asymmetric, so the spec carries absolute distances per side)."""
    if err is None:
        return None
    if isinstance(err, tuple):
        return np.vstack([np.asarray(err[0], dtype=float), np.asarray(err[1], dtype=float)])
    return np.asarray(err, dtype=float)


class MatplotlibRenderer:
    """Renders a ``FigureSpec`` to a ``matplotlib.figure.Figure`` via a headless ``Figure`` + ``FigureCanvasAgg`` (never through pyplot), dispatching each panel to a per-type ``_<kind>`` method."""

    backend = "matplotlib"
    # Bound at the bottom of this module from ``._matplotlib_scatter``.
    _scatter: ClassVar[Any]

    def render(self, spec: FigureSpec, *, static_legend: bool = False) -> Any:
        """Build the grid of subplots described by ``spec`` (row/col ratios, optional suptitle/caption), render every panel into its cell, and return the assembled ``Figure``."""
        # static_legend is a plotly-only concept (see PlotlyRenderer.render); matplotlib legends
        # are always static, so this backend accepts and ignores the flag to satisfy the Renderer Protocol.
        del static_legend
        # REMOVED ``matplotlib.use("Agg", force=False)``
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
        # band on figsize=(15, 4-5) - visible collision in saved PNGs.
        # constrained_layout reserves space for the suptitle.
        #
        # And whenever a panel carries axis furniture that needs room reserved for it. Many builders pass
        # ``suptitle=""`` deliberately, so those figures had NO layout engine: nothing reserved space for
        # 45-degree tick labels, a category axis of long names, or a colorbar. ``savefig(bbox_inches="tight")``
        # rescued the saved file by growing the canvas -- silently producing a figure that is not the
        # requested figsize -- and the interactive ``display(fig)`` path has no tight bbox at all, so in a
        # notebook those labels were simply clipped. Measured on a busy four-panel figure, the engine costs
        # ~156 ms against a 714 ms render (not the ~800 ms this comment used to claim), and a figure with no
        # such furniture still pays nothing.
        layout = "constrained" if (spec.constrained_layout or spec.suptitle or spec.caption or _needs_layout_engine(spec)) else None
        fig_kwargs: dict[str, Any] = {"figsize": spec.figsize, "layout": layout}
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
            row_axes: list = []
            for c, panel in enumerate(row):
                if panel is None:
                    row_axes.append(None)
                    continue
                share_x = axes_grid[0][c] if (spec.sharex and r > 0 and c < len(axes_grid[0]) and axes_grid[0][c] is not None) else None
                share_y = row_axes[0] if (spec.sharey and c > 0 and row_axes and row_axes[0] is not None) else None
                ax = fig.add_subplot(gs[r, c], sharex=share_x, sharey=share_y)
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

        # Title and caption both live in bands reserved OUTSIDE the axes rectangle, and both bands are measured
        # from the wrapped line count rather than assumed. constrained_layout is documented to make room for a
        # suptitle, but it under-reserves for a multi-line one: a three-line model identity on a 12x6 figure
        # started 25 px BELOW the top of the axes, printing the run's own metrics across its own chart.
        _dpi = fig.get_dpi()
        _h_px = fig.get_size_inches()[1] * (_dpi if _dpi > 0 else 100.0)
        _top_band = 0.0
        _bottom_band = 0.0
        _sup_text = ""
        _cap = ""
        if spec.suptitle:
            # Broken at the real edge of the canvas, not at a character count. The fixed ~90-char budget this
            # replaces was calibrated at one width and one font size and then applied at every width, so a
            # wide figure folded a verbose model identity into a narrow ragged column with a third of the
            # figure unused -- the text was being broken by an assumption rather than by the page.
            _sup_lines = wrap_text_to_width(spec.suptitle, fontsize=spec.suptitle_fontsize, width_in=fig.get_size_inches()[0], fallback_chars=_SUPTITLE_WRAP_CHARS)
            _sup_text = "\n".join(_sup_lines)
            _top_band = min(0.35, (len(_sup_lines) * (spec.suptitle_fontsize + 3.0) + 12.0) / _h_px)
        if spec.caption:
            # How-to-read footnote, small + dim, in a reserved bottom band so it never overlaps the x-axis label.
            # Wrap through the shared helper so an author-supplied line break in a caption SURVIVES.
            # ``textwrap.wrap`` treats a newline as ordinary whitespace and re-flows it away, silently
            # collapsing the deliberate structure captions are written with (one clause per line, a VERDICT
            # sentence on its own).
            _cap_lines = wrap_text_to_width(spec.caption, fontsize=_CAPTION_FONTSIZE, width_in=fig.get_size_inches()[0], fallback_chars=_CAPTION_WRAP_CHARS)
            _cap = "\n".join(_cap_lines)
            _bottom_band = min(0.30, (len(_cap_lines) * 11.0 + 12.0) / _h_px)
        if _top_band or _bottom_band:
            _eng = fig.get_layout_engine()
            if _eng is not None:
                try:
                    _eng.set(rect=(0.0, _bottom_band, 1.0, 1.0 - _bottom_band - _top_band))  # type: ignore[call-arg]  # matplotlib stubs type _eng as the base LayoutEngine; constrained_layout was forced on above so this is always a ConstrainedLayoutEngine, whose .set() does accept rect
                except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                    logger.debug("suppressed: %s", e)
                    pass
        if _sup_text:
            # Centred in its own band. Left to constrained_layout's own placement the multi-line case lands on
            # the axes; anchored here it cannot, because the band is exactly what the axes rectangle excludes.
            fig.suptitle(_sup_text, fontsize=spec.suptitle_fontsize, y=1.0 - _top_band * 0.5, va="center")
        if _cap:
            fig.text(0.5, _bottom_band * 0.5, _cap, ha="center", va="center", fontsize=_CAPTION_FONTSIZE, color="0.35")
        return fig

    def save(self, fig: Any, path: str, fmt: str) -> None:
        """Save ``fig`` to ``path`` in ``fmt`` (png/pdf/svg/jpg/jpeg), using a tight bbox + small padding so suptitles, y-tick labels and out-of-axes annotations aren't clipped."""
        fmt = fmt.lower()
        if fmt not in ("png", "pdf", "svg", "jpg", "jpeg"):
            raise ValueError(f"matplotlib doesn't support format {fmt!r}; " "supported: png/pdf/svg/jpg")
        # bbox_inches="tight" + small pad guarantees suptitle, ytick labels and any annotations outside the axes
        # box land inside the saved PNG. Without this the renderer crops at the figure box and long ytick labels
        # (FI plots) / suptitles get clipped.
        #
        # bench-attempt-rejected 2026-09-06: skipping this pass when constrained layout already ran is 1.30x on
        # savefig (2.10 s -> 1.62 s per multilabel figure) and the outside legend survives -- constrained layout
        # does reserve room for it. Panel TITLES do not survive: constrained layout allots vertical space for a
        # title but never widens the figure for one, so on a panel pushed right by long y-tick labels the
        # centred title overhangs the figure edge and is cropped mid-word ("... support n + P(y=1|le"). The
        # tight crop was what grew the canvas to fit it. Rejected until the title's wrap budget is measured
        # AFTER layout rather than before it (the budget currently reads the pre-layout axes width, so it is
        # too generous exactly on the panels where this bites); the bench is profiling/render_every_chart.py.
        fig.savefig(ensure_parent_dir(path), format=fmt, bbox_inches="tight", pad_inches=0.15)

    def show(self, fig: Any) -> None:
        """Display ``fig`` inline in an IPython/Jupyter kernel via ``IPython.display.display``, or best-effort pop a GUI window outside a kernel when matplotlib is in interactive mode; a no-op in headless / non-interactive contexts."""
        # The renderer builds figures via ``Figure()`` + ``FigureCanvasAgg`` (never through pyplot), so they
        # have no pyplot manager and no ``.number`` -- ``plt.figure(fig.number)`` would raise. In an IPython
        # kernel the right call is ``IPython.display.display(fig)``, which renders inline without pyplot. Outside
        # a kernel, attach to a pyplot manager and show a window (best-effort; headless/no-display is a no-op).
        import sys
        if "IPython" in sys.modules:
            try:
                ip = sys.modules["IPython"].get_ipython()
            except Exception as e:
                logger.debug("IPython.get_ipython() probe failed: %s", e)
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
            if manager is not None:
                manager.canvas.figure = fig
                fig.set_canvas(manager.canvas)
            plt.show()
        except Exception as e:
            logger.debug("MatplotlibRenderer.show() no-op (no interactive display): %s: %s", type(e).__name__, e)

    # ------------------------------------------------------------------
    # Per-panel dispatch
    # ------------------------------------------------------------------

    def _render_panel(self, ax, panel, fig, cbar_axes=None) -> None:
        """Dispatch a single panel spec to its rendering method by isinstance; raises ``TypeError`` for an unrecognized panel type."""
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
        """Render a free-text panel (no axes/data): centered text wrapped to the panel's own width, no ticks/spines."""
        # Wrap here rather than via matplotlib's `wrap=True`, which measures against the FIGURE box and never breaks
        # long tokens -- see wrap_annotation_text for the measured numbers.
        # Measured BEFORE constrained layout runs, which is a known conservatism rather than the overflow
        # it looks like: measured on a heatmap-plus-colorbar sibling, the cell this panel ends up in is
        # WIDER after layout (4.38in) than the rectangle wrapped against (3.52in), so the text under-uses
        # the panel and never spills out of it. Rewrapping in a draw callback would recover that width.
        bbox = ax.get_window_extent()
        _dpi = float(ax.figure.dpi)
        panel_w_in = float(bbox.width) / (_dpi if _dpi > 0 else 100.0)
        text = wrap_annotation_text(p.text, panel_w_in, p.fontsize)
        family = "monospace" if getattr(p, "monospace", False) else None
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=p.fontsize, transform=ax.transAxes, family=family)
        _set_panel_title(ax, p.title)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _histogram(self, ax, p: HistogramPanelSpec) -> None:
        """Render a histogram panel: uses pre-binned ``bin_centers`` when given (or pre-bins above ``_HIST_PREBIN_THRESHOLD`` rather than letting ``ax.hist`` re-scan the full array), else falls back to ``ax.hist`` on finite values only; optionally overlays a fitted Normal PDF."""
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
                if isinstance(p.bin_width, np.ndarray):
                    width = np.asarray(p.bin_width, dtype=float)
                else:
                    # `is not None`, matching the plotly twin: `bin_width=0.0` is a deliberate spec value, and
                    # `or` read it as "unset" and derived a width from the centre spacing instead -- one spec,
                    # two pictures.
                    width = float(p.bin_width if p.bin_width is not None else ((bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 1.0))
            colors_kw: dict[str, Any] = {"color": p.color}
            if p.bar_colors is not None:
                cm = matplotlib.colormaps[resolve_heatmap_cmap(p.colormap)]  # same resolver as every other lookup; a sentinel default must not reach matplotlib
                _h_min = float(np.min(p.bar_colors))
                _h_max = float(np.max(p.bar_colors))
                if _h_max <= _h_min:
                    _h_max = _h_min + 1.0
                colors_kw = {"color": cm((np.asarray(p.bar_colors) - _h_min) / (_h_max - _h_min))}
            ax.bar(bin_centers, heights, width=width, align="center", edgecolor="white", linewidth=0.5, **colors_kw)
            if len(bin_centers) > 0:
                assert width is not None
                overlay_x_lo, overlay_x_hi = histogram_bar_extent(bin_centers, width)
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
                assert overlay_x_hi is not None
                x_grid = np.linspace(overlay_x_lo, overlay_x_hi, 200)
                normal_pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)
                # `overlay_label=""` is a deliberate blank, which plotly honours; `or` substituted the auto label.
                # This cluster recognises the empty-string-means-no-label distinction elsewhere (see
                # `calibration.py`'s `colorbar_label`).
                label = p.overlay_label if p.overlay_label is not None else f"Normal(mu={mu:.2g}, sigma={sigma:.2g})"
                ax.plot(x_grid, normal_pdf, "r--", linewidth=1.4, label=label)
                _place_legend(ax)

        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        _set_panel_title(ax, p.title)
        # An empty bin is not drawn small on a log axis, it is not drawn at all, and nothing said so: a
        # 30-bin histogram of a tight cluster plus one far outlier silently loses 24 of its bars.
        _log_note = log_axis_dropped_note([patch.get_height() for patch in ax.patches], p.yscale)
        if _log_note:
            ax.text(0.99, 0.99, _log_note, ha="right", va="top", transform=ax.transAxes, fontsize=7, color="0.35")
        ax.set_yscale(p.yscale)
        if p.yscale == "linear":
            # A short panel gets matplotlib's sparsest tick set -- two or three labels for the whole axis, which
            # is not enough to read a bar's value off. Ask for a denser set that still lands on round numbers.
            from matplotlib.ticker import MaxNLocator

            ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4, steps=[1, 2, 2.5, 5, 10]))
        if p.yscale == "log":
            # matplotlib's default log locator only labels decades, so a histogram spanning a decade and a bit
            # gets ONE labelled tick -- the axis then carries a scale name and no readable values. Asking for
            # several ticks plus the 2/5 subdivisions puts numbers back on it.
            from matplotlib.ticker import LogFormatterSciNotation, LogLocator

            ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=8))
            ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
        if p.xlim is not None:
            ax.set_xlim(*p.xlim)
        if p.grid:
            ax.grid(True, alpha=0.3)

    def _heatmap(self, ax, p: HeatmapPanelSpec, fig) -> None:
        """Render a matrix heatmap: cell text (auto-flipped color by luminance) when the grid is small enough, iso-value threshold contours, and an optional trend/y=x line mapped from value-space into bin-index space via the panel's own binning range."""
        import matplotlib
        cmap_name = resolve_heatmap_cmap(p.colormap)
        cm = matplotlib.colormaps[cmap_name]
        # A density panel carrying ``trend_xy`` (the regression pred-vs-true heatmap) reads "bottom-up"
        # (row 0 = lowest value), so it needs origin="lower"; other heatmaps (confusion / drift) keep the
        # default top-down matrix orientation.
        _heatmap_origin = "lower" if getattr(p, "trend_xy", None) is not None else "upper"
        # Explicit colour bounds when the builder pinned them: a diverging colormap's midpoint means nothing
        # unless the scale is anchored, and autoscale anchors it to whatever the data happens to span.
        _clim = {}
        if p.color_vmin is not None:
            _clim["vmin"] = p.color_vmin
        if p.color_vmax is not None:
            _clim["vmax"] = p.color_vmax
        im = ax.imshow(p.matrix, cmap=cm, aspect="auto", origin=_heatmap_origin, **_clim)
        # How many names the axis can actually hold, from its real size rather than a fixed 8. A drift heatmap
        # grows its figure with the feature count (40 rows over ~14 inches) and then named 8 of them, so the
        # height it bought went to unlabelled rows. Labels are truncated as well, which the bar branches have
        # always done and this one never did -- a generated feature name runs off the left edge otherwise.
        # Deferred until after the colorbar, which shrinks the axes it is attached to: budgeting against the
        # pre-colorbar width bought a fifth more labels than the axes ends up able to hold.
        def _apply_tick_budget() -> None:
            """Set both axes' ticks to what the axes can hold, measured at its post-colorbar size."""
            try:
                _pos = ax.get_position()
                _fig_w, _fig_h = (float(v) for v in ax.figure.get_size_inches())
                _w_in: Optional[float] = float(_pos.width) * _fig_w
                _h_in: Optional[float] = float(_pos.height) * _fig_h
            except Exception:
                logger.debug("could not measure heatmap axes for tick budgeting; falling back to the fixed cap", exc_info=True)
                _w_in = _h_in = None
            _xt = _thin_tick_positions(len(p.col_labels), ticks_that_fit(_w_in, len(p.col_labels), pitch_in=rotated_tick_pitch_in(_HEATMAP_TICK_FONTSIZE, 45)))
            ax.set_xticks(_xt)
            ax.set_xticklabels([truncate_bar_label(p.col_labels[i]) for i in _xt], rotation=45, ha="right", fontsize=_HEATMAP_TICK_FONTSIZE)
            _yt = _thin_tick_positions(len(p.row_labels), ticks_that_fit(_h_in, len(p.row_labels), pitch_in=rotated_tick_pitch_in(_HEATMAP_TICK_FONTSIZE, 0)))
            ax.set_yticks(_yt)
            ax.set_yticklabels([truncate_bar_label(p.row_labels[i]) for i in _yt], fontsize=_HEATMAP_TICK_FONTSIZE)
        rng = _finite_range(p.matrix)
        if p.cell_text is not None and rng is not None and p.matrix.size <= _HEATMAP_CELL_TEXT_MAX:
            from mlframe.reporting.colors import auto_text_colors_batch
            # Compute global vmin / vmax so each cell's text color reflects
            # its position in the actual color range — naive
            # ``< 0.5`` threshold fails when the matrix range is e.g.
            # [0.3, 0.85] (all values map to the high-luminance end of
            # the colormap and white text becomes invisible).
            mat = p.matrix
            vmin, vmax = rng
            # The bounds the CELLS were drawn with, when the builder pinned them. The text colour is chosen by
            # sampling the colormap at the cell's position in the scale, so it has to use the same scale the
            # image did -- otherwise a pinned range paints pale cells while the text colour still believes the
            # data range, and white labels land on a near-white fill.
            if p.color_vmin is not None:
                vmin = float(p.color_vmin)
            if p.color_vmax is not None:
                vmax = float(p.color_vmax)
            # One vectorized colormap sample for the whole grid instead of one matplotlib call per cell
            # (bit-identical to the per-cell auto_text_color -- same pattern PlotlyRenderer._heatmap uses).
            text_colors = auto_text_colors_batch(np.where(np.isfinite(mat), mat, vmin), cmap_name, vmin=vmin, vmax=vmax)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    ax.text(j, i, format(p.cell_text[i, j], p.text_format), ha="center", va="center", fontsize=7, color=text_colors[i, j])
        # Iso-value contour overlays at named matrix levels (PSI 0.10 / 0.25 triage lines on the drift heatmap).
        # Contour coords are the imshow cell-center grid (0..ncols-1, 0..nrows-1) so lines land between cells.
        if p.threshold_contours:
            mat = np.asarray(p.matrix, dtype=float)
            if mat.ndim == 2 and mat.shape[0] >= 2 and mat.shape[1] >= 2:
                gx, gy = np.meshgrid(np.arange(mat.shape[1]), np.arange(mat.shape[0]))
                # Hoisted out of the loop: both are full-matrix reductions over the same unchanged matrix,
                # so recomputing them per contour level was O(levels * cells) for an O(cells) answer. The
                # plotly twin already hoists them.
                lo, hi = float(np.nanmin(mat)), float(np.nanmax(mat))
                for _entry in p.threshold_contours:
                    level, color = _entry[0], _entry[1]
                    dash = _entry[2] if len(_entry) > 2 else "solid"
                    label = _entry[3] if len(_entry) > 3 else ""
                    if lo < level < hi:  # contour only exists when the level is crossed
                        cs = ax.contour(gx, gy, mat, levels=[level], colors=[color], linewidths=1.4,
                                        linestyles={"solid": "-", "dash": "--", "dot": ":", "dashdot": "-."}.get(dash, "-"))
                        if label:
                            ax.clabel(cs, fmt={level: label}, fontsize=7)
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
                    # Shared with the plotly renderer so the two backends cannot drift on this map again.
                    _to_idx = heatmap_value_to_index(_lo, _hi, _nb)
                    # y=x reference in index space (origin="lower" -> bottom-left to top-right).
                    ax.plot([0, _nb - 1], [0, _nb - 1], color="0.4", linestyle=":", linewidth=1.0, label="y=x")
                    ends = robust_fit_endpoints(_xv, _yv, p.trend_line)
                    if ends is not None:
                        (tx0, ty0), (tx1, ty1) = ends
                        ax.plot(
                            [_to_idx(tx0), _to_idx(tx1)], [_to_idx(ty0), _to_idx(ty1)],
                            color=TREND_LINE, linestyle="-", linewidth=1.6,
                            label=f"robust fit ({p.trend_line})",
                        )
                    ax.set_xlim(-0.5, _nb - 0.5)
                    ax.set_ylim(-0.5, _nb - 0.5)
                    _place_legend(ax)
        cbar = fig.colorbar(im, ax=ax)
        if p.colorbar_label:
            cbar.set_label(p.colorbar_label)
        _apply_tick_budget()
        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        _set_panel_title(ax, p.title)

    def _confusion_margins(self, ax, p: ConfusionMarginsPanelSpec, fig) -> None:
        """Render a confusion matrix as a 2x2 small-multiple (predicted-volume bar on top, true-support bar on the right) by subdividing the panel's own subplotspec into sub-axes, replacing the placeholder ``ax``."""
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
        # Thin to the shared ceiling on BOTH axes. One tick per class smears past ~30 classes, and the plotly
        # twin already thins to _HEATMAP_MAX_TICKS -- so a large-K confusion matrix rendered with a readable
        # axis on one backend and an unreadable band on the other, from the same spec.
        _xt = _thin_tick_positions(len(p.col_labels))
        _yt = _thin_tick_positions(len(p.row_labels))
        ax_hm.set_xticks(_xt)
        ax_hm.set_xticklabels([p.col_labels[i] for i in _xt], rotation=45, ha="right", fontsize=8)
        ax_hm.set_yticks(_yt)
        ax_hm.set_yticklabels([p.row_labels[i] for i in _yt], fontsize=8)
        ax_hm.set_xlabel(p.xlabel)
        ax_hm.set_ylabel(p.ylabel)
        rng = _finite_range(p.matrix)
        if p.cell_text is not None and rng is not None and p.matrix.size <= _HEATMAP_CELL_TEXT_MAX:
            from mlframe.reporting.colors import auto_text_colors_batch
            vmin, vmax = rng
            # One vectorized colormap sample for the whole grid instead of one matplotlib call per cell.
            text_colors = auto_text_colors_batch(np.where(np.isfinite(p.matrix), p.matrix, vmin), cmap_name, vmin=vmin, vmax=vmax)
            for i in range(K):
                for j in range(p.matrix.shape[1]):
                    ax_hm.text(j, i, format(p.cell_text[i, j], p.text_format), ha="center", va="center", fontsize=7, color=text_colors[i, j])

        pos = np.arange(K)
        # Top bar: predicted-class volume, aligned to the heatmap columns (shared x, ticks hidden -- the heatmap owns them).
        ax_top.bar(pos, np.asarray(p.col_margin, dtype=float), color=CONFUSION_COL_MARGIN, width=0.8)
        ax_top.set_xlim(-0.5, K - 0.5)
        ax_top.set_xticks([])
        ax_top.tick_params(axis="y", labelsize=7)
        ax_top.set_ylabel(p.col_margin_label, fontsize=7)
        # Right bar: per-true-class support, aligned to the heatmap rows (imshow y runs top->bottom, so invert).
        ax_right.barh(pos, np.asarray(p.row_margin, dtype=float), color=CONFUSION_ROW_MARGIN, height=0.8)
        ax_right.set_ylim(-0.5, K - 0.5)
        ax_right.invert_yaxis()
        ax_right.set_yticks([])
        ax_right.tick_params(axis="x", labelsize=7, rotation=45)
        ax_right.set_xlabel(p.row_margin_label, fontsize=7)

        cbar = fig.colorbar(im, ax=ax_right, fraction=0.25, pad=0.35)
        if p.colorbar_label:
            cbar.set_label(p.colorbar_label, fontsize=8)
        title = p.title if not p.note else f"{p.title}\n{p.note}"
        _set_panel_title(ax_top, title)

    def _bar(self, ax, p: BarPanelSpec) -> None:
        """Render a bar panel: grouped bars when ``values`` is a tuple of series, single-series otherwise; supports a perpendicular reference line and thins x-tick labels above 25 categories so they don't overlap."""
        horizontal = p.orientation == "horizontal"
        pos = np.arange(len(p.categories))
        if isinstance(p.values, tuple):
            # Grouped bars.
            n_series = len(p.values)
            thickness = 0.8 / n_series
            for i, series in enumerate(p.values):
                offset = (i - (n_series - 1) / 2) * thickness
                kw: dict = {}
                if p.colors is not None and i < len(p.colors):
                    kw["color"] = p.colors[i]
                if p.hatches is not None and i < len(p.hatches) and p.hatches[i]:
                    kw["hatch"] = p.hatches[i]
                lbl = p.series_labels[i] if p.series_labels else None
                if horizontal:
                    ax.barh(pos + offset, series, height=thickness, label=lbl, **kw)
                else:
                    ax.bar(pos + offset, series, width=thickness, label=lbl, **kw)
            if p.series_labels:
                _place_legend(ax)
        else:
            # A colours tuple as long as ``values`` is PER-BAR, not per-series: matplotlib's bar/barh accept a
            # sequence. Reading ``colors[0]`` painted every bar the colour of the first one.
            kw = {"color": _bar_colors(p.colors, p.values)}
            if p.hatches and p.hatches[0]:
                kw["hatch"] = p.hatches[0]
            if p.value_err is not None:
                kw["xerr" if horizontal else "yerr"] = np.vstack(p.value_err)
                kw["error_kw"] = dict(ecolor="black", elinewidth=0.9, capsize=2)
            if horizontal:
                ax.barh(pos, p.values, **kw)
            else:
                ax.bar(pos, p.values, **kw)

        # Reference line perpendicular to the bars (global metric across a per-segment bar). axvline for
        # horizontal bars (value axis is x), axhline for vertical bars (value axis is y).
        if p.hline is not None:
            hval, hcolor, hlabel = p.hline
            # A symmetric band is two lines and ONE legend entry: the mirrored line passes label=None so the
            # key does not list the same threshold twice.
            values = (hval, -hval) if p.hline_symmetric else (hval,)
            for _i, _v in enumerate(values):
                _lbl = (hlabel or None) if _i == 0 else None
                if horizontal:
                    ax.axvline(_v, color=hcolor, linestyle="--", linewidth=1.3, label=_lbl)
                else:
                    ax.axhline(_v, color=hcolor, linestyle="--", linewidth=1.3, label=_lbl)
            if hlabel:
                _place_legend(ax)

        if horizontal:
            # Thin AND truncate, as the vertical branch below and both plotly orientations do. A 200-category horizontal feature-importance chart otherwise smears its
            # y axis into an unreadable band of overlapping text, and a long generated feature name runs off
            # the left edge. The bars stay 1-per-category; only the LABELS are subsampled.
            # Against the axis's MEASURED length, not the category count. The count-based "past 25, keep 20"
            # cancelled out the height a builder deliberately bought: slice_finder and
            # category_discriminability both grow the figure 0.5in per bar, so at top_k=40 the axis had room
            # for every label and the renderer still hid twenty of them, leaving twenty unidentifiable bars.
            n_cat = len(p.categories)
            _cats = [truncate_bar_label(c, keep_tail=p.label_keep_tail) for c in p.categories]
            _keep = _thin_tick_positions(n_cat, ticks_that_fit(_measured_axis_in(ax, horizontal=True), n_cat,
                                                               pitch_in=rotated_tick_pitch_in(_HEATMAP_TICK_FONTSIZE, 0)))
            ax.set_yticks(pos[np.asarray(_keep, dtype=np.int64)])
            ax.set_yticklabels([_cats[i] for i in _keep], fontsize=_HEATMAP_TICK_FONTSIZE)
            ax.invert_yaxis()  # first category on top -> worst-first ranking reads top-down
            _pad_sparse_category_axis(ax, len(p.categories), horizontal=True)
        else:
            # Thin the x-tick labels when there are many categories so they don't overlap into an
            # unreadable smear (e.g. a 50-lag residual-ACF bar chart). Keep ~20 evenly-spaced labels;
            # the bars themselves stay 1-per-category, only the LABELS are subsampled.
            # TRUNCATE here too. The horizontal branch's comment above says this branch already did, and the
            # plotly twin truncates on both orientations -- but it passed labels through untouched, so a
            # pathological generated column name ran off the bottom of the axis: exactly what
            # ``truncate_bar_label`` exists as a safety valve against. The two thinning constants are the module
            # ones now rather than 25 and 20 written out again, so the same numbers stop living in four places.
            n_cat = len(p.categories)
            _cats_v = [truncate_bar_label(c, keep_tail=p.label_keep_tail) for c in p.categories]
            # Unrotated labels on a horizontal axis sit end to end, so what has to fit is the widest one's
            # WIDTH; rotated ones are parallel lines and clear each other at a line height perpendicular to
            # themselves. Either way the budget comes from the axis's measured length, not the count.
            _axis_in = _measured_axis_in(ax, horizontal=False)
            _rotation = float(p.xtick_rotation or 0.0)
            if not _rotation:
                _flat = label_width_pitch_in(_cats_v, _HEATMAP_TICK_FONTSIZE)
                # Rotating buys labels: upright ones sit end to end and need the widest label's WIDTH, while
                # 45-degree ones are parallel lines clearing at a line height. Only take the rotation when
                # upright genuinely does not fit, which is also what the plotly twin does.
                if ticks_that_fit(_axis_in, n_cat, pitch_in=_flat) < n_cat:
                    _rotation = 45.0
            _pitch = rotated_tick_pitch_in(_HEATMAP_TICK_FONTSIZE, _rotation) if _rotation else label_width_pitch_in(_cats_v, _HEATMAP_TICK_FONTSIZE)
            _keep_v = _thin_tick_positions(n_cat, ticks_that_fit(_axis_in, n_cat, pitch_in=_pitch))
            ax.set_xticks(pos[np.asarray(_keep_v, dtype=np.int64)])
            ax.set_xticklabels(
                [_cats_v[i] for i in _keep_v],
                rotation=_rotation,
                ha="right" if _rotation else "center", fontsize=_HEATMAP_TICK_FONTSIZE,
            )
            _pad_sparse_category_axis(ax, len(p.categories), horizontal=False)
        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        _set_panel_title(ax, p.title)
        if p.grid:
            ax.grid(True, alpha=0.3, axis="x" if horizontal else "y")

    def _line(self, ax, p: LinePanelSpec, fig=None) -> None:
        """Render a multi-series line panel: per-series x grids, styles, colors, secondary y-axis (twinx, lazily created), fill bands, vspans/vlines, and point markers; merges legends from both axes and optionally places the legend outside the panel."""
        from mlframe.reporting.colors import line_color

        ys = p.y if isinstance(p.y, tuple) else (p.y,)
        # Per-series x: a tuple of x arrays parallel to ``y`` (ROC overlays with different fpr grids); else shared.
        xs_per_series = isinstance(p.x, tuple)
        labels = p.series_labels or (None,) * len(ys)
        styles = p.line_styles or ("-",) * len(ys)
        cols = p.colors if p.colors else tuple(line_color(i) for i in range(len(ys)))
        sec = _per_series_flags(p.secondary_y, len(ys))
        fills = _per_series_flags(p.fill_to_baseline, len(ys))

        def _xi(i):
            """Return the x-array for series ``i``: per-series when ``p.x`` is a tuple of grids, else the shared grid."""
            return p.x[i] if xs_per_series else p.x

        # Lazily create the twin axis only when a series actually needs it.
        ax2 = ax.twinx() if any(sec) else None
        proxies = []  # legend proxies for labeled vspans

        if p.band is not None:
            lower, upper = np.asarray(p.band[0]), np.asarray(p.band[1])
            band_color = p.band_color or cols[0]
            ax.fill_between(_xi(0), lower, upper, color=band_color, alpha=0.2, label=p.band_label, zorder=0)

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
            # The label rides the legend entry above and is deliberately NOT repeated next to the marker. It used
            # to be both, so every operating point printed its own caption twice on one panel -- and the second
            # copy, anchored at the point with a fixed offset, ran off the axes on the PR panel and struck
            # through the F1 curve on the threshold panel. The marker's colour and symbol tie it to its entry.

        if ax2 is not None:
            ax2.set_ylabel(p.secondary_ylabel)
        if any(labels) or p.band_label or any(p.vlines or ()) or any(p.point_markers or ()) or proxies or (ax2 is not None and any(sec)):
            handles, leg_labels = ax.get_legend_handles_labels()
            if ax2 is not None:
                h2, l2 = ax2.get_legend_handles_labels()
                handles += h2
                leg_labels += l2
            handles += proxies
            leg_labels += [pr.get_label() for pr in proxies]
            if handles:
                _place_legend(ax, handles, leg_labels,
                              force_outside=bool(getattr(p, "legend_outside", False)),
                              ncol=int(getattr(p, "legend_ncol", 1)),
                              has_secondary_y=ax2 is not None)
        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        _set_panel_title(ax, p.title)
        if p.grid:
            ax.grid(True, alpha=0.3)
        # LinePanelSpec carries ylim and builders set it deliberately (decision_curve clips the y-window so
        # a steeply-diving reference cannot crush the informative band near 0), but only _scatter ever read
        # it -- a line panel's window was silently discarded on BOTH backends.
        if p.ylim is not None:
            ax.set_ylim(*p.ylim)
        if p.x_is_time:
            # The numeric x carries epoch NANOSECONDS, which read as "1.62e18" unless converted. The COUNT
            # comes from the axis rather than the helper's fixed six: a date label rotated 30 degrees needs
            # real room, and six of them crowd a narrow panel the same way a fixed cap crowded the heatmap.
            _date_pitch = rotated_tick_pitch_in(_HEATMAP_TICK_FONTSIZE, 30)
            _n_dates = ticks_that_fit(_measured_axis_in(ax, horizontal=False), 6, floor=2, pitch_in=_date_pitch)
            _tickvals, _ticktext = epoch_ns_ticks(_xi(0), n_ticks=_n_dates)
            if _tickvals is not None:
                ax.set_xticks(_tickvals)
                ax.set_xticklabels(_ticktext)
            # Rotate THIS axes only. ``fig.autofmt_xdate()`` is a FIGURE-level call: it hides the x tick
            # labels of every non-last-row axes and clears their xlabel, so on a multi-row grid it erased
            # the date ticks just computed here AND stripped the labels off unrelated panels sharing the
            # row. Its sole remaining contribution was rotation, which this does per-axes instead.
            ax.tick_params(axis="x", rotation=30)

    def _violin(self, ax, p: ViolinPanelSpec) -> None:
        """Render a per-group violin panel (medians shown when ``show_box``, extrema and mean markers suppressed).

        Groups are coloured from the shared palette so they match the plotly twin, which already cycles a
        colour per group -- matplotlib drew every violin in one default blue, losing the group identity the
        chart exists to show.
        """
        from mlframe.reporting.colors import line_color

        # ``ax.violinplot`` raises on an EMPTY group (it computes a kernel over no points), so a spec with one
        # empty group crashed matplotlib while plotly rendered the rest happily. Drop empty groups and mark
        # them in the tick label rather than failing the whole figure.
        kept = [(np.asarray(g, dtype=float), lab) for g, lab in zip(p.groups, p.group_labels)]
        kept = [(g[np.isfinite(g)], lab) for g, lab in kept]
        drawable = [(g, lab) for g, lab in kept if g.size > 0]
        if not drawable:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "no finite values in any group", ha="center", va="center", transform=ax.transAxes)
            _set_panel_title(ax, p.title)
            return
        parts = ax.violinplot([g for g, _ in drawable], showmeans=False, showextrema=False, showmedians=False)
        if p.show_box:
            # A median LINE here against a full quartile box on plotly meant one spec produced two different amounts
            # of information. Both backends now show the quartile box.
            ax.boxplot([g for g, _ in drawable], widths=0.12, showfliers=False, whis=(5, 95),
                       medianprops=dict(color="black", linewidth=1.2),
                       boxprops=dict(color="black", linewidth=0.9),
                       whiskerprops=dict(color="black", linewidth=0.9),
                       capprops=dict(color="black", linewidth=0.9))
        for i, body in enumerate(parts.get("bodies", [])):
            body.set_facecolor(line_color(i))
            body.set_alpha(0.6)
        labels = [lab for _, lab in drawable]
        empty = [str(lab) for g, lab in kept if g.size == 0]
        # Name the dropped groups in the title: a violin that silently vanishes reads as "this group has no
        # spread", which is a different statement from "this group has no data".
        _violin_title = f"{p.title} (no data: {', '.join(empty)})" if (empty and p.title) else p.title
        # Truncated AND thinned, the two guards the bar branches have and this one had neither of. A per-class
        # violin on 20 string class names put 20 rotated 30-char labels under the axis: they overlap into a
        # staircase and eat ~40% of the figure height. The violins stay one per group; only the LABELS
        # subsample, and the tail is kept because that is what distinguishes generated names.
        try:
            _panel_w = float(ax.get_position().width) * float(ax.figure.get_size_inches()[0])
        except Exception:
            logger.debug("could not measure violin panel width for label thinning", exc_info=True)
            _panel_w = None
        _keep = _thin_tick_positions(len(labels), ticks_that_fit(_panel_w, len(labels), pitch_in=rotated_tick_pitch_in(_HEATMAP_TICK_FONTSIZE, 30)))
        ax.set_xticks([i + 1 for i in _keep])
        ax.set_xticklabels([truncate_bar_label(labels[i], _VIOLIN_LABEL_MAXLEN, keep_tail=6) for i in _keep], rotation=30, ha="right", fontsize=_HEATMAP_TICK_FONTSIZE)
        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        _set_panel_title(ax, _violin_title)
        if p.grid:
            ax.grid(True, alpha=0.3, axis="y")

    def _network(self, ax, p: NetworkPanelSpec, fig) -> None:
        """Render a node-link network panel: edges as a single ``LineCollection`` (one draw call for O(E) edges, width+color both encoding weight), per-edge arrows for directed edges, then nodes on top with labels and an optional node-color legend."""
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
            # Through the resolver, like every other colormap lookup here: ``NetworkPanelSpec.colormap``
            # defaults to the HEATMAP_GENERIC sentinel, which is a token rather than a colormap name, so the
            # raw subscript raised KeyError('__mlframe_default_heatmap__') on any network panel that did not
            # name a colormap -- while the plotly twin silently fell back to Viridis with a warning.
            cmap = matplotlib.colormaps[resolve_heatmap_cmap(p.colormap)]
            lo, hi = p.edge_width_range
            if wmax > wmin:
                lws = lo + (weights - wmin) / (wmax - wmin) * (hi - lo)
            else:
                lws = np.full_like(weights, (lo + hi) / 2.0)
            lc = LineCollection(segments, linewidths=lws.tolist(), colors=cmap(norm(weights)), alpha=0.8, zorder=1)
            ax.add_collection(lc)

            # Arrows for directed edges, under the SAME ceiling the plotly twin applies. The old comment
            # leaned on "the friend-graph max_nodes guard keeps edge counts modest", which is a property of
            # one caller rather than of the renderer -- and the two backends then disagreed about what the
            # figure shows: past 500 directed edges plotly drew no arrowheads and matplotlib drew all of
            # them, from one spec. (``Axes.annotate`` appends in O(1), unlike plotly's, so this bounds an
            # unbounded constant rather than a complexity class.)
            directed = p.edge_directed
            if np.isscalar(directed):
                directed = np.full(e_src.shape, bool(directed))
            else:
                directed = np.asarray(directed, dtype=bool)
            if int(directed.sum()) > _NETWORK_MAX_ARROWS:
                logger.debug("network panel has %d directed edges; arrowheads skipped past the %d cap, matching the plotly renderer",
                             int(directed.sum()), _NETWORK_MAX_ARROWS)
                directed = np.zeros_like(directed)
            for a, b, d in zip(e_src, e_dst, directed):
                if d:
                    ax.annotate("", xy=tuple(nx_pos[b]), xytext=tuple(nx_pos[a]),
                                arrowprops=dict(arrowstyle="-|>", color="0.35",
                                                alpha=0.6, shrinkA=8, shrinkB=8),
                                zorder=2)

            # A colorbar over edge weights that are all the same encodes nothing, and matplotlib renders the
            # degenerate range with an offset exponent ("1e-9+1") that reads as a real scale. The spectral
            # embedding passes a constant weight vector, so it drew exactly that beside every figure.
            if wmax > wmin:
                sm = ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax)
                if p.colorbar_label:
                    cbar.set_label(p.colorbar_label)

        ax.scatter(nx_pos[:, 0], nx_pos[:, 1], s=np.asarray(p.node_size, dtype=float), c=list(p.node_color), edgecolors="black", linewidths=0.5, zorder=3)
        # Only the biggest nodes are named, and the names are truncated. Every node used to be labelled at
        # full length, centred on its own marker, so a friend graph rendered its names as a single illegible
        # mat over the middle of the panel. The unlabelled nodes keep their marker, their size and their
        # colour -- the graph still shows them, it just does not try to name all 200.
        # Capping the COUNT is not the same as keeping them readable: a spectral embedding whose nodes
        # collapse into one region printed its surviving names on top of each other as a black smudge. Of
        # the biggest nodes, keep the ones whose label boxes actually clear each other.
        _cand = network_label_indices(p.node_size)
        _texts = [truncate_bar_label(p.node_label[_i], _NETWORK_LABEL_MAXLEN) for _i in _cand]
        _x_lo, _x_hi = float(np.min(nx_pos[:, 0])), float(np.max(nx_pos[:, 0]))
        _y_lo, _y_hi = float(np.min(nx_pos[:, 1])), float(np.max(nx_pos[:, 1]))
        _sizes = np.asarray(p.node_size, dtype=float).ravel()
        # Explicit None checks rather than ``measured or REF``: a measured extent of exactly 0 is a
        # degenerate axes, not a failed measurement, and the two deserve different treatment for different
        # reasons -- the same trap the panel-title width budget documents a few hundred lines up.
        _panel_w = _measured_axis_in(ax, horizontal=False)
        _panel_h = _measured_axis_in(ax, horizontal=True)
        _keep_lbl = non_colliding_label_indices(
            nx_pos[_cand, 0], nx_pos[_cand, 1], _texts, fontsize=7,
            x_span=max(_x_hi - _x_lo, 1e-9), y_span=max(_y_hi - _y_lo, 1e-9),
            width_in=_TITLE_REF_WIDTH_IN if _panel_w is None else _panel_w,
            height_in=_TITLE_REF_WIDTH_IN if _panel_h is None else _panel_h,
            priority=[_sizes[_i] for _i in _cand] if _sizes.size == len(p.node_label) else None,
        )
        for _j in _keep_lbl:
            _x, _y = nx_pos[_cand[_j]]
            ax.annotate(_texts[_j], (_x, _y), fontsize=7, ha="center", va="center", zorder=4)

        if p.node_legend:
            handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=col, markersize=8, label=lbl) for lbl, col in p.node_legend]
            _place_legend(ax, handles, [h.get_label() for h in handles])

        _set_panel_title(ax, p.title)
        ax.set_xlabel(p.xlabel)
        ax.set_ylabel(p.ylabel)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.margins(0.12)


__all__ = ["MatplotlibRenderer"]


# ``_scatter`` lives in a sibling module (this file crossed the 1000-LOC house limit once the low-evidence
# split and the per-label contrast landed); bound back onto the class here so the ``_render_panel`` dispatch
# and any external ``MatplotlibRenderer._scatter`` reference keep resolving unchanged.
from ._matplotlib_scatter import _scatter as _scatter_impl

MatplotlibRenderer._scatter = _scatter_impl
