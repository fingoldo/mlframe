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
import os
from typing import Any, ClassVar, List, Optional

from mlframe._output_paths import ensure_parent_dir
import numpy as np

from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, ConfusionMarginsPanelSpec, FigureSpec,
    HeatmapPanelSpec, HistogramPanelSpec, LinePanelSpec, NetworkPanelSpec,
    ScatterPanelSpec, ViolinPanelSpec,
)

# Kaleido lifecycle + static-image write plumbing lives in the sibling module; re-exported here so
# ``from mlframe.reporting.renderers.plotly import get_kaleido_oneshot_stats`` (and the recovery-test
# imports of ``_restart_kaleido_server`` etc.) keep resolving from the same place.
from ._kaleido import (
    _ensure_kaleido_server_started,  # noqa: F401 -- re-exported, pinned by test_inv57_public_kaleido_surface_reexported_from_plotly
    _is_kaleido_persistent_burned,  # noqa: F401 -- re-exported, pinned by test_inv57_public_kaleido_surface_reexported_from_plotly
    _mark_kaleido_persistent_burned,  # noqa: F401 -- re-exported, pinned by test_inv57_public_kaleido_surface_reexported_from_plotly
    _record_kaleido_persistent_failure,  # noqa: F401 -- re-exported, pinned by test_inv57_public_kaleido_surface_reexported_from_plotly
    _restart_kaleido_server,  # noqa: F401 -- re-exported for test_kaleido_recovery.py
    get_kaleido_oneshot_stats,  # noqa: F401 -- re-exported for _phase_finalize.py / test_plotly_kaleido_module_split_inv57.py
    record_kaleido_oneshot_call,  # noqa: F401 -- re-exported, pinned by test_inv57_public_kaleido_surface_reexported_from_plotly
    reset_kaleido_oneshot_stats,  # noqa: F401 -- re-exported for _phase_finalize.py
    write_image_via_kaleido,
)
from ._plotly_interactivity import apply_interactivity, html_config
from ._plotly_color import _rgba, _mpl_to_plotly_cmap
from ._shared_helpers import (  # noqa: F401 -- _HEATMAP_MAX_TICKS re-exported for callers importing the tick-thinning constant from this module
    _HEATMAP_CELL_TEXT_MAX, _HEATMAP_MAX_TICKS, _HIST_PREBIN_THRESHOLD, _SCATTER_MAX_POINTS,
    _finite_range, _per_series_flags, _thin_tick_positions, epoch_ns_ticks,
    histogram_bar_extent, low_evidence_mask, panel_title_wrap_chars, select_per_point, truncate_bar_label, wrap_annotation_text,
    wrap_text_to_width, wrap_title_lines,
)

from mlframe.reporting.colors import NORMAL_OVERLAY
logger = logging.getLogger(__name__)

# plotly is an optional, heavy dependency: keep the import lazy (deferred off module load) but declare it
# once here and cache the module so the ~8 render methods reuse a single import instead of re-importing.
_GO_MODULE = None


def _go():
    """Lazily import and cache ``plotly.graph_objects``; reused across all render methods."""
    global _GO_MODULE
    if _GO_MODULE is None:
        import plotly.graph_objects as go
        _GO_MODULE = go
    return _GO_MODULE


# Text-wrap budgets mirror the matplotlib renderer (~90 chars/line for the full-width suptitle, ~46 for one panel); plotly annotations need ``<br>`` (not ``\n``). Wrappers live inline because strict file-ownership scopes this fix to plotly.py.
_SUPTITLE_WRAP_CHARS = 90
# Caption point size, shared with the width measurement that wraps it (matches the matplotlib twin).
_CAPTION_FONTSIZE = 10
# Subplot-title font. plotly's own default (16) overflows horizontally into the adjacent subplot at a
# typical 3-column figsize; 11 matches matplotlib's panel titles.
_PANEL_TITLE_FONTSIZE = 11
# matplotlib's default figure dpi; ``FigureSpec.figsize`` is in matplotlib inches, so both backends must
# use the same px-per-inch or the same spec yields two differently-sized figures.
_PX_PER_INCH = 100
# Past this many bar categories thin x-tick labels to ~20 evenly-spaced (matches matplotlib); truncate labels over _BAR_XTICK_MAXLEN chars so long feature names don't crowd.
_BAR_XTICK_THIN_THRESHOLD = 25
_BAR_XTICK_KEEP = 20
# 60, not 24: the matplotlib renderer truncates nothing at all and stays readable at the same figsize
# because both backends already rotate these labels -- so a 24-char cap only made the plotly twin LESS
# informative than its matplotlib counterpart, turning e.g. "job_posted_at_day_of_year_cos" into
# "job_posted_at_day_of_ye...". The cap is kept purely as a safety valve against a pathological name
# (a 200-char generated column) blowing out the bottom margin; ordinary feature names now render in full.
_BAR_XTICK_MAXLEN = 60


def _wrap_text(text: str, width: int, *, sep: str = "<br>") -> str:
    """Wrap ``text`` to ``width`` chars/line (each ``\n``-delimited line independently, preserving explicit breaks), folded with ``sep``."""
    return sep.join(wrap_title_lines(text, width))


def _wrap_text_to_figure(text: str, *, fontsize: float, width_in: float, fallback_chars: int, sep: str = "<br>") -> str:
    """Wrap to the figure's real width, measuring the font, and fold with plotly's ``<br>``.

    Same reasoning as the matplotlib twin: a fixed chars-per-line budget is a claim about one width and one
    font size, so at any other width the headline is broken early and the rest of the page goes unused.
    """
    return sep.join(wrap_text_to_width(text, fontsize=fontsize, width_in=width_in, fallback_chars=fallback_chars))


# matplotlib marker token -> plotly symbol name. Anything outside this table is a marker the caller chose
# deliberately, so it is worth a warning rather than a silent substitution -- see ``_marker_symbol``.
_MARKER_MAP: dict[str, str] = {
    "*": "star", "D": "diamond", "d": "diamond-tall", "o": "circle", "s": "square", "^": "triangle-up",
    "v": "triangle-down", "<": "triangle-left", ">": "triangle-right", "p": "pentagon", "h": "hexagon",
    "H": "hexagon2", "x": "x-thin", "X": "x", "+": "cross-thin", "P": "cross", ".": "circle", ",": "circle",
    "8": "octagon", "|": "line-ns", "_": "line-ew",
}
_MARKER_WARNED: set = set()


def _plotlyjs_mode():
    """How plotly.js reaches the saved HTML: ``"cdn"`` (default) or ``True`` to inline the ~3-4 MB bundle.

    A CDN reference produces blank panels on an air-gapped host and says nothing about why, so the offline choice
    has to be reachable without editing the renderer: set ``MLFRAME_PLOTLY_JS=inline``.
    """
    mode = (os.environ.get("MLFRAME_PLOTLY_JS") or "").strip().lower()
    if mode in ("inline", "embed", "true", "1", "yes", "on"):
        return True
    if mode == "directory":
        return "directory"  # one shared plotly.min.js beside the reports; smallest total for a whole run
    return "cdn"


def _marker_symbol(msym: str) -> str:
    """Map a matplotlib marker token to a plotly symbol, warning ONCE per unmapped token.

    The previous ``_MARKER_MAP.get(msym, "star")`` turned every unmapped marker into a star, so a builder's
    deliberate choice (say ``"v"`` for a downward-pointing threshold marker) silently became a different
    glyph on plotly than on matplotlib, with nothing anywhere saying so. Falling back is still the right
    behaviour -- a chart should not fail over a marker -- but it should be audible.
    """
    sym = _MARKER_MAP.get(msym)
    if sym is not None:
        return sym
    if msym not in _MARKER_WARNED:
        _MARKER_WARNED.add(msym)
        logger.warning(
            "[plotly-render] marker %r has no plotly equivalent in _MARKER_MAP; drawing it as a star, so this "
            "point will not match its matplotlib twin. Add the mapping to silence this. Fires once per token.",
            msym,
        )
    return "star"


# ``_truncate_label`` is ``truncate_bar_label`` from ._shared_helpers -- one definition, both backends.
_truncate_label = truncate_bar_label

# Renderer-level safety nets for specs carrying raw large-n data. Builders are expected to
# pre-sample / pre-bin, but the renderer is public API: above these thresholds a raw spec would
# embed n values into the HTML (37 MB / 73 MB per panel at 2M, browser-freezing).
# WebGL traces render large scatters orders of magnitude faster than SVG-mode go.Scatter.
_SCATTER_WEBGL_THRESHOLD = 10_000
_SCATTER_DOWNSAMPLE_WARNED = False
# Above this many heatmap cells, per-cell text is unreadable soup AND the plotly add_annotation loop (one layout
# copy per cell) stalls; skip the text past it (matches the matplotlib renderer cap).


def _warn_scatter_downsample(n: int) -> None:
    """Log once per process that a scatter panel with ``n`` raw points got downsampled to ``_SCATTER_MAX_POINTS`` to keep the HTML output responsive."""
    global _SCATTER_DOWNSAMPLE_WARNED
    if not _SCATTER_DOWNSAMPLE_WARNED:
        logger.warning(
            "[plotly-render] scatter panel carries %d raw points; downsampled to %d "
            "(extremes preserved) to keep the figure responsive. Pre-sample at the spec "
            "builder to silence this. Fires once per process.",
            n, _SCATTER_MAX_POINTS,
        )
        _SCATTER_DOWNSAMPLE_WARNED = True


def _line_uses_secondary_y(p) -> bool:
    """True if any series in a ``LinePanelSpec`` requests the secondary y-axis; drives whether the subplot cell must be created with ``secondary_y=True``."""
    n = len(p.y) if isinstance(p.y, tuple) else 1
    return any(_per_series_flags(p.secondary_y, n))


def _single_panel_has_labelled_series(spec: FigureSpec) -> bool:
    """True for a one-panel figure whose sole panel names its series.

    Gates the interactive-HTML legend: with one panel there is no cross-panel legend soup to avoid, and
    explicit ``series_labels`` are the author saying these lines need telling apart.
    """
    panels = [p for row in spec.panels for p in row if p is not None]
    if len(panels) != 1:
        return False
    labels = getattr(panels[0], "series_labels", None)
    if not labels:
        return False
    return any(bool(lab) for lab in labels)


def _err_to_plotly(err):
    """Spec error-bar field -> plotly ``error_y`` / ``error_x`` dict (data mode, asymmetric where a pair is given)."""
    if err is None:
        return None
    if isinstance(err, tuple):
        return dict(type="data", symmetric=False, array=np.asarray(err[1], dtype=float), arrayminus=np.asarray(err[0], dtype=float), visible=True)
    return dict(type="data", symmetric=True, array=np.asarray(err, dtype=float), visible=True)


class PlotlyRenderer:
    """Renders a ``FigureSpec`` to a plotly ``go.Figure`` and handles save/show for the interactive HTML backend.

    Panel-type dispatch mirrors the matplotlib renderer one-for-one (scatter/histogram/heatmap/confusion-margins/
    bar/line/violin/network/annotation), so both backends produce visually equivalent output from the same spec.
    """

    backend = "plotly"

    # Bound at this module's bottom from ``._plotly_network`` (that panel was carved out to keep this file
    # under the 1000-LOC house limit). Declared here so the dynamic assignment is visible to type-checkers
    # and to anyone reading the class surface.
    _NETWORK_MAX_ARROWS: ClassVar[int]
    _network: ClassVar[Any]
    # Bound at the bottom of this module from ``._plotly_heatmap`` (carved out for the LOC limit);
    # declared here so the dispatch below type-checks against a real attribute.
    _heatmap: ClassVar[Any]
    _confusion_margins: ClassVar[Any]
    _colorbar_placement: ClassVar[Any]
    # Bound at the bottom of this module from ``._plotly_scatter``, same as the two families above.
    _scatter: ClassVar[Any]

    def render(self, spec: FigureSpec, *, static_legend: bool = False) -> Any:
        """Build a plotly figure from the spec.

        ``static_legend`` enables a figure-level legend. The interactive HTML output identifies series via
        hover tooltips, so legends stay off there; a STATIC export (png/svg/pdf) has no hover, so when the
        save-format set includes one the caller passes ``static_legend=True`` to make the export readable.
        """
        _go()  # prime the cached plotly.graph_objects module for the render helpers below
        from plotly.subplots import make_subplots

        rows = len(spec.panels)
        cols = max((len(r) for r in spec.panels), default=0)
        if rows == 0 or cols == 0:
            raise ValueError("FigureSpec has no panels")

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

        # Subplot titles are HTML annotations: wrap long panel titles (~46 chars/line, matching matplotlib) so they fold instead of bleeding into the adjacent subplot, and convert ``\n`` -> ``<br>`` (plotly drops a raw newline).
        # Measured against the font rather than counted in characters, for the reason the matplotlib twin
        # documents: a diagnostic title is digits and CamelCase identifiers, not average-width characters.
        _panel_wrap = panel_title_wrap_chars(spec.figsize, cols)
        _panel_w_in = float(spec.figsize[0]) / max(cols, 1)
        subplot_titles = []
        for row in spec.panels:
            for c in range(cols):
                if c >= len(row) or row[c] is None:
                    subplot_titles.append("")
                else:
                    subplot_titles.append(
                        _wrap_text_to_figure(
                            getattr(row[c], "title", "") or "",
                            fontsize=_PANEL_TITLE_FONTSIZE, width_in=_panel_w_in, fallback_chars=_panel_wrap,
                        )
                    )

        # Row 1's titles live in the top MARGIN, but every later row's is stamped into the inter-row gap -- which
        # was sized from the row count alone, so a tall title below row 1 had nothing reserved for it. Grow the gap
        # with the tallest title in ANY row.
        _max_title_lines = max((t.count("<br>") + 1) for t in subplot_titles if t) if any(subplot_titles) else 1

        subplots_kwargs = dict(
            rows=rows, cols=cols,
            specs=sub_specs,
            subplot_titles=subplot_titles,
            shared_xaxes=spec.sharex,
            shared_yaxes=spec.sharey,
            horizontal_spacing=0.08,
            # Roomier vertical gap so a row's subplot-title annotation (stamped just above the subplot domain) clears the data/xticks of the row above and wrapped multi-line titles don't overlap the row beneath; capped at plotly's 1/(rows-1) ceiling.
            vertical_spacing=(min(0.16 + 0.03 * max(_max_title_lines - 1, 0), 0.9 / max(rows - 1, 1)) if rows > 1 else 0.16),
        )
        if spec.row_height_ratios is not None:
            total = sum(spec.row_height_ratios)
            subplots_kwargs["row_heights"] = [r / total for r in spec.row_height_ratios]
        if spec.col_width_ratios is not None:
            total = sum(spec.col_width_ratios)
            subplots_kwargs["column_widths"] = [c / total for c in spec.col_width_ratios]

        fig = make_subplots(**subplots_kwargs)

        for ann in fig.layout.annotations:
            ann.font = dict(size=_PANEL_TITLE_FONTSIZE)

        for r, row in enumerate(spec.panels, start=1):
            for c in range(1, cols + 1):
                if c - 1 >= len(row) or row[c - 1] is None:
                    continue
                self._render_panel(fig, row[c - 1], r, c)

        # Figure-level layout. Reserve vertical headroom for the suptitle so it never lands on the first row of subplot titles: wrap it (~90 chars/line, matching matplotlib) then grow the top margin by the wrapped line count.
        n_suptitle_lines = 1
        if spec.suptitle:
            wrapped_suptitle = _wrap_text_to_figure(spec.suptitle, fontsize=spec.suptitle_fontsize, width_in=spec.figsize[0], fallback_chars=_SUPTITLE_WRAP_CHARS)
            n_suptitle_lines = wrapped_suptitle.count("<br>") + 1
            fig.update_layout(title=dict(
                text=wrapped_suptitle,
                font=dict(size=spec.suptitle_fontsize),
                x=0.5, xanchor="center", yanchor="top",
            ))

        # The top band has to hold BOTH the suptitle and the first row's subplot titles: plotly stamps a
        # subplot title as an annotation just ABOVE its subplot domain, i.e. inside this margin. Sizing the
        # band from the suptitle alone made a multi-line panel title land on top of the suptitle -- the
        # overlap seen on every wide multi-panel diagnostic figure. Reserve for the tallest row-1 title too.
        _row1_title_lines = max((t.count("<br>") + 1) for t in subplot_titles[:cols] if t) if any(subplot_titles[:cols]) else 0
        _panel_title_band = _row1_title_lines * (_PANEL_TITLE_FONTSIZE + 4)
        top_margin = (40 + n_suptitle_lines * (spec.suptitle_fontsize + 8) if spec.suptitle else 30) + _panel_title_band

        # How-to-read footnote pinned to the bottom edge (paper coords), small + dim. Grows the bottom margin so it
        # never overlaps the axes or the below-figure legend.
        n_caption_lines = 0
        if spec.caption:
            wrapped_caption = _wrap_text_to_figure(spec.caption, fontsize=_CAPTION_FONTSIZE, width_in=spec.figsize[0], fallback_chars=_SUPTITLE_WRAP_CHARS)
            n_caption_lines = wrapped_caption.count("<br>") + 1
            fig.add_annotation(
                text=wrapped_caption, xref="paper", yref="paper", x=0.5, y=0, xanchor="center", yanchor="top",
                yshift=-((90 if static_legend else 30) + 8), showarrow=False, font=dict(size=9, color="#595959"),
            )
        bottom_margin = (90 if static_legend else 50) + n_caption_lines * 16

        fig.update_layout(
            # ``figsize`` is in matplotlib inches and matplotlib renders at 100 dpi by default, so 80 px/in
            # rendered every plotly figure 20% smaller than its matplotlib twin built from the SAME spec --
            # the "plotly version looks cramped" difference. Match the backends at 100 px/in.
            width=int(spec.figsize[0] * _PX_PER_INCH),
            # BOTH margins are added on top of the requested figure height. Adding only the top one made the
            # plot AREA come out short of figsize -- measured 550px interactive / 510px static against
            # matplotlib's 600px for the same spec -- which is the same "plotly twin looks cramped" class of
            # bug as the px/in mismatch above, just from the other direction.
            height=int(spec.figsize[1] * _PX_PER_INCH) + top_margin + bottom_margin,
            # Bottom margin grows when the legend is shown so the below-figure legend has room.
            margin=dict(l=60, r=40, t=top_margin, b=bottom_margin),
            # Interactive HTML identifies series via hover, so the legend stays off on MULTI-panel figures to
            # avoid the legend soup (every panel's series pooled into one list: precision/recall/F1 mixed with
            # reliability lines). That reasoning does not hold for a SINGLE labelled panel -- there is no soup,
            # and without a legend a chart like the decision curve renders three unlabelled lines that a reader
            # cannot tell apart at a glance. A static export has no hover at all, so it always gets the legend.
            showlegend=static_legend or _single_panel_has_labelled_series(spec),
        )
        if static_legend:
            # Park the legend BELOW the plot area (horizontal, centred) so it never overlaps subplot titles / the suptitle the way a default top-right in-plot legend does on multi-panel figures.
            fig.update_layout(legend=dict(
                font=dict(size=9), itemsizing="constant",
                bgcolor="rgba(255,255,255,0.6)",
                orientation="h", yanchor="top", y=-0.08, xanchor="center", x=0.5,
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
        apply_interactivity(fig, spec, static_legend=static_legend)
        return fig

    def save(self, fig: Any, path: str, fmt: str) -> None:
        """Write ``fig`` to ``path`` in ``fmt`` (case-insensitive): ``html`` via ``write_html``, ``json`` via ``to_json``, ``png/svg/pdf`` via kaleido (falls back to html with a WARN if kaleido is missing). Raises ``ValueError`` on an unsupported format."""
        fmt = fmt.lower()
        if fmt == "html":
            # include_plotlyjs="cdn" (the default) references plotly.js instead of inlining ~3-4 MB into every report -- a
            # deliberate file-size tradeoff that renders as BLANK PANELS, with no error shown to the viewer, on a
            # host with no outbound internet (air-gapped training box, audited enterprise network). See
            # ``_plotlyjs_mode`` for the escape hatch.
            fig.write_html(ensure_parent_dir(path), include_plotlyjs=_plotlyjs_mode(), auto_open=False, config=html_config())
        elif fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                f.write(fig.to_json())
        elif fmt in ("png", "svg", "pdf"):
            write_image_via_kaleido(fig, path, fmt)
        else:
            raise ValueError(f"plotly doesn't support format {fmt!r}; " "supported: html/png/svg/pdf/json")

    def show(self, fig: Any) -> None:
        """Open ``fig`` in the default renderer (browser/notebook); any display-backend failure is swallowed and logged at debug level rather than raised."""
        try:
            fig.show()
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed: %s", e)
            pass

    # ------------------------------------------------------------------
    # Per-panel dispatch
    # ------------------------------------------------------------------

    def _render_panel(self, fig, panel, row: int, col: int) -> None:
        """Dispatch a single panel spec to its type-specific ``_<kind>`` renderer at subplot cell ``(row, col)``; raises ``TypeError`` for an unrecognised panel spec class."""
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
        """Render a text-only panel (no axes): centers ``p.text`` in the subplot cell and hides both axes so the cell reads as a plain note/caption."""
        # plotly does not wrap free text at all, and paints annotations above traces, so an unwrapped line lands
        # visually on top of the neighbouring panel. Wrap to this subplot's own width before handing it over.
        _grid = getattr(fig, "_grid_ref", None)
        _cols = max(1, len(_grid[0])) if _grid else 1
        _panel_w_in = max(float(fig.layout.width or (_PX_PER_INCH * 8)) / _PX_PER_INCH / _cols, 1.0)
        _text = wrap_annotation_text(p.text, _panel_w_in, p.fontsize)
        fig.add_annotation(text=_text.replace("\n", "<br>"), x=0.5, y=0.5,
                           xref="x domain", yref="y domain", showarrow=False,
                           font=dict(size=p.fontsize, family="monospace" if getattr(p, "monospace", False) else None),
                           align="left" if getattr(p, "monospace", False) else "center",
                           row=row, col=col)
        fig.update_xaxes(visible=False, row=row, col=col)
        fig.update_yaxes(visible=False, row=row, col=col)

    def _histogram(self, fig, p: HistogramPanelSpec, row: int, col: int) -> None:
        """Render a histogram panel: uses spec-supplied ``bin_centers`` directly, pre-bins raw values above ``_HIST_PREBIN_THRESHOLD`` (avoids embedding huge raw arrays into HTML), else falls back to plotly's own ``go.Histogram`` binning; optionally overlays a fitted Normal PDF curve spanning the bin range."""
        go = _go()

        # ``overlay_x_lo/hi`` anchors the Normal-overlay grid. When we pre-bin (here or upstream) they come from
        # the bin EDGES, avoiding two extra full-n min/max passes over raw values (PERF-18).
        overlay_x_lo = overlay_x_hi = None
        bin_centers = p.bin_centers
        heights = None
        if bin_centers is None and len(np.asarray(p.values)) > _HIST_PREBIN_THRESHOLD:
            # Raw spec with n above the embed-hazard ceiling: bin once with numpy instead of shipping n values
            # into the HTML (37 MB / browser-freezing at 2M).
            from mlframe.reporting.charts import prebin_histogram
            heights, centers, width0 = prebin_histogram(np.asarray(p.values), p.bins, p.density)
            if heights is not None:
                bin_centers = centers

        if bin_centers is not None:
            # A per-bar width array and a single float are both valid here (plotly's ``Bar.width`` accepts either),
            # so the binding is deliberately widened rather than coerced.
            width_any: Any
            if heights is None:
                heights = np.asarray(p.values)
                if isinstance(p.bin_width, np.ndarray):
                    width_any = np.asarray(p.bin_width, dtype=float)
                else:
                    width_any = float(p.bin_width if p.bin_width is not None else ((bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 1.0))
            else:
                width_any = float(width0)
            width = width_any
            colors_kw: dict[str, Any] = dict(color=p.color)
            if p.bar_colors is not None:
                # The range and its degeneracy guard were computed and then discarded -- `colors_kw` was
                # reassigned without them -- so plotly auto-scaled per trace while matplotlib's twin pinned
                # cmin/cmax. On a constant bar_colors vector plotly's autoscale is undefined; the guard exists
                # precisely for that, so pass both through as matplotlib already does.
                _h_min = float(np.min(p.bar_colors))
                _h_max = float(np.max(p.bar_colors))
                if _h_max <= _h_min:
                    _h_max = _h_min + 1.0
                colors_kw = dict(
                    color=np.asarray(p.bar_colors),
                    colorscale=_mpl_to_plotly_cmap(p.colormap),
                    cmin=_h_min,
                    cmax=_h_max,
                )
            fig.add_trace(
                go.Bar(x=np.asarray(bin_centers), y=np.asarray(heights),
                       width=width,
                       marker=dict(line=dict(color="white", width=0.5), **colors_kw),
                       showlegend=False),
                row=row, col=col,
            )
            if len(bin_centers) > 0:
                overlay_x_lo, overlay_x_hi = histogram_bar_extent(bin_centers, width)
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
                assert overlay_x_hi is not None
                x_grid = np.linspace(overlay_x_lo, overlay_x_hi, 200)
                normal_pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2)
                label = p.overlay_label if p.overlay_label is not None else f"Normal(mu={mu:.2g}, sigma={sigma:.2g})"
                fig.add_trace(
                    go.Scatter(x=x_grid, y=normal_pdf, mode="lines", line=dict(color=NORMAL_OVERLAY, dash="dash", width=1.4), name=label, showlegend=True),
                    row=row,
                    col=col,
                )

        if p.xlim is not None:
            fig.update_xaxes(range=list(p.xlim), row=row, col=col)
        fig.update_xaxes(title_text=p.xlabel, row=row, col=col, showgrid=p.grid)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid, type="log" if p.yscale == "log" else "linear")

    # ``_confusion_margins`` / ``_colorbar_placement`` / ``_heatmap`` live in ``._plotly_heatmap`` and are
    # bound onto this class at the bottom of the module. Carved out to keep this file under the house
    # 1000-LOC limit; they are the largest self-contained group here and share no state with the rest.

    def _bar(self, fig, p: BarPanelSpec, row: int, col: int) -> None:
        """Render a bar panel (grouped when ``p.values`` is a tuple of series), with an optional reference line perpendicular to the bars and long category-label truncation/thinning/rotation on the value-orthogonal axis."""
        go = _go()

        from mlframe.reporting.colors import line_color
        horizontal = p.orientation == "horizontal"
        cats = list(p.categories)

        # matplotlib hatch tokens -> plotly pattern shapes, so a hatch set for colour-vision redundancy survives
        # the backend switch instead of silently becoming a plain fill.
        _HATCH_TO_PATTERN = {"/": "/", "//": "/", "\\": "\\", "\\\\": "\\", "x": "x", "xx": "x", "-": "-", "|": "|", "+": "+", ".": ".", "..": "."}

        def _add_bar(values, color, label, show, hatch="", err=None):
            """One bar trace; ``err`` is (lower, upper) distances along the value axis."""
            """Add one ``go.Bar`` trace for ``values`` with the given ``color``/legend ``label``/``show`` (showlegend) flag, oriented per the enclosing panel's horizontal/vertical setting."""
            if horizontal:
                # Categories on y, values on x; reverse so the first category sits on top (worst-first reads down).
                fig.add_trace(
                    go.Bar(y=cats, x=np.asarray(values), orientation="h", name=label, showlegend=show,
                           error_x=(dict(type="data", symmetric=False, array=np.asarray(err[1]),
                                         arrayminus=np.asarray(err[0])) if err is not None else None),
                           marker=dict(color=color, pattern=dict(shape=_HATCH_TO_PATTERN.get(hatch, "")))),
                    row=row,
                    col=col,
                )
            else:
                fig.add_trace(
                    go.Bar(x=cats, y=np.asarray(values), name=label, showlegend=show,
                           error_y=(dict(type="data", symmetric=False, array=np.asarray(err[1]),
                                         arrayminus=np.asarray(err[0])) if err is not None else None),
                           marker=dict(color=color, pattern=dict(shape=_HATCH_TO_PATTERN.get(hatch, "")))),
                    row=row,
                    col=col,
                )

        if isinstance(p.values, tuple):
            for i, series in enumerate(p.values):
                lbl = p.series_labels[i] if p.series_labels else f"series {i}"
                # plotly's default qualitative palette clashes with matplotlib's tab10 in the same figure; fall
                # back to ``line_color(i)`` (tab10) when the spec doesn't pin colors for cross-backend parity.
                color = p.colors[i] if (p.colors is not None and i < len(p.colors)) else line_color(i)
                _add_bar(series, color, lbl, True, p.hatches[i] if (p.hatches and i < len(p.hatches)) else "")
            # ``barmode`` is a FIGURE-level property, so setting it here from inside one panel silently
            # applies to every bar and histogram trace in the whole figure -- a sibling histogram panel that
            # wants "overlay" can never get it, and the setting arrives depending on panel ORDER. Only set it
            # when nothing else has, so the first grouped-bar panel establishes it and no panel overrides a
            # value another one already needs.
            if fig.layout.barmode is None:
                fig.update_layout(barmode="group")
        else:
            # A colours tuple as long as ``values`` is PER-BAR, not per-series: plotly's marker.color accepts an
            # array. Reading ``colors[0]`` painted every bar the colour of the first one.
            _bar_color = list(p.colors) if (p.colors and len(p.colors) == len(p.values) and len(p.colors) > 1) else (p.colors[0] if p.colors else "steelblue")
            _add_bar(p.values, _bar_color, "", False, p.hatches[0] if p.hatches else "", p.value_err)

        # Reference line perpendicular to the bars (global metric). vline for horizontal bars (value axis is x),
        # hline for vertical bars (value axis is y).
        if p.hline is not None:
            hval, hcolor, hlabel = p.hline
            line_kw = dict(line=dict(color=hcolor, dash="dash", width=1.3), annotation_text=hlabel or None, annotation_position="top right", row=row, col=col)
            if horizontal:
                fig.add_vline(x=hval, **line_kw)
            else:
                fig.add_hline(y=hval, **line_kw)

        if horizontal:
            if any(len(str(c)) > _BAR_XTICK_MAXLEN for c in cats):  # truncate long feature-name labels on the y-axis so they don't crowd the panel
                fig.update_yaxes(tickmode="array", tickvals=cats, ticktext=[_truncate_label(c) for c in cats], row=row, col=col)
            fig.update_yaxes(autorange="reversed", row=row, col=col)
            fig.update_xaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid)
            fig.update_yaxes(title_text=p.xlabel, row=row, col=col)
        else:
            n_cat = len(cats)
            # Rotate + truncate long category labels; thin to ~20 evenly-spaced past 25 categories (matching matplotlib) so they don't smear.
            tickangle = -p.xtick_rotation if p.xtick_rotation else 0
            needs_trunc = any(len(str(c)) > _BAR_XTICK_MAXLEN for c in cats)
            if n_cat > _BAR_XTICK_THIN_THRESHOLD:
                step = math.ceil(n_cat / _BAR_XTICK_KEEP)
                sel = list(range(0, n_cat, step))
                fig.update_xaxes(tickmode="array",
                                 tickvals=[cats[i] for i in sel],
                                 ticktext=[_truncate_label(cats[i]) for i in sel],
                                 tickangle=tickangle if p.xtick_rotation else -45,
                                 row=row, col=col, title_text=p.xlabel)
            elif needs_trunc:
                fig.update_xaxes(tickmode="array", tickvals=cats,
                                 ticktext=[_truncate_label(c) for c in cats],
                                 tickangle=tickangle if p.xtick_rotation else -30,
                                 row=row, col=col, title_text=p.xlabel)
            else:
                fig.update_xaxes(title_text=p.xlabel, row=row, col=col, tickangle=tickangle)
            fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid)

    def _line(self, fig, p: LinePanelSpec, row: int, col: int) -> None:
        """Render a multi-series line panel: per-series style/color/secondary-y/fill-to-baseline, an optional uncertainty band, vspans/vlines (datetime-safe), and point markers; secondary-y series get their own right-hand axis when any series requests it."""
        go = _go()
        from mlframe.reporting.colors import line_color

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

        for span in p.vspans or ():
            vx0, vx1, vcolor, valpha = span[0], span[1], span[2], span[3]
            vlabel = span[4] if len(span) > 4 else ""
            fig.add_vrect(x0=vx0, x1=vx1, fillcolor=_rgba(vcolor, valpha), line_width=0, layer="below", row=row, col=col)
            if vlabel:
                # No native per-vrect legend in plotly; the invisible scatter proxy carries the label INTO the
                # legend, and the annotation carries it onto the band itself -- which is the only one that survives
                # on a multi-panel interactive figure, where the legend is off (hover identifies the series).
                fig.add_trace(
                    go.Scatter(x=[None], y=[None], mode="markers",
                               marker=dict(size=8, color=_rgba(vcolor, max(valpha, 0.3)), symbol="square"),
                               name=vlabel, showlegend=True),
                    row=row, col=col,
                )
                fig.add_annotation(x=vx0, y=1.0, yref="y domain", yanchor="bottom", xanchor="left",
                                   text=vlabel, showarrow=False, font=dict(size=8, color=vcolor),
                                   row=row, col=col)
        for vx, vcolor, vlabel in p.vlines or ():
            # add_vline does arithmetic on x that raises on a datetime axis; a line-shape with the x in data coords
            # and y spanning the panel's y-domain works on numeric AND datetime axes alike.
            self._add_vline_datetime_safe(fig, vx, vcolor, vlabel, row, col)

        for mx, my, mlabel, mcolor, msym in p.point_markers or ():
            fig.add_trace(
                go.Scatter(x=[mx], y=[my], mode="markers+text",
                           marker=dict(color=mcolor, size=13, symbol=_marker_symbol(msym),
                                       line=dict(color="black", width=0.6)),
                           text=[mlabel or ""], textposition="bottom right", textfont=dict(size=8),
                           name=mlabel or None, showlegend=bool(mlabel)),
                row=row, col=col,
            )

        # ``x_is_time`` with a NUMERIC x means epoch nanoseconds; rotating the labels (all this used to do)
        # leaves them reading "1.62e18". ``epoch_ns_ticks`` no-ops on an already-datetime axis.
        # See the matplotlib twin: builders set ylim deliberately and no line-panel path honoured it.
        _ylim = getattr(p, "ylim", None)
        if _ylim is not None:
            fig.update_yaxes(range=[float(_ylim[0]), float(_ylim[1])], row=row, col=col, secondary_y=False)
        _xkw: dict = dict(title_text=p.xlabel, row=row, col=col, showgrid=p.grid, tickangle=-30 if p.x_is_time else 0)
        _tv, _tt = epoch_ns_ticks(_xi(0)) if p.x_is_time else (None, None)
        if _tv is not None:
            _xkw.update(tickmode="array", tickvals=_tv, ticktext=_tt)
        fig.update_xaxes(**_xkw)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid, secondary_y=False)
        if has_secondary:
            fig.update_yaxes(title_text=p.secondary_ylabel, row=row, col=col, secondary_y=True, showgrid=False)

    @staticmethod
    def _is_datetime_like(v) -> bool:
        """True if ``v`` is a ``numpy.datetime64`` or a stdlib ``datetime``/``date``; gates the datetime-safe vline path since ``fig.add_vline`` raises ``TypeError`` on datetime x."""
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
            x_coord = pd.Timestamp(vx)
            fig.add_shape(
                type="line", x0=x_coord, x1=x_coord, y0=0, y1=1, yref="y domain", xref="x", line=dict(color=vcolor, dash="dot", width=1.2), row=row, col=col
            )
            if vlabel:
                fig.add_annotation(x=x_coord, y=1, yref="y domain", yanchor="bottom", text=vlabel, showarrow=False, font=dict(size=9), row=row, col=col)
        else:
            fig.add_vline(x=vx, line=dict(color=vcolor, dash="dot", width=1.2), annotation_text=vlabel or None, annotation_position="top", row=row, col=col)

    def _violin(self, fig, p: ViolinPanelSpec, row: int, col: int) -> None:
        """Render one ``go.Violin`` trace per group in ``p.groups`` (tab10 color cycle for cross-backend parity with matplotlib), with an optional inner box overlay."""
        go = _go()
        from mlframe.reporting.colors import line_color

        for i, group in enumerate(p.groups):
            # tab10 cycle for cross-backend parity (plotly default
            # 'Plotly' qualitative is over-saturated next to mpl bars).
            color = line_color(i)
            fig.add_trace(
                go.Violin(y=np.asarray(group),
                          name=p.group_labels[i],
                          box_visible=p.show_box,
                          meanline_visible=False,
                          line_color=color,
                          fillcolor=color,
                          opacity=0.6,
                          showlegend=False),
                row=row, col=col,
            )
        fig.update_xaxes(title_text=p.xlabel, row=row, col=col, tickangle=-30)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid)


__all__ = ["PlotlyRenderer"]


# ``_network`` lives in a sibling module (this file was over the 1000-LOC house limit); bound back onto the
# class here so ``PlotlyRenderer._network`` and the ``_render_panel`` dispatch keep resolving unchanged.
from ._plotly_network import _NETWORK_MAX_ARROWS, _network as _network_impl

PlotlyRenderer._NETWORK_MAX_ARROWS = _NETWORK_MAX_ARROWS
PlotlyRenderer._network = _network_impl

# Same pattern for the heatmap family (``_heatmap`` / ``_confusion_margins`` / ``_colorbar_placement``),
# carved out for the same LOC reason. ``_colorbar_placement`` is a staticmethod on the class, so it is
# wrapped back into one -- binding the bare function would silently pass ``self`` as ``fig``.
from ._plotly_heatmap import (
    _colorbar_placement as _colorbar_placement_impl,
    _confusion_margins as _confusion_margins_impl,
    _heatmap as _heatmap_impl,
)

PlotlyRenderer._heatmap = _heatmap_impl
PlotlyRenderer._confusion_margins = _confusion_margins_impl
PlotlyRenderer._colorbar_placement = staticmethod(_colorbar_placement_impl)

# ``_scatter`` too, for the same reason: the low-evidence split (a second, hollow trace for bins whose interval
# is too wide to read) pushed this file back over the limit.
from ._plotly_scatter import _scatter as _scatter_impl

PlotlyRenderer._scatter = _scatter_impl
