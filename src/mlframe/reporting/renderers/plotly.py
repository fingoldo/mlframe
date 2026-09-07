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
    _HEATMAP_CELL_TEXT_MAX, _HEATMAP_MAX_TICKS, _HIST_PREBIN_THRESHOLD, _SCATTER_MAX_POINTS, PX_PER_INCH,
    CAPTION_FONTSIZE, CAPTION_WRAP_CHARS, PANEL_TITLE_FONTSIZE, SUPTITLE_WRAP_CHARS,
    _finite_range, _per_series_flags, _thin_tick_positions, epoch_ns_ticks, label_width_pitch_in, plotly_axis_suffix, rotated_tick_pitch_in, stagger_label_rows, ticks_that_fit,
    histogram_bar_extent, low_evidence_mask, panel_title_wrap_chars, select_per_point, truncate_bar_label, wrap_annotation_text,
    wrap_text_to_width, wrap_title_lines,
)

from mlframe.reporting.colors import BAR_PRIMARY, NORMAL_OVERLAY
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
_SUPTITLE_WRAP_CHARS = SUPTITLE_WRAP_CHARS
# Caption point size, shared with the width measurement that wraps it AND with the matplotlib twin.
_CAPTION_FONTSIZE = CAPTION_FONTSIZE
# Subplot-title font. plotly's own default (16) overflows horizontally into the adjacent subplot at a
# typical 3-column figsize.
_PANEL_TITLE_FONTSIZE = PANEL_TITLE_FONTSIZE
# matplotlib's default gridline (#b0b0b0) at the alpha=0.3 every panel draws it with, flattened against the
# white panel background so plotly's opaque gridlines read at the same weight.
_GRID_COLOR = "#e7e7e7"
# One definition in ._shared_helpers: the heatmap tick budget converts a plotly pixel extent back to inches
# and has to agree with whatever this renderer sized the figure at.
_PX_PER_INCH = PX_PER_INCH
# Past this many bar categories thin x-tick labels to ~20 evenly-spaced (matches matplotlib); truncate labels over _BAR_XTICK_MAXLEN chars so long feature names don't crowd.
# Labels stamped just above a panel (vspan regimes, vline change points) all share one y, so neighbours
# overprint. Staggering across a few rows separates them without arrows or a de-collision solver.
# How many rows a stack of marker labels may use. Three, not two: a threshold sweep routinely carries three
# operating points within a percent of each other, and two rows put the first and third back on top of one
# another.
_STACKED_LABEL_ROWS = 3
_STACKED_LABEL_SHIFT_PX = 11
_VIOLIN_LABEL_MAXLEN = 20  # matches the matplotlib twin; a 30-deg rotated label projects most of its length
# Past this many categories the vertical branch switches to explicit, rotated tick text; how many of
# them survive is then decided by ``_bar_tick_budget`` from the axis's real length.
_BAR_XTICK_THIN_THRESHOLD = 25
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


def pd_timestamp(value):
    """``value`` as a pandas Timestamp; the vline shape needs a real datetime, not the raw label."""
    import pandas as pd

    return pd.Timestamp(value)


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


def _any_panel_needs_a_legend(spec: FigureSpec) -> bool:
    """True when some panel carries its own legend keys, independent of series labels.

    A network panel names its node CLASSES rather than series, so it has no ``series_labels`` and the
    single-panel test above answers False for it. The panel used to compensate by calling
    ``update_layout(showlegend=True)`` from inside its own body -- but ``render`` sets ``layout.showlegend``
    AFTER every panel has run, so that was overwritten and the node-class legend silently vanished on the
    interactive HTML backend while matplotlib drew it. Same figure-level-property-set-from-one-panel trap the
    ``barmode`` comment further down already documents.
    """
    return any(getattr(p, "node_legend", None) for row in spec.panels for p in row if p is not None)


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

        # A colorbar is pinned just outside its own subplot's right edge, and its TICK LABELS stick out further
        # still -- straight into the next column's y-axis title. The gap has to hold the bar, its labels and the
        # neighbour's axis furniture, which the default 0.08 does not on a multi-column figure.
        _has_colorbar = any(isinstance(pn, HeatmapPanelSpec) for rw in spec.panels for pn in rw if pn is not None)
        _hspace = 0.08
        if _has_colorbar and cols > 1:
            from ._plotly_heatmap import _COLORBAR_GUTTER_PX, _NEIGHBOUR_AXIS_PX

            _hspace = max(_hspace, (_COLORBAR_GUTTER_PX + _NEIGHBOUR_AXIS_PX) / (spec.figsize[0] * _PX_PER_INCH))

        subplots_kwargs = dict(
            rows=rows, cols=cols,
            specs=sub_specs,
            subplot_titles=subplot_titles,
            shared_xaxes=spec.sharex,
            shared_yaxes=spec.sharey,
            horizontal_spacing=_hspace,
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
        # Panels are drawn before ``update_layout`` sets width/height below, so a panel that needs to know how
        # much room it has (the heatmap's tick budget) cannot read it off the figure yet. Stamp the requested
        # size on the figure now, in the same px-per-inch the final layout uses, so the two agree.
        fig.layout.width = int(spec.figsize[0] * _PX_PER_INCH)
        fig.layout.height = int(spec.figsize[1] * _PX_PER_INCH)

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
            wrapped_caption = _wrap_text_to_figure(spec.caption, fontsize=_CAPTION_FONTSIZE, width_in=spec.figsize[0], fallback_chars=CAPTION_WRAP_CHARS)
            n_caption_lines = wrapped_caption.count("<br>") + 1
            fig.add_annotation(
                text=wrapped_caption, xref="paper", yref="paper", x=0.5, y=0, xanchor="center", yanchor="top",
                yshift=-((90 if static_legend else 30) + 8), showarrow=False, font=dict(size=_CAPTION_FONTSIZE, color="#595959"),
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
            showlegend=static_legend or _single_panel_has_labelled_series(spec) or _any_panel_needs_a_legend(spec),
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
        # Plotly's cartesian gridlines are drawn at full strength; matplotlib's are alpha=0.3 over the same
        # data, so the two backends printed the same chart at two visual densities. Pin the weight here, once,
        # rather than on every axis call.
        fig.update_xaxes(gridcolor=_GRID_COLOR, gridwidth=1)
        fig.update_yaxes(gridcolor=_GRID_COLOR, gridwidth=1)
        # Heatmap ticks are budgeted against the axes' real extent, and the margins that decide that extent
        # are only final here -- while the panels were being drawn they were still plotly's defaults.
        from ._plotly_heatmap import apply_heatmap_tick_budget

        for _r, _row in enumerate(spec.panels, start=1):
            for _c, _panel in enumerate(_row, start=1):
                if isinstance(_panel, HeatmapPanelSpec):
                    apply_heatmap_tick_budget(fig, _panel, _r, _c)
                elif isinstance(_panel, BarPanelSpec):
                    self._bar_tick_budget(fig, _panel, _r, _c)
                elif isinstance(_panel, ViolinPanelSpec):
                    _kept = [(np.asarray(g, dtype=float), lab) for g, lab in zip(_panel.groups, _panel.group_labels)]
                    self._violin_tick_budget(fig, [(g[np.isfinite(g)], lab) for g, lab in _kept if g[np.isfinite(g)].size > 0], _r, _c)
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
            # Matplotlib's twin drops non-finite values before binning and says so when nothing survives;
            # go.Histogram silently renders an empty framed panel instead, which reads as "no data at all".
            raw_vals = np.asarray(p.values, dtype=float).ravel()
            raw_vals = raw_vals[np.isfinite(raw_vals)]
            if raw_vals.size:
                fig.add_trace(
                    go.Histogram(x=raw_vals,
                                 nbinsx=p.bins,
                                 histnorm="probability density" if p.density else "",
                                 marker=dict(color=p.color, line=dict(color="white", width=0.4)),
                                 opacity=0.6, showlegend=False),
                    row=row, col=col,
                )
            else:
                # A subplot cell holding no trace at all is never laid out, and an annotation anchored to its
                # (undrawn) axes silently lands on a neighbouring panel. One empty scatter forces the cell to
                # exist, which also gives the reader the same framed-but-empty axes matplotlib draws.
                fig.add_trace(go.Scatter(x=[], y=[], mode="markers", showlegend=False, hoverinfo="skip"), row=row, col=col)
                fig.add_annotation(text="no finite values", x=0.5, y=0.5, xref="x domain", yref="y domain",
                                   showarrow=False, font=dict(size=9), row=row, col=col)

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
            _bar_color = list(p.colors) if (p.colors and len(p.colors) == len(p.values) and len(p.colors) > 1) else (p.colors[0] if p.colors else BAR_PRIMARY)
            _add_bar(p.values, _bar_color, "", False, p.hatches[0] if p.hatches else "", p.value_err)

        # Reference line perpendicular to the bars (global metric). vline for horizontal bars (value axis is x),
        # hline for vertical bars (value axis is y).
        if p.hline is not None:
            hval, hcolor, hlabel = p.hline
            # Mirrors the matplotlib twin: a symmetric band draws both bounds and annotates only the first, so
            # the reader sees the threshold on the side their data actually falls on.
            for _i, _v in enumerate((hval, -hval) if p.hline_symmetric else (hval,)):
                # A vline's label belongs at the line it names. "top right" pinned it to the panel corner,
                # metres from the vertical reference on a horizontal bar chart and usually on top of the
                # longest bar; matplotlib puts it in a legend that loc="best" moves out of the way. The
                # label is placed by hand rather than through ``annotation_position`` because every "top"
                # variant that library offers sits in the same strip as the subplot title, which is the
                # collision this fix exists to avoid -- it has to hang INSIDE the plot area.
                _label_here = (hlabel or None) if _i == 0 else None
                line_kw = dict(line=dict(color=hcolor, dash="dash", width=1.3), row=row, col=col)
                if horizontal:
                    fig.add_vline(x=_v, **line_kw)
                    if _label_here:
                        fig.add_annotation(x=_v, y=1.0, yref="y domain", yanchor="top", xanchor="left", xshift=3, yshift=-3,
                                           text=_label_here, showarrow=False, font=dict(size=9, color=hcolor), row=row, col=col)
                else:
                    fig.add_hline(y=_v, annotation_text=_label_here, annotation_position="top right", annotation_font=dict(size=9, color=hcolor), **line_kw)

        if horizontal:
            # THINNED as well as truncated, matching the matplotlib twin and this renderer's own VERTICAL
            # branch -- only the horizontal one was left out. A 200-row feature-importance chart had a clean
            # 20-label axis in the PNG and an unreadable band of overlapping text in the HTML, from one spec.
            # The bars stay one per category; only the labels subsample.
            # Truncation here; the SUBSAMPLING runs after the layout is final (``_bar_tick_budget``), because
            # the axis length it has to fit into is not known while panels are being drawn.
            fig.update_yaxes(tickmode="array", tickvals=list(cats), ticktext=[_truncate_label(c, keep_tail=p.label_keep_tail) for c in cats], row=row, col=col)
            fig.update_yaxes(autorange="reversed", row=row, col=col)
            # ``xlabel`` names the VALUE and ``ylabel`` the CATEGORY, whatever the orientation -- that is what
            # every horizontal-bar builder in charts/ passes ("ECE (lower = better calibrated)" / "subgroup",
            # "quality (higher is better)" / "metric") and what the matplotlib twin draws. Swapping them here
            # put "subgroup", "model" and "metric" along the value axis of six chart types in the HTML report
            # while the PNG of the same spec read correctly.
            fig.update_xaxes(title_text=p.xlabel, row=row, col=col, showgrid=p.grid)
            fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=False)
        else:
            n_cat = len(cats)
            # Rotate + truncate long category labels; thin to ~20 evenly-spaced past 25 categories (matching matplotlib) so they don't smear.
            tickangle = -p.xtick_rotation if p.xtick_rotation else 0
            needs_trunc = any(len(str(c)) > _BAR_XTICK_MAXLEN for c in cats)
            if n_cat > _BAR_XTICK_THIN_THRESHOLD:
                fig.update_xaxes(tickmode="array",
                                 tickvals=list(cats),
                                 ticktext=[_truncate_label(c, keep_tail=p.label_keep_tail) for c in cats],
                                 tickangle=tickangle if p.xtick_rotation else -45,
                                 row=row, col=col, title_text=p.xlabel, showgrid=False)
            elif needs_trunc:
                fig.update_xaxes(tickmode="array", tickvals=cats,
                                 ticktext=[_truncate_label(c, keep_tail=p.label_keep_tail) for c in cats],
                                 tickangle=tickangle if p.xtick_rotation else -30,
                                 row=row, col=col, title_text=p.xlabel, showgrid=False)
            else:
                fig.update_xaxes(title_text=p.xlabel, row=row, col=col, tickangle=tickangle, showgrid=False)
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

    @classmethod
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

    @staticmethod
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
        _w_in = float(fig.layout.width or 600) / _PX_PER_INCH / max(_cols, 1)
        return stagger_label_rows(_num_xs, texts, fontsize=8, x_span=_x_span, width_in=_w_in, max_rows=_STACKED_LABEL_ROWS)

    def _violin(self, fig, p: ViolinPanelSpec, row: int, col: int) -> None:
        """Render one ``go.Violin`` trace per group in ``p.groups`` (tab10 color cycle for cross-backend parity with matplotlib), with an optional inner box overlay.

        Empty groups are dropped and NAMED in the title, and the inner box's whiskers span the 5th-95th
        percentiles -- both matching the matplotlib twin. Previously this iterated every group unfiltered, so an
        empty one rendered as a labelled category with nothing in it and no note, while matplotlib dropped it and
        said so; and the box drew plotly's default 1.5x IQR fences against matplotlib's 5/95. One spec made two
        different claims about the same data, which is what the matplotlib box was added to stop.
        """
        go = _go()
        from mlframe.reporting.colors import line_color

        kept = [(np.asarray(g, dtype=float), lab) for g, lab in zip(p.groups, p.group_labels)]
        kept = [(g[np.isfinite(g)], lab) for g, lab in kept]
        drawable = [(g, lab) for g, lab in kept if g.size > 0]
        empty = [str(lab) for g, lab in kept if g.size == 0]

        for i, (group, label) in enumerate(drawable):
            # tab10 cycle for cross-backend parity (plotly default
            # 'Plotly' qualitative is over-saturated next to mpl bars).
            color = line_color(i)
            fig.add_trace(
                go.Violin(
                    # The trace NAME is the category label on the x axis, so it is truncated here the same way
                    # and to the same length as the matplotlib twin -- 20 rotated 30-char class names overlap
                    # into a staircase on either backend. The full name stays on the hover.
                    y=group, name=_truncate_label(label, _VIOLIN_LABEL_MAXLEN, keep_tail=6), box_visible=False,
                    meanline_visible=False, line_color=color, fillcolor=color, opacity=0.6, showlegend=False,
                    hovertext=str(label), hoverinfo="y+text",
                ),
                row=row,
                col=col,
            )
        if p.show_box:
            # matplotlib overlays a real boxplot with `whis=(5, 95)`. `box_visible` plus `quartilemethod`
            # does NOT reproduce that: quartilemethod only picks how the quartiles are COMPUTED, and
            # go.Violin's inner box always whiskers to the data range -- so the same group showed a visibly
            # longer whisker in the HTML than in the PNG. Draw the box explicitly with the fences named.
            for group, label in drawable:
                _q1, _med, _q3 = (float(v) for v in np.percentile(group, (25, 50, 75)))
                _lo, _hi = (float(v) for v in np.percentile(group, (5, 95)))
                fig.add_trace(
                    go.Box(
                        x=[_truncate_label(label, _VIOLIN_LABEL_MAXLEN, keep_tail=6)],
                        q1=[_q1], median=[_med], q3=[_q3], lowerfence=[_lo], upperfence=[_hi],
                        boxpoints=False, width=0.12, showlegend=False, hoverinfo="skip",
                        line=dict(color="black", width=0.9), fillcolor="rgba(0,0,0,0)",
                    ),
                    row=row, col=col,
                )
        if empty:
            # A violin that silently vanishes reads as "this group has no spread", which is a different statement
            # from "this group has no data". matplotlib names the dropped groups in the panel title; plotly's
            # subplot titles are fixed at `make_subplots` time, before this runs, so the note goes in as a
            # subplot annotation carrying the same information.
            fig.add_annotation(
                x=0.5, y=1.0, xref="x domain", yref="y domain", xanchor="center", yanchor="bottom",
                text=f"no data: {', '.join(empty)}", showarrow=False, font=dict(size=8), row=row, col=col,
            )
        fig.update_xaxes(title_text=p.xlabel, row=row, col=col, tickangle=-30)
        fig.update_yaxes(title_text=p.ylabel, row=row, col=col, showgrid=p.grid)
        self._violin_tick_budget(fig, drawable, row, col)

    def _bar_tick_budget(self, fig, p: BarPanelSpec, row: int, col: int) -> None:
        """Subsample the bar category labels to what the axis can hold, as the matplotlib twin does.

        A flat "past 25 categories keep 20" cancelled out the size a builder deliberately bought:
        slice_finder and category_discriminability both grow the figure half an inch per bar, so a 40-bar
        chart had room for every label and twenty of them were dropped anyway.
        """
        from ._plotly_heatmap import _cell_domains

        _horizontal = getattr(p, "orientation", "vertical") == "horizontal"
        _axis = fig.layout[("yaxis" if _horizontal else "xaxis") + plotly_axis_suffix(fig, row, col, len(fig._grid_ref[0]) if fig._grid_ref else 1)]
        _text = list(_axis.ticktext or ())
        if not _text:
            return
        _dom = _cell_domains(fig, row, col)
        _extent_in = None
        if _dom is not None and fig.layout.width and fig.layout.height:
            (_x0, _x1), (_y0, _y1), _, _ = _dom
            _m = fig.layout.margin
            if _horizontal:
                _extent_in = (float(_y1) - float(_y0)) * max(float(fig.layout.height) - float(_m.t or 0) - float(_m.b or 0), 1.0) / _PX_PER_INCH
            else:
                _extent_in = (float(_x1) - float(_x0)) * max(float(fig.layout.width) - float(_m.l or 0) - float(_m.r or 0), 1.0) / _PX_PER_INCH
        if _horizontal:
            _pitch = rotated_tick_pitch_in(9, 0)
        else:
            _rot = float(_axis.tickangle or 0)
            _pitch = rotated_tick_pitch_in(9, _rot) if _rot else label_width_pitch_in(_text, 9)
        _keep = _thin_tick_positions(len(_text), ticks_that_fit(_extent_in, len(_text), pitch_in=_pitch))
        if len(_keep) < len(_text):
            _vals = list(_axis.tickvals or ())
            _axis.tickvals = [_vals[i] for i in _keep] if _vals else None
            _axis.ticktext = [_text[i] for i in _keep]

    def _violin_tick_budget(self, fig, drawable, row: int, col: int) -> None:
        """Thin the violin category labels to what the panel can hold, as the matplotlib twin does.

        Only matplotlib thinned them, so the same 20-class panel showed 18 labels in the PNG and all 20 --
        overlapping at -30 degrees -- in the HTML. Every violin still draws; only the labels subsample.
        """
        _names = [_truncate_label(lab, _VIOLIN_LABEL_MAXLEN, keep_tail=6) for _, lab in drawable]
        if not _names:
            return
        # Local import: the heatmap family is carved into a sibling that is bound onto this class at the
        # BOTTOM of this module, so a top-level import here would depend on that ordering.
        from ._plotly_heatmap import _cell_domains

        _dom = _cell_domains(fig, row, col)
        _w_px = fig.layout.width
        _w_in = None
        if _dom is not None and _w_px:
            (_x0, _x1), _, _, _ = _dom
            _m = fig.layout.margin
            _w_in = (float(_x1) - float(_x0)) * max(float(_w_px) - float(_m.l or 0) - float(_m.r or 0), 1.0) / _PX_PER_INCH
        _keep = _thin_tick_positions(len(_names), ticks_that_fit(_w_in, len(_names), pitch_in=rotated_tick_pitch_in(8, 30)))
        fig.update_xaxes(row=row, col=col, tickmode="array", tickvals=[_names[i] for i in _keep], ticktext=[_names[i] for i in _keep])


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
