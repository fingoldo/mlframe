"""Pure-data chart specifications.

A spec is a frozen dataclass describing one panel (or one full figure
made of panels). Specs contain only data + style hints; no matplotlib
or plotly objects. The same spec can be rendered identically on either
backend by ``Renderer.render(spec)``.

Panels are typed by visual idiom (Scatter / Histogram / Heatmap / Bar /
Line / Violin) rather than by domain (calibration / regression / ...) so
the same panel type is reused across charts (e.g. HistogramPanelSpec is
used by both regression residuals and calibration bin-population).

A figure is a 2-D grid of panels (``FigureSpec.panels[row][col]``); cells
can be ``None`` to leave space empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np

from mlframe.reporting import colors as _colors


# ----------------------------------------------------------------------------
# Panel specs
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ScatterPanelSpec:
    """Scatter plot. Optional perfect-fit diagonal line for calibration /
    regression scatter use cases."""

    x: np.ndarray
    y: np.ndarray
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    perfect_fit_line: bool = False
    # ``point_color``: scalar string => uniform; ndarray same length as x =>
    # colormap-driven (one color per point via ``colormap``).
    point_color: Optional[Union[str, np.ndarray]] = None
    colormap: str = _colors.CALIBRATION
    point_alpha: float = 0.3
    # ``point_size``: scalar => uniform; ndarray => per-point sizes (e.g.
    # bin-population for calibration scatter).
    point_size: Union[float, np.ndarray] = 10.0
    # Inline text labels: list of ``(x, y, text)`` tuples drawn next to points.
    inline_labels: Optional[Tuple[Tuple[float, float, str], ...]] = None
    legend_label: Optional[str] = None
    grid: bool = True
    # When the colorbar represents a meaningful axis (e.g. bin population),
    # set ``colorbar_label`` so renderers add a labelled colorbar.
    colorbar_label: Optional[str] = None


@dataclass(frozen=True)
class HistogramPanelSpec:
    """Histogram with optional fitted-Normal overlay (regression residuals)
    and per-bar colormap (calibration bin-population)."""

    values: np.ndarray
    bins: int = 30
    title: str = ""
    xlabel: str = ""
    ylabel: str = "Density"
    color: str = _colors.BAR_PRIMARY
    # ``bar_colors``: when supplied, each bar gets its own color via the
    # named colormap (used by calibration bin-population to match the
    # scatter's colorbar). Length must equal ``bins`` (matplotlib auto
    # bins); for pre-binned data pass ``values=bin_counts`` + custom kwarg.
    bar_colors: Optional[np.ndarray] = None
    colormap: str = _colors.CALIBRATION
    # Fit a Normal density on top: pass (mu, sigma).
    overlay_normal: Optional[Tuple[float, float]] = None
    overlay_label: Optional[str] = None
    yscale: Literal["linear", "log"] = "linear"
    grid: bool = True
    density: bool = True  # normalize histogram to PDF (matplotlib density=True)
    # When ``values`` is pre-binned (e.g. calibration ``hits``), pass the
    # x-axis bin centers here. Renderers then draw a bar plot at those
    # centers instead of computing histogram bins.
    bin_centers: Optional[np.ndarray] = None
    bin_width: Optional[float] = None  # required when bin_centers given


@dataclass(frozen=True)
class HeatmapPanelSpec:
    """Heatmap with optional cell-text overlay (e.g. confusion matrix
    counts written in each cell)."""

    matrix: np.ndarray
    row_labels: Tuple[str, ...]
    col_labels: Tuple[str, ...]
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    colormap: str = _colors.HEATMAP_GENERIC
    # Per-cell text: same shape as ``matrix``. Useful for showing raw
    # counts on top of normalised heatmap colors.
    cell_text: Optional[np.ndarray] = None
    text_format: str = ".2f"
    colorbar_label: Optional[str] = None


@dataclass(frozen=True)
class BarPanelSpec:
    """Single-series or grouped bar chart.

    For grouped bars: pass ``values`` as a tuple of equal-length
    arrays, one per series, and ``series_labels`` as the legend.
    """

    categories: Tuple[str, ...]
    # 1-D ndarray => single series; tuple of 1-D ndarrays => grouped.
    values: Union[np.ndarray, Tuple[np.ndarray, ...]]
    series_labels: Optional[Tuple[str, ...]] = None
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    colors: Optional[Tuple[str, ...]] = None
    grid: bool = True
    # Rotate x-tick labels (useful for long category names).
    xtick_rotation: float = 0.0


@dataclass(frozen=True)
class LinePanelSpec:
    """Line plot with one or more series.

    For multi-series: pass ``y`` as a tuple of arrays and
    ``series_labels`` for the legend.
    """

    x: np.ndarray
    y: Union[np.ndarray, Tuple[np.ndarray, ...]]
    series_labels: Optional[Tuple[str, ...]] = None
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    # Per-series draw style. matplotlib linestyle tokens ('-', '--', ':', '-.') plus two extra
    # tokens: "markers" (marker-only series, e.g. y_true as points under a fitted line) and
    # "lines+markers". Cycled when shorter than the series tuple.
    line_styles: Optional[Tuple[str, ...]] = None
    colors: Optional[Tuple[str, ...]] = None
    grid: bool = True
    # Vertical reference lines: tuple of (x, color, label). label may be "" for unlabeled.
    vlines: Optional[Tuple[Tuple[float, str, str], ...]] = None
    # Shaded vertical spans (change-points / train-val-test split shading): (x0, x1, color, alpha).
    vspans: Optional[Tuple[Tuple[float, float, str, float], ...]] = None
    # X axis carries timestamps: renderers rotate/format tick labels (mpl autofmt_xdate, plotly tickangle).
    x_is_time: bool = False
    # Shaded band between two series (interval bands, metric +- std over time): (lower, upper) arrays
    # of the same length as ``x``. Drawn underneath the line series.
    band: Optional[Tuple[np.ndarray, np.ndarray]] = None
    band_color: Optional[str] = None  # defaults to the first series color
    band_label: Optional[str] = None


@dataclass(frozen=True)
class ViolinPanelSpec:
    """Per-group violin plot (e.g. probability distribution per true class)."""

    groups: Tuple[np.ndarray, ...]
    group_labels: Tuple[str, ...]
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    show_box: bool = True
    grid: bool = True


@dataclass(frozen=True)
class NetworkPanelSpec:
    """Node-link (graph) panel for feature-relationship diagrams.

    Layout positions are precomputed by the caller (the spec stays pure data,
    so renderers never depend on a graph-layout library). Nodes and edges are
    stored as flat parallel arrays so a renderer can emit the whole graph in
    one or two traces regardless of node count -- the plotly backend batches
    all edges into a single ``Scattergl`` line trace and all nodes into one
    ``Scattergl`` marker trace, which scales to the WebGL ~100k-point regime.

    Edge endpoints are integer indices into the node arrays. ``node_size`` uses
    matplotlib ``scatter(s=...)`` area semantics (points-squared) so both
    backends size markers identically. Per-node ``node_color`` is a resolved
    color string (the domain layer maps its classes to colors), keeping the
    spec backend-agnostic. ``edge_weight`` drives both edge width and edge color
    through ``colormap``; ``edge_directed`` selects which edges draw an arrow.
    """

    node_x: np.ndarray
    node_y: np.ndarray
    node_size: np.ndarray            # matplotlib area units (pt^2)
    node_color: Tuple[str, ...]      # one resolved color per node
    node_label: Tuple[str, ...]
    edge_src: np.ndarray             # int index into node arrays
    edge_dst: np.ndarray             # int index into node arrays
    edge_weight: np.ndarray          # float; drives width + color
    title: str = ""
    xlabel: str = ""
    ylabel: str = ""
    # Rich per-node hover text (plotly tooltips). When None, ``node_label`` is used.
    node_hovertext: Optional[Tuple[str, ...]] = None
    # Per-edge arrow flag (len == edges) or a single bool for all edges.
    edge_directed: Union[bool, np.ndarray] = False
    colormap: str = _colors.HEATMAP_GENERIC
    colorbar_label: Optional[str] = None
    # Optional (label, color) pairs for a node-class legend (e.g. unique / sink / middling).
    node_legend: Optional[Tuple[Tuple[str, str], ...]] = None
    # Edge width range (min, max) in points; weights are linearly mapped into it.
    edge_width_range: Tuple[float, float] = (0.5, 6.0)


@dataclass(frozen=True)
class AnnotationPanelSpec:
    """Centered free-text panel on empty axes.

    Replaces fake degenerate-chart placeholders (e.g. a [0.0] histogram standing in for
    "metric unavailable") with an honest text cell.
    """

    text: str
    title: str = ""
    fontsize: int = 10


# Type alias: union of all panel specs (renderers dispatch on isinstance).
PanelSpec = Union[
    ScatterPanelSpec,
    HistogramPanelSpec,
    HeatmapPanelSpec,
    BarPanelSpec,
    LinePanelSpec,
    ViolinPanelSpec,
    NetworkPanelSpec,
    AnnotationPanelSpec,
    None,  # ``None`` = empty grid cell
]


# ----------------------------------------------------------------------------
# Top-level figure spec
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureSpec:
    """A grid of panels + optional figure-level suptitle.

    ``panels`` is rows-then-cols: ``panels[row][col]``. Cells may be
    ``None`` to leave the slot blank. Layout sizing:
    - rows = ``len(panels)``
    - cols = ``max(len(row) for row in panels)``
    """

    suptitle: str = ""
    panels: Tuple[Tuple[PanelSpec, ...], ...] = field(default_factory=tuple)
    figsize: Tuple[float, float] = (12.0, 4.0)
    # Per-figure DPI for saved PNG. None defers to matplotlib's default; set
    # via ReportingConfig.plot_dpi to control globally (~30% chart wall savings
    # at dpi=80 vs default 100).
    dpi: Optional[int] = None
    # constrained_layout default False: the iterative tight-bbox solver adds
    # ~700-800 ms per multi-panel figure (~2x slowdown vs default geometry)
    # without visible benefit on standard panel layouts. Override per-spec
    # when bbox-aware solving is genuinely needed (e.g. multi-line suptitles).
    constrained_layout: bool = False
    suptitle_fontsize: int = 11
    # Optional: row height ratios (e.g. (3, 1) for calibration scatter +
    # smaller bin-population panel). Length must equal len(panels).
    row_height_ratios: Optional[Tuple[float, ...]] = None
    # Optional: column width ratios (rare; used for asymmetric layouts).
    col_width_ratios: Optional[Tuple[float, ...]] = None
    # Optional: shared X / Y axes between panels (matplotlib sharex/sharey
    # and plotly equivalent). True = share within columns / rows.
    sharex: bool = False
    sharey: bool = False


__all__ = [
    "ScatterPanelSpec",
    "HistogramPanelSpec",
    "HeatmapPanelSpec",
    "BarPanelSpec",
    "LinePanelSpec",
    "ViolinPanelSpec",
    "NetworkPanelSpec",
    "AnnotationPanelSpec",
    "PanelSpec",
    "FigureSpec",
]
