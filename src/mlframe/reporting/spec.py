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
    line_styles: Optional[Tuple[str, ...]] = None  # e.g. ('-', '--', ':')
    colors: Optional[Tuple[str, ...]] = None
    grid: bool = True


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


# Type alias: union of all panel specs (renderers dispatch on isinstance).
PanelSpec = Union[
    ScatterPanelSpec,
    HistogramPanelSpec,
    HeatmapPanelSpec,
    BarPanelSpec,
    LinePanelSpec,
    ViolinPanelSpec,
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
    "PanelSpec",
    "FigureSpec",
]
