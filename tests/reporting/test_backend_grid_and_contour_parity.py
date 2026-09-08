"""Three ways the same FigureSpec used to print at two visual densities on the two backends.

* A bar chart's CATEGORY axis carried gridlines on plotly and never on matplotlib, so the HTML had vertical
  rules running between every category that the PNG did not -- and plotly's gridlines are drawn at full
  strength against matplotlib's ``alpha=0.3``.
* PSI threshold contours carry a triage name ("moderate 0.1"). matplotlib draws it with ``clabel``; the
  plotly loop unpacked only level/colour/dash, leaving two anonymous coloured squiggles in the HTML.
* The scatter overlay band hardcoded the literal RGB of ``colors.OVERLAY_LINE`` on plotly only, so the next
  repaint of that constant would recolour PNGs and leave the HTML purple.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.colors import OVERLAY_LINE
from mlframe.reporting.renderers._plotly_color import _rgba
from mlframe.reporting.renderers.plotly import PlotlyRenderer, _GRID_COLOR
from mlframe.reporting.spec import BarPanelSpec, FigureSpec, HeatmapPanelSpec, ScatterPanelSpec

CATS = tuple(f"feature_{i}" for i in range(8))
VALS = np.linspace(0.1, 0.9, 8)


def _bar_figure(orientation: str) -> FigureSpec:
    """One bar panel in the requested orientation, gridlines on."""
    panel = BarPanelSpec(categories=CATS, values=VALS, title="t", xlabel="PSI", ylabel="feature", grid=True, orientation=orientation)
    return FigureSpec(panels=((panel,),), figsize=(7.0, 4.0))


@pytest.mark.parametrize("orientation,category_axis", [("vertical", "xaxis"), ("horizontal", "yaxis")])
def test_the_category_axis_carries_no_grid(orientation, category_axis):
    """A rule between every category is noise: matplotlib grids the VALUE axis only."""
    fig = PlotlyRenderer().render(_bar_figure(orientation))
    assert fig.layout[category_axis].showgrid is False, f"{orientation} bars keep a grid on the {category_axis} (the category axis)"


@pytest.mark.parametrize("orientation,value_axis", [("vertical", "yaxis"), ("horizontal", "xaxis")])
def test_the_value_axis_keeps_its_grid_at_matplotlib_weight(orientation, value_axis):
    """Guard: the fix must remove the category grid, not every grid."""
    fig = PlotlyRenderer().render(_bar_figure(orientation))
    axis = fig.layout[value_axis]
    assert axis.showgrid is not False, f"{orientation} bars lost the grid on the {value_axis} (the value axis)"
    assert axis.gridcolor == _GRID_COLOR, f"the value grid is {axis.gridcolor}, not the pinned {_GRID_COLOR}"


def test_the_threshold_contours_are_named():
    """Two coloured squiggles the reader cannot tell apart is the defect; the triage wording is the fix."""
    m = np.abs(np.random.default_rng(1).normal(0.12, 0.08, size=(12, 10)))
    panel = HeatmapPanelSpec(
        matrix=m,
        row_labels=tuple(f"f{i}" for i in range(12)),
        col_labels=tuple(f"b{i}" for i in range(10)),
        title="drift PSI",
        threshold_contours=((0.1, "orange", "dash", "moderate 0.1"), (0.25, "red", "solid", "significant 0.25")),
    )
    fig = PlotlyRenderer().render(FigureSpec(panels=((panel,),), figsize=(7.0, 5.0)))
    contours = [t for t in fig.data if t.type == "contour"]
    assert len(contours) == 2, f"expected both thresholds, got {len(contours)}"
    assert {t.name for t in contours} == {"moderate 0.1", "significant 0.25"}, f"the labels were dropped: {[t.name for t in contours]}"
    assert all(t.showlegend for t in contours), "a named contour that never reaches the legend is still anonymous on the page"
    assert all(t.contours.showlabels for t in contours), "the inline level label is off, so an HTML reader has only the legend"


def test_the_overlay_band_reads_the_shared_colour():
    """A literal RGB on one backend drifts silently the day the constant is repainted."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    panel = ScatterPanelSpec(x=x, y=x + rng.normal(scale=0.3, size=200), title="t", overlay_band=(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20) - 0.4, np.linspace(-2, 2, 20) + 0.4))
    fig = PlotlyRenderer().render(FigureSpec(panels=((panel,),), figsize=(6.0, 4.0)))
    fills = [t.fillcolor for t in fig.data if getattr(t, "fillcolor", None)]
    assert _rgba(OVERLAY_LINE, 0.18) in fills, f"the band is filled {fills}, not with {OVERLAY_LINE} at 0.18"


def _two_heatmaps() -> FigureSpec:
    """Two heatmap panels side by side, each drawing its own colorbar."""
    m = np.abs(np.random.default_rng(1).normal(0.12, 0.08, size=(12, 10)))
    panel = HeatmapPanelSpec(
        matrix=m,
        row_labels=tuple(f"f{i}" for i in range(12)),
        col_labels=tuple(f"b{i}" for i in range(10)),
        title="drift PSI",
        xlabel="bucket",
        ylabel="feature",
    )
    return FigureSpec(panels=((panel, panel),), figsize=(12.0, 4.5))


def test_a_colorbar_column_gap_clears_the_next_panels_axis_title():
    """A colorbar is pinned outside its subplot and its TICK LABELS stick out further still.

    At the default 0.08 column gap the left panel's colorbar labels landed on the right panel's y-axis
    title -- two pieces of text on top of each other, seen in the rendered PNG.
    """
    fig = PlotlyRenderer().render(_two_heatmaps())
    left_end, right_start = float(fig.layout.xaxis.domain[1]), float(fig.layout.xaxis2.domain[0])
    bar_x = float(fig.data[0].colorbar.x)
    assert bar_x > left_end, "the colorbar is inside its own panel rather than beside it"
    from mlframe.reporting.renderers._plotly_heatmap import _COLORBAR_GUTTER_PX

    gap_px = (right_start - bar_x) * float(fig.layout.width)
    assert (
        gap_px >= _COLORBAR_GUTTER_PX
    ), f"only {gap_px:.0f}px separates the colorbar from the next panel, under the {_COLORBAR_GUTTER_PX}px its bar and tick labels need"


def test_a_single_column_figure_keeps_the_tight_gap():
    """Guard: the wider gap is for the collision, not a blanket loosening of every layout."""
    import mlframe.reporting.renderers.plotly as plotly_mod

    one = _two_heatmaps()
    stacked = FigureSpec(panels=((one.panels[0][0],), (one.panels[0][1],)), figsize=(6.0, 9.0))
    fig = plotly_mod.PlotlyRenderer().render(stacked)
    assert len(fig.layout.xaxis.domain) == 2 and float(fig.layout.xaxis.domain[1]) == 1.0, "a one-column figure lost width to a gap it has no neighbour for"
