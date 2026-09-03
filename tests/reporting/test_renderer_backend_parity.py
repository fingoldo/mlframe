"""Cross-backend parity tests: the same FigureSpec must encode the same thing in matplotlib and plotly.

These divergences are invisible in a single-backend test suite -- each backend renders happily on its own and
only disagrees with its twin. A reader comparing the saved PNG against the interactive HTML sees two different
charts built from one spec, with no indication which is right.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import FigureSpec, HeatmapPanelSpec, LinePanelSpec


def _density_panel(**kw):
    """A pred-vs-actual density heatmap whose axes are category labels but whose trend arrives in value space."""
    rng = np.random.default_rng(0)
    xs = rng.uniform(3000.0, 6600.0, 4000)
    ys = xs + rng.normal(0.0, 300.0, 4000)
    edges = np.linspace(min(xs.min(), ys.min()), max(xs.max(), ys.max()), 11)
    matrix, _, _ = np.histogram2d(ys, xs, bins=[edges, edges])
    labels = tuple(f"{e:.0f}" for e in edges[:-1])
    return HeatmapPanelSpec(
        matrix=matrix, row_labels=labels, col_labels=labels, title="density",
        xlabel="pred", ylabel="true", colormap="viridis", **kw,
    ), labels


class TestFillBaselineShadesTheSameRegion:
    """plotly's `tonexty` fills to the PREVIOUS TRACE, not to a baseline."""

    def test_non_zero_baseline_gets_an_explicit_baseline_trace(self):
        """With fill_baseline != 0 the filled trace must have something at that y to fill against."""
        x = np.arange(10.0)
        spec = FigureSpec(
            panels=((LinePanelSpec(
                x=x, y=(np.full(10, 8.0), x + 1.0), series_labels=("other", "filled"),
                fill_to_baseline=(False, True), fill_baseline=5.0,
            ),),),
            figsize=(6.0, 4.0),
        )
        traces = list(PlotlyRenderer().render(spec).data)
        filled = [t for t in traces if t.fill == "tonexty"]
        assert len(filled) == 1
        # The trace immediately before the filled one must be the constant baseline, not the unrelated "other"
        # series -- pre-fix the shading ran between "filled" and "other", a region encoding nothing.
        prior = traces[traces.index(filled[0]) - 1]
        assert np.allclose(np.asarray(prior.y, dtype=float), 5.0)
        assert prior.showlegend is False


class TestHeatmapTrendLandsOnTheCategoryAxis:
    """The heatmap axes are categorical; a trend fitted in value space is thousands of positions off-grid."""

    def test_trend_endpoints_land_on_the_panels_own_axis(self):
        """Every plotted x/y must be somewhere the axis can place it, or the line is simply not on the chart.

        This used to require every coordinate to be a LABEL the axis already has, which forced the renderer to
        round and clamp each endpoint to the nearest category -- and that MOVES an extrapolated endpoint to the
        axis edge, changing the drawn segment's slope, which is the one thing the panel exists to show. A
        category axis also accepts a numeric POSITION, so the robust fit is now drawn at fractional bin indices:
        measured on a 20-bin panel, x spans 2.88..15.79 with slope 1.004 against a true 1.0, where the clamped
        form collapsed both endpoints to integers. The y=x reference stays on label strings, since its endpoints
        are the first and last categories by construction.

        The real contract is that nothing lands off-grid: a label the axis has, or an index inside its range.
        """
        panel, labels = _density_panel()
        rng = np.random.default_rng(0)
        xs = rng.uniform(3000.0, 6600.0, 4000)
        panel = HeatmapPanelSpec(
            matrix=panel.matrix, row_labels=labels, col_labels=labels, title=panel.title,
            colormap="viridis", trend_line="theil-sen", trend_xy=(xs, xs + rng.normal(0.0, 300.0, 4000)),
        )
        fig = PlotlyRenderer().render(FigureSpec(panels=((panel,),), figsize=(6.0, 5.0)))
        named = [t for t in fig.data if t.name in {"y=x", "robust fit (theil-sen)"}]
        assert len(named) == 2
        for trace in named:
            for value in list(trace.x) + list(trace.y):
                if isinstance(value, str):
                    assert value in labels
                else:
                    assert -1.0 <= float(value) <= len(labels), f"{value} is off the {len(labels)}-category axis"

    def test_the_trend_slope_is_not_quantised_by_the_axis(self):
        """The defect the label requirement caused: rounding both endpoints changes the line's slope."""
        panel, labels = _density_panel()
        rng = np.random.default_rng(0)
        xs = rng.uniform(3000.0, 6600.0, 4000)
        panel = HeatmapPanelSpec(
            matrix=panel.matrix, row_labels=labels, col_labels=labels, title=panel.title,
            colormap="viridis", trend_line="theil-sen", trend_xy=(xs, xs + rng.normal(0.0, 300.0, 4000)),
        )
        fig = PlotlyRenderer().render(FigureSpec(panels=((panel,),), figsize=(6.0, 5.0)))
        fit = next(t for t in fig.data if t.name == "robust fit (theil-sen)")
        x0, x1 = (float(v) for v in fit.x)
        y0, y1 = (float(v) for v in fit.y)
        assert (y1 - y0) / (x1 - x0) == pytest.approx(1.0, abs=0.05), (fit.x, fit.y)


class TestHeatmapRowOrderMatchesMatplotlib:
    """matplotlib flips to origin='lower' for a density panel; reversing plotly unconditionally mirrors it."""

    def test_density_panel_is_not_reversed(self):
        """A panel carrying trend_xy reads bottom-up in BOTH backends."""
        rng = np.random.default_rng(0)
        xs = rng.uniform(3000.0, 6600.0, 4000)
        panel, _ = _density_panel(trend_line="theil-sen", trend_xy=(xs, xs + rng.normal(0.0, 300.0, 4000)))
        fig = PlotlyRenderer().render(FigureSpec(panels=((panel,),), figsize=(6.0, 5.0)))
        assert fig.layout.yaxis.autorange != "reversed"

    def test_plain_heatmap_keeps_the_top_down_matrix_order(self):
        """A confusion / drift heatmap has no trend_xy and must still read top-down, matching imshow's default."""
        panel, _ = _density_panel()
        fig = PlotlyRenderer().render(FigureSpec(panels=((panel,),), figsize=(6.0, 5.0)))
        assert fig.layout.yaxis.autorange == "reversed"

    def test_both_backends_render_the_density_panel(self):
        """Parity fix must not break either backend's own rendering path."""
        rng = np.random.default_rng(0)
        xs = rng.uniform(3000.0, 6600.0, 4000)
        panel, _ = _density_panel(trend_line="theil-sen", trend_xy=(xs, xs + rng.normal(0.0, 300.0, 4000)))
        spec = FigureSpec(panels=((panel,),), figsize=(6.0, 5.0))
        assert MatplotlibRenderer().render(spec) is not None
        assert PlotlyRenderer().render(spec) is not None
