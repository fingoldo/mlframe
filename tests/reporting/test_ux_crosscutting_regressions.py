"""Regression tests for the reporting_ux_crosscutting audit findings.

Three themes: a figure that reports a NUMBER where nothing could be measured, a figure that computes the verdict
and then does not say it, and a spec field one backend honours and the other silently drops.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mlframe.reporting.charts.multiclass import compose_multiclass_figure
from mlframe.reporting.charts.multilabel import compose_multilabel_figure
from mlframe.reporting.charts.quantile import _TOKEN_BUILDERS as QUANTILE_BUILDERS
from mlframe.reporting.charts.regression import compose_regression_figure
from mlframe.reporting.charts.risk_coverage import build_risk_coverage_spec
from mlframe.reporting.charts.spectral_embedding import compose_spectral_embedding_figure
from mlframe.reporting.colors import LINE_PALETTE, line_style
from mlframe.reporting.renderers import get_renderer
from mlframe.reporting.spec import (
    AnnotationPanelSpec, BarPanelSpec, FigureSpec, LinePanelSpec, ScatterPanelSpec, ViolinPanelSpec,
)


def _panels(fig):
    """Flatten a FigureSpec's panel grid."""
    return [p for row in fig.panels for p in row if p is not None]


class TestNoFabricatedNumbersOnEmptyInput:
    """A builder with no usable row must SAY so, not print a zero that reads as a measurement."""

    @pytest.mark.parametrize("token", sorted(QUANTILE_BUILDERS))
    def test_every_quantile_panel_annotates_at_n_zero(self, token):
        """Four of ten annotated; the rest quoted zeros as measured."""
        panel = QUANTILE_BUILDERS[token](np.empty(0), np.empty((0, 3)), (0.1, 0.5, 0.9))
        assert isinstance(panel, AnnotationPanelSpec)

    def test_every_regression_panel_annotates_at_n_zero(self):
        """The scatter rendered bare axes with an EMPTY title, and the decile bar drew a fabricated zero."""
        panels = _panels(compose_regression_figure(np.empty(0), np.empty(0)))
        assert panels and all(isinstance(p, AnnotationPanelSpec) for p in panels)
        assert all(p.text for p in panels)

    def test_multilabel_and_multiclass_short_circuit_the_whole_figure(self):
        """Both drew a full grid on zero rows, including a per-row Jaccard of 0.000."""
        ml = _panels(compose_multilabel_figure(np.empty((0, 3)), np.empty((0, 3)), ["a", "b", "c"]))
        mc = _panels(compose_multiclass_figure(np.empty(0, dtype=int), np.empty((0, 3)), [0, 1, 2]))
        assert len(ml) == len(mc) == 1
        assert isinstance(ml[0], AnnotationPanelSpec) and isinstance(mc[0], AnnotationPanelSpec)

    def test_risk_coverage_blames_the_data_not_the_model(self):
        """The all-NaN curve was titled "constant confidence: no ranking signal" -- a verdict about the MODEL."""
        res = build_risk_coverage_spec(np.empty(0), np.empty(0))
        panel = _panels(res.figure)[0]
        assert isinstance(panel, AnnotationPanelSpec)
        assert "rows" in panel.text

    @pytest.mark.parametrize(("n_nodes", "edges"), [(0, []), (6, [])])
    def test_spectral_embedding_guards_both_degenerate_graphs(self, n_nodes, edges):
        """Zero nodes raised out of the whole report; an edgeless graph returned arbitrary coordinates."""
        panel = _panels(compose_spectral_embedding_figure(n_nodes, edges))[0]
        assert isinstance(panel, AnnotationPanelSpec)


class TestSpecFieldsAreHonouredByBothBackends:
    """A field one backend reads and the other drops makes the PNG and the HTML disagree with no signal."""

    def test_matplotlib_shares_the_x_axis_when_asked(self):
        """sharex reached only a colorbar anchor; the calibration scatter + population stack never shared x here."""
        x = np.arange(10.0)
        spec = FigureSpec(panels=((LinePanelSpec(x=x, y=x),), (LinePanelSpec(x=x, y=x * 2),)), figsize=(6.0, 6.0), sharex=True)
        axes = get_renderer("matplotlib").render(spec).get_axes()
        assert axes[1] in axes[0].get_shared_x_axes().get_siblings(axes[0])

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
    def test_equal_aspect_applies_without_a_perfect_fit_line(self, backend):
        """Both backends handled the flag only INSIDE the perfect-fit branch, so it was a no-op alone."""
        panel = ScatterPanelSpec(x=np.arange(10.0), y=np.arange(10.0) * 2, equal_aspect=True, perfect_fit_line=False)
        fig = get_renderer(backend).render(FigureSpec(panels=((panel,),), figsize=(5.0, 5.0)))
        if backend == "matplotlib":
            assert fig.get_axes()[0].get_aspect() == 1.0
        else:
            assert fig.layout.yaxis.scaleanchor is not None

    def test_plotly_places_the_legend_outside_when_asked(self):
        """legend_outside/legend_ncol were matplotlib-only, so HTML covered the very curves they protect."""
        x = np.arange(10.0)
        panel = LinePanelSpec(x=x, y=(x, x * 2), series_labels=("a", "b"), legend_outside=True, legend_ncol=2)
        fig = get_renderer("plotly").render(FigureSpec(panels=((panel,),), figsize=(6.0, 4.0)))
        assert fig.layout.legend.x >= 1.0 and fig.layout.legend.xanchor == "left"

    def test_matplotlib_violin_shows_a_quartile_box_like_plotly(self):
        """A median LINE against a full quartile box is two different amounts of information from one field."""
        rng = np.random.default_rng(0)
        panel = ViolinPanelSpec(groups=(rng.normal(size=200), rng.normal(1.0, 1.0, 200)), group_labels=("a", "b"), show_box=True)
        ax = get_renderer("matplotlib").render(FigureSpec(panels=((panel,),), figsize=(6.0, 4.0))).get_axes()[0]
        assert len(ax.lines) >= 4  # median + box + whiskers, per group


class TestPlotlyHoverIsInformative:
    """Only the heatmap set a hovertemplate; everything else fell back to raw coordinates and a trace id."""

    @pytest.mark.parametrize("panel", [
        ScatterPanelSpec(x=np.arange(10.0), y=np.arange(10.0), xlabel="predicted", ylabel="true"),
        BarPanelSpec(categories=("a", "b"), values=np.array([1.0, 2.0]), xlabel="class", ylabel="count"),
    ])
    def test_axis_names_reach_the_tooltip(self, panel):
        """The axis labels the spec already carries are exactly what names those numbers."""
        fig = get_renderer("plotly").render(FigureSpec(panels=((panel,),), figsize=(6.0, 4.0)))
        templates = [t.hovertemplate for t in fig.data if t.hovertemplate]
        assert templates and panel.xlabel in templates[0] and panel.ylabel in templates[0]

    def test_builder_supplied_support_text_wins(self):
        """A rate from 3 rows rendered identically to one from 300k; the count is what a reader needs."""
        panel = BarPanelSpec(categories=("a", "b"), values=np.array([0.5, 0.2]), hovertext=("a: 3 rows", "b: 300,000 rows"))
        fig = get_renderer("plotly").render(FigureSpec(panels=((panel,),), figsize=(6.0, 4.0)))
        assert any(t.hovertext == ("a: 3 rows", "b: 300,000 rows") for t in fig.data)

    def test_a_line_panel_keeps_spike_lines_when_a_heatmap_shares_the_figure(self):
        """One heatmap anywhere disabled unified hover for every line panel in the figure."""
        from mlframe.reporting.spec import HeatmapPanelSpec
        x = np.arange(10.0)
        labels = ("a", "b", "c")
        spec = FigureSpec(
            panels=((LinePanelSpec(x=x, y=x), HeatmapPanelSpec(matrix=np.arange(9.0).reshape(3, 3), row_labels=labels, col_labels=labels)),),
            figsize=(10.0, 4.0),
        )
        fig = get_renderer("plotly").render(spec)
        assert fig.layout.xaxis.showspikes


class TestPaletteSurvivesColourVisionDeficiency:
    """Colour was the only channel separating series, on a palette whose extension collides under deuteranopia."""

    def test_the_lightness_variants_are_gone(self):
        """A tab10 hue and its tab20 twin separate by 2.8 under simulation, against 14.6 for the worst tab10 pair."""
        assert len(LINE_PALETTE) == 10

    def test_the_dash_pattern_changes_on_every_palette_wrap(self):
        """Past 10 series the colour repeats, so the dash is the only thing left to tell them apart."""
        assert line_style(0) != line_style(len(LINE_PALETTE))
