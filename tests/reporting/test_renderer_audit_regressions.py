"""Regression tests for the reporting_renderers audit findings.

Most of these are CROSS-BACKEND: each backend rendered happily on its own and only disagreed with its twin,
which a single-backend suite cannot see. A reader comparing the saved PNG against the interactive HTML got
two different charts from one spec with no indication which was right.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mlframe.reporting.colors import OVERLAY_LINE, TREND_LINE
from mlframe.reporting.renderers import _shared_helpers as shared
from mlframe.reporting.renderers import matplotlib as mpl_mod
from mlframe.reporting.renderers import plotly as plotly_mod
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer, _marker_symbol
from mlframe.reporting.spec import (
    BarPanelSpec, ConfusionMarginsPanelSpec, FigureSpec, HeatmapPanelSpec, LinePanelSpec,
    ScatterPanelSpec, ViolinPanelSpec,
)


def _labels(k):
    """K class labels."""
    return tuple(f"class_{i}" for i in range(k))


class TestDegenerateInputDoesNotCrashOneBackendOnly:
    """A spec one backend renders happily must not crash the other."""

    def test_matplotlib_violin_survives_an_empty_group(self):
        """ax.violinplot raises on an empty group, where plotly renders the rest happily."""
        rng = np.random.default_rng(0)
        panel = ViolinPanelSpec(
            groups=(rng.normal(0.0, 1.0, 200), np.array([]), rng.normal(1.0, 1.0, 200)),
            group_labels=("a", "empty", "c"), title="scores",
        )
        ax = MatplotlibRenderer().render(FigureSpec(panels=((panel,),), figsize=(7.0, 4.0))).get_axes()[0]
        # Dropped silently, the missing violin reads as "this group has no spread" -- a different claim
        # from "this group has no data" -- so the dropped group is named in the title.
        assert "no data: empty" in ax.get_title()
        assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "c"]

    def test_violin_groups_are_coloured_per_group(self):
        """plotly cycles a colour per group; matplotlib drew every one in the same default blue."""
        rng = np.random.default_rng(0)
        panel = ViolinPanelSpec(groups=(rng.normal(0, 1, 100), rng.normal(1, 1, 100)), group_labels=("a", "b"))
        ax = MatplotlibRenderer().render(FigureSpec(panels=((panel,),), figsize=(6.0, 4.0))).get_axes()[0]
        faces = [tuple(np.round(c.get_facecolor()[0][:3], 3)) for c in ax.collections[:2]]
        assert faces[0] != faces[1]


class TestScatterLegendLabel:
    """legend_label is part of the spec, so both backends must honour it."""

    def test_matplotlib_passes_legend_label_to_the_artist(self):
        """Honoured by plotly, dropped by matplotlib -- which then drew an EMPTY legend box."""
        rng = np.random.default_rng(0)
        panel = ScatterPanelSpec(x=rng.random(50), y=rng.random(50), legend_label="my series")
        fig = MatplotlibRenderer().render(FigureSpec(panels=((panel,),), figsize=(5.0, 4.0)))
        legend = fig.get_axes()[0].get_legend()
        assert legend is not None
        assert [t.get_text() for t in legend.get_texts()] == ["my series"]


class TestPlotlyGridAndSizing:
    """Grid shape and figure size must match what the spec asked for."""

    def test_a_none_cell_creates_no_subplot(self):
        """An empty dict is a DEFAULT xy subplot to plotly, not the absence of one."""
        x = np.arange(10.0)
        line = LinePanelSpec(x=x, y=x)
        spec = FigureSpec(panels=((line, line), (line, None)), figsize=(10.0, 6.0))
        fig = PlotlyRenderer().render(spec)
        # Pre-fix: 4 axes for 3 panels, drawing an empty framed box matplotlib leaves blank.
        assert len([k for k in fig.layout if k.startswith("xaxis")]) == 3

    def test_plot_area_is_not_shrunk_below_figsize(self):
        """Height must add BOTH margins; adding only the top one shrank the plot area."""
        x = np.arange(10.0)
        spec = FigureSpec(panels=((LinePanelSpec(x=x, y=x),),), figsize=(8.0, 6.0))
        fig = PlotlyRenderer().render(spec)
        assert fig.layout.height > 6.0 * plotly_mod._PX_PER_INCH

    def test_each_heatmap_gets_its_own_colorbar_position(self):
        """Left at the default, every colorbar falls to the same paper position and they stack."""
        m = np.arange(9.0).reshape(3, 3)
        labs = _labels(3)

        def hp(title):
            """One heatmap panel."""
            return HeatmapPanelSpec(matrix=m, row_labels=labs, col_labels=labs, title=title, colorbar_label="v")

        fig = PlotlyRenderer().render(FigureSpec(panels=((hp("one"), hp("two")),), figsize=(12.0, 5.0)))
        xs = [t.colorbar.x for t in fig.data if t.colorbar is not None]
        assert len(xs) == 2 and xs[0] != xs[1]


class TestTickThinningMatchesAcrossBackends:
    """A dense axis must be readable on both backends, not just the one that thins."""

    def test_matplotlib_confusion_margins_thins_like_plotly(self):
        """One tick per class smears past ~30; plotly already thinned, matplotlib did not."""
        k = 30
        m = np.random.default_rng(0).random((k, k))
        labs = _labels(k)
        panel = ConfusionMarginsPanelSpec(
            matrix=m, row_labels=labs, col_labels=labs,
            row_margin=np.arange(k, dtype=float), col_margin=np.arange(k, dtype=float),
            row_margin_label="support", col_margin_label="volume", title="cm",
        )
        fig = MatplotlibRenderer().render(FigureSpec(panels=((panel,),), figsize=(8.0, 8.0)))
        heat_ax = next(a for a in fig.get_axes() if a.images)
        assert len(heat_ax.get_xticks()) == len(shared._thin_tick_positions(k))

    def test_matplotlib_horizontal_bar_labels_are_thinned_and_capped(self):
        """A 200-category horizontal FI chart smeared its y axis; plotly already capped labels."""
        cats = tuple(f"a_very_long_generated_feature_name_number_{i}" * 2 for i in range(200))
        panel = BarPanelSpec(categories=cats, values=np.arange(200.0), orientation="horizontal", title="FI")
        ax = MatplotlibRenderer().render(FigureSpec(panels=((panel,),), figsize=(8.0, 10.0))).get_axes()[0]
        drawn = [t.get_text() for t in ax.get_yticklabels()]
        assert len(drawn) <= mpl_mod._BAR_TICK_KEEP + 1
        assert max(len(s) for s in drawn) <= shared._BAR_LABEL_MAXLEN + 2  # +2 for the ellipsis, as plotly


class TestSharedConstantsHaveOneDefinition:
    """Constants whose whole purpose is identical cross-backend behaviour must not be duplicated."""

    @pytest.mark.parametrize("name", ["_SCATTER_MAX_POINTS", "_HIST_PREBIN_THRESHOLD", "_HEATMAP_CELL_TEXT_MAX"])
    def test_both_renderers_read_the_same_object(self, name):
        """Two copies of such a number is a drift waiting to happen."""
        assert getattr(mpl_mod, name) is getattr(shared, name) is getattr(plotly_mod, name)


class TestOverlayColoursAreNamed:
    """Nine hardcoded literals across two backends had to change in nine places or they drifted."""

    def test_renderers_use_the_shared_palette_constants(self):
        """No renderer should carry a raw overlay colour literal any more."""
        assert TREND_LINE and OVERLAY_LINE
        for mod in (mpl_mod, plotly_mod):
            with open(mod.__file__, encoding="utf-8") as fh:
                src = fh.read()
            assert 'color="darkorange"' not in src
            assert 'color="purple"' not in src


class TestUnmappedMarkerIsAudible:
    """Falling back is right; doing it silently is not."""

    def test_a_known_marker_maps(self):
        """The mapping table is unchanged for tokens it knows."""
        assert _marker_symbol("D") == "diamond"

    def test_an_unknown_marker_warns_once_then_falls_back(self, caplog):
        """Silently turning a deliberate marker into a star made the backends disagree with no signal."""
        plotly_mod._MARKER_WARNED.discard("v")
        with caplog.at_level("WARNING"):
            assert _marker_symbol("v") == "star"
            assert _marker_symbol("v") == "star"
        assert sum("no plotly equivalent" in r.message for r in caplog.records) == 1


class TestCaptionKeepsAuthorLineBreaks:
    """Captions are written with deliberate structure; wrapping must not re-flow it away."""

    def test_a_newline_in_a_caption_survives_wrapping(self):
        """textwrap.wrap treats a newline as ordinary whitespace and collapsed the structure."""
        x = np.arange(10.0)
        caption = "First clause on its own line.\nVERDICT: the second line is deliberate."
        fig = MatplotlibRenderer().render(FigureSpec(panels=((LinePanelSpec(x=x, y=x),),), figsize=(8.0, 5.0), caption=caption))
        rendered = [t.get_text() for t in fig.texts if "VERDICT" in t.get_text()]
        assert rendered and "\n" in rendered[0]


class TestPlotlyModuleStaysUnderTheLocLimit:
    """The carve must keep every call site resolving exactly as before."""

    def test_carved_methods_are_bound_and_callable(self):
        """renderer._heatmap(...) must still resolve, and come from the sibling module."""
        m = np.arange(9.0).reshape(3, 3)
        labs = _labels(3)
        panel = HeatmapPanelSpec(matrix=m, row_labels=labs, col_labels=labs, title="h", cell_text=m, text_format=".0f")
        assert PlotlyRenderer().render(FigureSpec(panels=((panel,),), figsize=(6.0, 5.0))) is not None
        assert PlotlyRenderer._heatmap.__module__.endswith("_plotly_heatmap")
