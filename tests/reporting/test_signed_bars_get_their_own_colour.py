"""Every bar in the WoE chart was painted the colour of the FIRST bar, while the title said colour meant the sign.

`category_discriminability` builds a per-bar `colors` tuple -- green where the level tilts toward y=1, red where
it tilts toward y=0 -- and its own title tells the reader so. Both renderers read only `p.colors[0]` on a
single-series `BarPanelSpec`, so the whole chart came out one colour. Rows are ranked by |WoE| descending, so
which colour that was depended on whichever level happened to have the largest absolute WoE: a dataset whose
strongest level tilts positive rendered every bar green, including the negative ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.renderers.matplotlib import _bar_colors
from mlframe.reporting.spec import BarPanelSpec

VALUES = np.array([1.5, -1.2, 0.8, -0.4])
SIGNED = ("green", "red", "green", "red")


class TestPerBarColoursSurvive:
    """The colour tuple is the chart's only encoding of the sign."""

    def test_a_full_length_tuple_is_passed_through(self):
        """One colour per bar, in order."""
        assert _bar_colors(SIGNED, VALUES) == list(SIGNED)

    def test_a_single_colour_still_applies_to_every_bar(self):
        """The grouped/one-series case must keep working."""
        assert _bar_colors(("steelblue",), VALUES) == "steelblue"

    def test_no_colours_falls_back_to_the_default(self):
        """Unchanged contract."""
        assert _bar_colors(None, VALUES) == "steelblue"

    def test_a_mismatched_length_is_not_treated_as_per_bar(self):
        """Two colours against four bars is a series tuple, not a per-bar one; matplotlib would cycle it."""
        assert _bar_colors(("green", "red"), VALUES) == "green"

    def test_the_negative_bars_are_not_painted_the_positive_colour(self):
        """The user-visible defect, stated as the property that failed."""
        resolved = _bar_colors(SIGNED, VALUES)
        assert resolved[1] != resolved[0] and resolved[3] != resolved[2]


class TestTheChartStillProducesSignedColours:
    """The builder side, so the renderer fix has something to render."""

    @staticmethod
    def _panel(seed):
        """A WoE panel over four levels whose positive rates straddle the base rate."""
        import pandas as pd

        from mlframe.reporting.charts.category_discriminability import category_discriminability_panel

        rng = np.random.default_rng(seed)
        n = 4000
        level = rng.integers(0, 4, n)
        y = (rng.random(n) < np.array([0.05, 0.25, 0.65, 0.95])[level]).astype(int)
        return category_discriminability_panel(pd.DataFrame({"lvl": level.astype(str)}), y)

    def test_the_woe_chart_emits_one_colour_per_bar(self):
        """A per-bar tuple as long as the values is what makes the renderer take the per-bar path."""
        panel = self._panel(0)
        assert getattr(panel, "colors", None) is not None and len(panel.colors) == len(panel.values)

    def test_both_signs_are_present(self):
        """A fixture whose levels straddle the base rate must produce two distinct colours."""
        panel = self._panel(1)
        assert len(set(panel.colors)) == 2, panel.colors

    def test_the_colours_follow_the_sign_of_each_bar(self):
        """Not merely "two colours": each bar's colour has to match its own direction."""
        panel = self._panel(2)
        pos = {c for c, v in zip(panel.colors, panel.values) if v >= 0}
        neg = {c for c, v in zip(panel.colors, panel.values) if v < 0}
        assert pos and neg and pos.isdisjoint(neg)

    def test_the_renderer_would_paint_them_separately(self):
        """End to end: the builder's tuple survives the renderer's colour resolution."""
        panel = self._panel(3)
        assert _bar_colors(panel.colors, panel.values) == list(panel.colors)


class TestTheRendererActuallyPaintsThem:
    """The helper being right is not enough -- the call site has to use it."""

    def _spec(self):
        """A one-panel figure with a signed per-bar colour tuple."""
        from mlframe.reporting.spec import FigureSpec

        panel = BarPanelSpec(
            categories=("a", "b", "c", "d"),
            values=VALUES,
            colors=SIGNED,
            orientation="horizontal",
            title="signed",
        )
        return FigureSpec(suptitle="t", panels=((panel,),))

    def test_matplotlib_paints_each_bar_its_own_colour(self):
        """Reading `colors[0]` painted all four bars the first colour."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.colors import to_hex

        from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer

        fig = MatplotlibRenderer().render(self._spec())
        try:
            bars = [p for ax in fig.axes for p in ax.patches][:4]
            painted = [to_hex(p.get_facecolor()) for p in bars]
            assert len(set(painted)) == 2, f"every bar came out the same colour: {painted}"
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)

    def test_plotly_passes_the_whole_colour_array(self):
        """plotly's marker.color accepts an array; the call site sent a single string."""
        pytest.importorskip("plotly")
        from mlframe.reporting.renderers.plotly import PlotlyRenderer

        fig = PlotlyRenderer().render(self._spec())
        colors = [tr.marker.color for tr in fig.data if getattr(tr, "type", "") == "bar"]
        assert colors and not isinstance(colors[0], str), f"a single colour was sent for four bars: {colors}"
