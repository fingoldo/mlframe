"""RUX-67: both backends must draw the confusion margins as real bar axes, not one of them as tick-label text.

matplotlib built true marginal axes via a subgridspec while plotly folded the same numbers into the tick label
strings, so one spec produced two visibly different figures -- and only one of them let a reader compare two class
supports by BAR LENGTH, which is the whole reason the margins are on the figure.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from mlframe.reporting.renderers import get_renderer
from mlframe.reporting.spec import ConfusionMarginsPanelSpec, FigureSpec

LABELS = ("a", "b", "c")
MATRIX = np.array([[40.0, 5.0, 1.0], [3.0, 22.0, 6.0], [0.0, 4.0, 17.0]])


def _panel(**kw) -> ConfusionMarginsPanelSpec:
    """The panel under test, with the margins the builders actually pass."""
    base = dict(
        matrix=MATRIX,
        row_labels=LABELS,
        col_labels=LABELS,
        row_margin=MATRIX.sum(axis=1),
        col_margin=MATRIX.sum(axis=0),
        row_margin_label="support",
        col_margin_label="volume",
        title="confusion",
    )
    base.update(kw)
    return ConfusionMarginsPanelSpec(**base)


def _plotly_fig(panel):
    """Render one panel through the real subplot path (a bare figure has no cell domains to split)."""
    return get_renderer("plotly").render(FigureSpec(panels=((panel,),), figsize=(8.0, 6.0)))


class TestPlotlyDrawsRealMarginalAxes:
    """The margins must be drawable geometry, not text baked into a tick label."""

    def test_two_bar_traces_carry_the_margins(self):
        """One horizontal trace for row support, one vertical for column volume."""
        fig = _plotly_fig(_panel())
        bars = [t for t in fig.data if t.type == "bar"]
        assert len(bars) == 2
        horizontal = [t for t in bars if t.orientation == "h"]
        vertical = [t for t in bars if t.orientation != "h"]
        assert len(horizontal) == 1 and len(vertical) == 1
        assert np.array_equal(np.asarray(horizontal[0].x), MATRIX.sum(axis=1))
        assert np.array_equal(np.asarray(vertical[0].y), MATRIX.sum(axis=0))

    def test_the_margin_values_are_no_longer_hidden_in_tick_labels(self):
        """The old rendering spelled the numbers into the category strings, where nothing can compare them."""
        fig = _plotly_fig(_panel())
        heat = next(t for t in fig.data if t.type == "heatmap")
        assert tuple(heat.x) == LABELS
        assert tuple(heat.y) == LABELS

    def test_the_strips_share_the_heatmap_axes_so_bars_stay_aligned(self):
        """A bar that does not line up with its row measures nothing; the strips MATCH the heatmap's categoricals."""
        fig = _plotly_fig(_panel())
        matched = {fig.layout[k].matches for k in fig.layout if isinstance(k, str) and k.startswith(("xaxis", "yaxis")) and fig.layout[k].matches}
        assert matched == {"x", "y"}

    def test_the_strips_do_not_overlap_the_grid(self):
        """The heatmap shrinks to make room; overlapping domains would draw the bars on top of the cells."""
        fig = _plotly_fig(_panel())
        heat_x = tuple(fig.layout["xaxis"].domain)
        heat_y = tuple(fig.layout["yaxis"].domain)
        strips = [
            (tuple(fig.layout[k].domain), k) for k in fig.layout if isinstance(k, str) and k.startswith(("xaxis", "yaxis")) and k not in ("xaxis", "yaxis")
        ]
        assert strips, "no nested strip axes were created"
        # Four nested axes exist (each strip needs an x AND a y); the two that carry the strips are the ones whose
        # domain starts past the shrunk grid. The other two mirror the grid's own domain to stay aligned with it.
        right = [d for d, k in strips if k.startswith("xaxis") and d[0] > heat_x[0]]
        top = [d for d, k in strips if k.startswith("yaxis") and d[0] > heat_y[0]]
        assert len(right) == 1 and len(top) == 1
        assert right[0][0] >= heat_x[1], "the support strip overlaps the grid"
        assert top[0][0] >= heat_y[1], "the volume strip overlaps the grid"

    def test_matplotlib_still_draws_its_own_marginal_axes(self):
        """The parity claim needs both halves: matplotlib's subgridspec axes must survive the plotly change."""
        fig = get_renderer("matplotlib").render(FigureSpec(panels=((_panel(),),), figsize=(8.0, 6.0)))
        assert len(fig.get_axes()) >= 3  # grid + two margins


class TestDegenerateInputStillRenders:
    """A zero matrix and an empty one must render on both backends rather than raising in the new geometry code."""

    @pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
    def test_all_zero_margins(self, backend):
        """Zero-length bars are a legitimate drawing; the strips must not divide by a zero span."""
        panel = _panel(matrix=np.zeros((3, 3)), row_margin=np.zeros(3), col_margin=np.zeros(3))
        get_renderer(backend).render(FigureSpec(panels=((panel,),), figsize=(6.0, 5.0)))
