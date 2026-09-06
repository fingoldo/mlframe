"""Density-heatmap axes must thin their per-bin ticks (80 labels -> readable few), and a FigureSpec caption must
render as a bottom footnote (not bloat a panel title). Regression sensors for the chart-readability fixes."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.spec import FigureSpec, HeatmapPanelSpec, LinePanelSpec


def _dense_heatmap_spec(nbins: int = 80) -> FigureSpec:
    """Helper: Dense heatmap spec."""
    mat = np.random.default_rng(0).random((nbins, nbins))
    labels = tuple(f"{v:.3g}" for v in np.linspace(8770, 14300, nbins))
    panel = HeatmapPanelSpec(
        matrix=mat,
        row_labels=labels,
        col_labels=labels,
        title="Predictions vs true (log-density)",
        xlabel="Predictions",
        ylabel="True values",
    )
    return FigureSpec(panels=((panel,),), figsize=(7.0, 5.0))


def test_thin_tick_positions_caps_and_keeps_endpoints():
    """Thin tick positions caps and keeps endpoints."""
    from mlframe.reporting.renderers.matplotlib import _thin_tick_positions, _HEATMAP_MAX_TICKS

    pos = _thin_tick_positions(80)
    assert len(pos) <= _HEATMAP_MAX_TICKS
    assert pos[0] == 0 and pos[-1] == 79
    # Small axes are untouched (one tick per label).
    assert _thin_tick_positions(5) == [0, 1, 2, 3, 4]


def test_matplotlib_heatmap_thins_dense_ticks():
    """Every kept label has room for itself.

    The budget used to be a flat eight ticks per axis. That wasted a drift heatmap grown to fourteen inches,
    so it became width-aware -- and then overshot in the other direction, because a -45-degree label needs
    MORE room along the axis than a stacked one, not the same. The contract this pins is the readable one:
    the spacing between kept ticks clears a line height measured perpendicular to the labels themselves.
    """
    from mlframe.reporting.renderers._shared_helpers import rotated_tick_pitch_in
    from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer, _HEATMAP_TICK_FONTSIZE

    fig = MatplotlibRenderer().render(_dense_heatmap_spec(80))
    ax = fig.axes[0]
    pos, (fig_w, fig_h) = ax.get_position(), fig.get_size_inches()
    for axis, extent_in, rotation in ((ax.get_xticks(), pos.width * fig_w, 45), (ax.get_yticks(), pos.height * fig_h, 0)):
        assert len(axis) < 80, "no thinning happened at all: one tick per bin is unreadable soup"
        pitch = extent_in / max(len(axis) - 1, 1)
        needed = rotated_tick_pitch_in(_HEATMAP_TICK_FONTSIZE, rotation)
        assert pitch >= needed, f"{len(axis)} labels over {extent_in:.2f}in is {pitch:.3f}in apart, under the {needed:.3f}in they need"


def test_plotly_heatmap_thins_dense_ticks():
    """Plotly heatmap thins dense ticks."""
    pytest.importorskip("plotly")
    from mlframe.reporting.renderers.plotly import PlotlyRenderer

    from mlframe.reporting.renderers._plotly_heatmap import _COLORBAR_ALLOWANCE_PX
    from mlframe.reporting.renderers._shared_helpers import PX_PER_INCH, rotated_tick_pitch_in

    fig = PlotlyRenderer().render(_dense_heatmap_spec(80))
    # The plot region, not the figure: a domain is a fraction of what is left after the margins, and the
    # colorbar sits inside it. Budgeting against the raw figure width is what kept 26 labels in a 22-label
    # axis on this backend.
    m = fig.layout.margin
    plot_w_in = (float(fig.layout.width) - float(m.l or 0) - float(m.r or 0) - _COLORBAR_ALLOWANCE_PX) / PX_PER_INCH
    plot_h_in = (float(fig.layout.height) - float(m.t or 0) - float(m.b or 0)) / PX_PER_INCH
    for tickvals, extent_in, rotation in ((fig.layout.xaxis.tickvals, plot_w_in, 45), (fig.layout.yaxis.tickvals, plot_h_in, 0)):
        assert tickvals is not None and len(tickvals) < 80, "no thinning happened at all: one tick per bin is unreadable soup"
        pitch = extent_in / max(len(tickvals) - 1, 1)
        needed = rotated_tick_pitch_in(8, rotation)
        assert pitch >= needed, f"{len(tickvals)} labels over {extent_in:.2f}in is {pitch:.3f}in apart, under the {needed:.3f}in they need"


def test_caption_renders_as_bottom_footnote_matplotlib():
    """Caption renders as bottom footnote matplotlib."""
    line = LinePanelSpec(x=np.linspace(0, 1, 10), y=(np.linspace(0, 1, 10),), series_labels=("a",), title="t")
    cap = "Sort predictions by confidence; x = coverage, y = error on kept rows. Flat => no signal."
    fig = FigureSpec(panels=((line,),), figsize=(8.0, 5.8), caption=cap)
    from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer

    rendered = MatplotlibRenderer().render(fig)
    texts = [t.get_text() for t in rendered.texts]
    assert any("coverage" in t for t in texts), "caption footnote should be drawn as a figure text"
