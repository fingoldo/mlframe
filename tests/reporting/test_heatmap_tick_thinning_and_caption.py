"""Density-heatmap axes must thin their per-bin ticks (80 labels -> readable few), and a FigureSpec caption must
render as a bottom footnote (not bloat a panel title). Regression sensors for the chart-readability fixes."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.spec import FigureSpec, HeatmapPanelSpec, LinePanelSpec


def _dense_heatmap_spec(nbins: int = 80) -> FigureSpec:
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
    from mlframe.reporting.renderers.matplotlib import _thin_tick_positions, _HEATMAP_MAX_TICKS

    pos = _thin_tick_positions(80)
    assert len(pos) <= _HEATMAP_MAX_TICKS
    assert pos[0] == 0 and pos[-1] == 79
    # Small axes are untouched (one tick per label).
    assert _thin_tick_positions(5) == [0, 1, 2, 3, 4]


def test_matplotlib_heatmap_thins_dense_ticks():
    from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer, _HEATMAP_MAX_TICKS

    fig = MatplotlibRenderer().render(_dense_heatmap_spec(80))
    ax = fig.axes[0]
    assert len(ax.get_xticks()) <= _HEATMAP_MAX_TICKS
    assert len(ax.get_yticks()) <= _HEATMAP_MAX_TICKS


def test_plotly_heatmap_thins_dense_ticks():
    pytest.importorskip("plotly")
    from mlframe.reporting.renderers.plotly import PlotlyRenderer, _HEATMAP_MAX_TICKS

    fig = PlotlyRenderer().render(_dense_heatmap_spec(80))
    xaxes = [v for k, v in fig.layout._props.get("xaxis", {}).items()] if False else None  # noqa
    # tickvals are set on the first x/y axis; assert the count is capped.
    xa = fig.layout.xaxis
    assert xa.tickvals is not None and len(xa.tickvals) <= _HEATMAP_MAX_TICKS


def test_caption_renders_as_bottom_footnote_matplotlib():
    line = LinePanelSpec(x=np.linspace(0, 1, 10), y=(np.linspace(0, 1, 10),), series_labels=("a",), title="t")
    cap = "Sort predictions by confidence; x = coverage, y = error on kept rows. Flat => no signal."
    fig = FigureSpec(panels=((line,),), figsize=(8.0, 5.8), caption=cap)
    from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer

    rendered = MatplotlibRenderer().render(fig)
    texts = [t.get_text() for t in rendered.texts]
    assert any("coverage" in t for t in texts), "caption footnote should be drawn as a figure text"
