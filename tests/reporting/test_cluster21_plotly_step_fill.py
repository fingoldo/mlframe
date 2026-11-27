"""Cluster 21 regression: plotly steps the fill (not the line) when ``step_fill`` is set, as matplotlib does (audit 2026-11-20, 21-F8)."""

import numpy as np

from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import FigureSpec, LinePanelSpec


def _fig(step_fill, baseline=0.0):
    """Render one filled line panel."""
    panel = LinePanelSpec(x=np.arange(8.0), y=np.array([0.0, 1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 7.0]), fill_to_baseline=True, fill_baseline=baseline, step_fill=step_fill)
    return PlotlyRenderer().render(FigureSpec(panels=((panel,),), suptitle="t"))


def test_step_fill_adds_a_stepped_fill_trace_and_keeps_the_line_straight():
    """With step_fill the fill lives on an hv-shaped trace; the visible series line stays straight and unfilled (21-F8)."""
    fig = _fig(True)
    stepped = [t for t in fig.data if t.line.shape == "hv"]
    assert len(stepped) == 1 and stepped[0].fill in ("tozeroy", "tonexty")
    lines = [t for t in fig.data if t.showlegend is not False or t.line.width != 0]
    assert all(t.line.shape != "hv" for t in lines) and all(t.fill is None for t in lines if t.line.width != 0)


def test_without_step_fill_the_fill_stays_on_the_series_trace():
    """No step_fill: no hv trace, and the series trace carries the fill as before (21-F8)."""
    fig = _fig(False)
    assert not [t for t in fig.data if t.line.shape == "hv"]
    assert any(t.fill == "tozeroy" for t in fig.data)


def test_nonzero_baseline_steps_between_the_baseline_trace_and_the_curve():
    """A non-zero baseline keeps its constant baseline trace and fills the stepped trace to it with tonexty (21-F8)."""
    fig = _fig(True, baseline=1.5)
    stepped = [t for t in fig.data if t.line.shape == "hv"]
    assert len(stepped) == 1 and stepped[0].fill == "tonexty"
    idx = list(fig.data).index(stepped[0])
    assert np.allclose(np.asarray(fig.data[idx - 1].y, dtype=float), 1.5)
