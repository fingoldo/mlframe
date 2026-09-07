"""Bands and change points were drawn one plotly API call at a time.

Each of ``add_vrect`` / ``add_trace`` / ``add_annotation`` re-validates its whole growing collection on
every call, so a per-item loop is super-quadratic across the three together. The audit filed this as a
latent shape defect because today's callers pass one or two bands -- but a regime chart is exactly the
thing that grows with the data, and measured on this panel: 2 spans 82ms, 20 spans 737ms, 100 spans 14.3s,
300 spans 150.8s. Batched, the same figures cost 39ms / 111ms / 207ms / 601ms.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import FigureSpec, LinePanelSpec

X = np.linspace(0.0, 200.0, 400)


def _panel(n_spans: int = 0, n_lines: int = 0) -> FigureSpec:
    """A line panel carrying ``n_spans`` bands and ``n_lines`` change points."""
    spans = tuple((i * 200.0 / n_spans, (i + 0.4) * 200.0 / n_spans, "#888888", 0.15, f"regime_{i}") for i in range(n_spans))
    lines = tuple((i * 200.0 / max(n_lines, 1), "#333333", f"cp_{i}") for i in range(n_lines))
    panel = LinePanelSpec(x=X, y=np.sin(X / 12.0), series_labels=("metric",), title="regimes", vspans=spans, vlines=lines)
    return FigureSpec(panels=((panel,),), figsize=(11.0, 4.0))


def _elapsed(spec) -> float:
    """Best of three renders, which is what a shared machine can defend."""
    return min((lambda t0: (PlotlyRenderer().render(spec), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(3))


@pytest.mark.parametrize("kind", ["spans", "lines"])
def test_the_markers_cost_a_constant_number_of_plotly_calls(kind):
    """The property, counted rather than timed.

    Each per-item ``add_*`` re-validates its whole growing collection, so the defect is the CALL COUNT
    growing with the marker count. Counting it is deterministic; a wall-clock assertion in a suite that
    runs under ``-n 4`` measures core contention as much as the renderer.
    """
    import plotly.graph_objects as go

    calls = {"n": 0}
    real_ann, real_rect = go.Figure.add_annotation, go.Figure.add_vrect

    def counted_ann(self, *a, **kw):
        """Count per-item annotation calls."""
        calls["n"] += 1
        return real_ann(self, *a, **kw)

    def counted_rect(self, *a, **kw):
        """Count per-item vrect calls."""
        calls["n"] += 1
        return real_rect(self, *a, **kw)

    counts = {}
    go.Figure.add_annotation, go.Figure.add_vrect = counted_ann, counted_rect
    try:
        for n in (4, 100):
            calls["n"] = 0
            PlotlyRenderer().render(_panel(**{f"n_{kind}": n}))
            counts[n] = calls["n"]
    finally:
        go.Figure.add_annotation, go.Figure.add_vrect = real_ann, real_rect
    assert counts[100] == counts[4], f"{kind}: 100 markers made {counts[100]} per-item calls against {counts[4]} for 4; the loop is back"


def test_every_band_is_still_drawn():
    """Batching must not lose one: a band that vanishes is a regime the reader never sees."""
    fig = PlotlyRenderer().render(_panel(n_spans=12))
    rects = [s for s in fig.layout.shapes if s.type == "rect"]
    assert len(rects) == 12, f"{len(rects)} of 12 bands were drawn"
    assert len([a for a in fig.layout.annotations if a.text and a.text.startswith("regime_")]) == 12
    assert len([t for t in fig.data if t.name and t.name.startswith("regime_")]) == 12, "a band lost its legend proxy"


def test_every_change_point_is_still_drawn():
    """The same guard for the vline path: a dropped change point is an event the reader never sees."""
    fig = PlotlyRenderer().render(_panel(n_lines=9))
    assert len([s for s in fig.layout.shapes if s.type == "line"]) == 9
    assert len([a for a in fig.layout.annotations if a.text and a.text.startswith("cp_")]) == 9


def test_the_band_shape_is_what_add_vrect_produced():
    """The batched shapes are built by hand, so their fields have to match the API call they replace."""
    import plotly.graph_objects as go

    fig = PlotlyRenderer().render(_panel(n_spans=1))
    ours = next(s for s in fig.layout.shapes if s.type == "rect").to_plotly_json()

    reference = go.Figure(go.Scatter(x=X, y=X))
    reference.add_vrect(x0=ours["x0"], x1=ours["x1"], fillcolor=ours["fillcolor"], line_width=0, layer="below")
    theirs = reference.layout.shapes[0].to_plotly_json()
    assert ours == theirs, f"the hand-built shape differs from add_vrect's: {ours} vs {theirs}"


@pytest.mark.parametrize("kind", ["spans", "lines"])
def test_the_labels_hang_inside_the_plot_area(kind):
    """Stacked upward they land in the subplot title's strip -- seen in the render, "recovery" across the title."""
    fig = PlotlyRenderer().render(_panel(**{f"n_{kind}": 3}))
    prefix = "regime_" if kind == "spans" else "cp_"
    labels = [a for a in fig.layout.annotations if a.text and a.text.startswith(prefix)]
    assert labels, "no marker label was drawn"
    for ann in labels:
        assert ann.yanchor == "top" and float(ann.y) == 1.0, f"{ann.text!r} is anchored above the panel, where the title is"
        assert int(ann.yshift or 0) <= 0, f"{ann.text!r} shifts upward ({ann.yshift}) out of the plot area"
