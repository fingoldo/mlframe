"""Renderer edge inputs: list-valued per-point colours and sizes, NaN edge weights, NaN scatter values and an empty network."""

from __future__ import annotations

import numpy as np

from mlframe.reporting.renderers import _plotly_scatter
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import FigureSpec, NetworkPanelSpec, ScatterPanelSpec


def _scatter(**kw) -> FigureSpec:
    """A one-panel scatter spec."""
    kw.setdefault("y", np.asarray(kw["x"], dtype=float))
    return FigureSpec(panels=((ScatterPanelSpec(title="t", xlabel="x", ylabel="y", **kw),),), figsize=(6.0, 4.0))


def test_list_point_colours_follow_the_downsample():
    """With more points than the cap, a list of per-point colours is reordered with x and y, so each kept point keeps its own colour."""
    n = _plotly_scatter._SCATTER_MAX_POINTS + 500
    x = np.arange(n, dtype=float)
    colours = ["red" if i % 2 == 0 else "blue" for i in range(n)]
    fig = PlotlyRenderer().render(_scatter(x=x, y=x, point_color=colours))
    trace = fig.data[0]
    kept_x = np.asarray(trace.x)
    got = list(trace.marker.color)
    assert len(got) == len(kept_x) < n
    assert all(c == ("red" if int(v) % 2 == 0 else "blue") for v, c in zip(kept_x, got))


def test_list_point_size_renders():
    """A per-point size list no longer raises."""
    fig = PlotlyRenderer().render(_scatter(x=np.arange(3.0), point_size=[10, 20, 30]))
    assert len(fig.data[0].marker.size) == 3


def test_nan_does_not_erase_the_perfect_fit_line():
    """One NaN y leaves the diagonal drawn between the finite extremes."""
    fig = PlotlyRenderer().render(_scatter(x=np.array([0.1, 0.4, 0.5]), y=np.array([0.1, np.nan, 0.5]), perfect_fit_line=True))
    line = next(t for t in fig.data if t.name == "Perfect fit")
    assert np.all(np.isfinite(np.asarray(line.x, dtype=float)))


def _network(weights, n_nodes=4) -> FigureSpec:
    """A small network spec with the given edge weights."""
    k = len(weights)
    return FigureSpec(
        panels=((NetworkPanelSpec(
            node_x=np.arange(n_nodes, dtype=float), node_y=np.arange(n_nodes, dtype=float), node_size=np.full(n_nodes, 100.0),
            node_color=tuple(["#2ca02c"] * n_nodes), node_label=tuple(f"n{i}" for i in range(n_nodes)),
            edge_src=np.zeros(k, dtype=np.int64), edge_dst=np.arange(1, k + 1) % n_nodes, edge_weight=np.asarray(weights, dtype=float), title="g"),),),
        figsize=(6.0, 4.0),
    )


def test_nan_edge_weight_keeps_every_edge_drawn_on_plotly():
    """A NaN weight used to send every edge to an invalid bin, so no edge line was drawn at all."""
    fig = PlotlyRenderer().render(_network([0.2, np.nan, 0.8]))
    drawn = sum(len(t.x) for t in fig.data if t.mode == "lines" and t.x is not None)
    assert drawn > 0


def test_nan_edge_weight_gives_finite_widths_on_matplotlib():
    """Matplotlib line widths stay finite for NaN weights."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig = MatplotlibRenderer().render(_network([0.2, np.nan, 0.8]))
    try:
        lcs = [c for c in fig.axes[0].collections if isinstance(c, LineCollection)]
        assert lcs and np.all(np.isfinite(np.asarray(lcs[0].get_linewidths())))
    finally:
        plt.close(fig)


def test_empty_network_renders_on_matplotlib():
    """A network with no nodes draws an empty panel instead of raising on the empty extent reduction."""
    import matplotlib.pyplot as plt

    spec = FigureSpec(
        panels=((NetworkPanelSpec(
            node_x=np.array([]), node_y=np.array([]), node_size=np.array([]), node_color=(), node_label=(),
            edge_src=np.array([], dtype=np.int64), edge_dst=np.array([], dtype=np.int64), edge_weight=np.array([]), title="empty"),),),
        figsize=(6.0, 4.0),
    )
    fig = MatplotlibRenderer().render(spec)
    plt.close(fig)
