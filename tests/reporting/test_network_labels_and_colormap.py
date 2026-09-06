"""A network panel must render on its own defaults, and must not try to name every node.

Two defects, both found by rendering a friend-graph-shaped panel rather than by reading the spec.

``NetworkPanelSpec.colormap`` defaults to the HEATMAP_GENERIC sentinel -- a token, not a colormap name --
and the matplotlib branch subscripted it directly, so any network panel that did not name a colormap raised
``KeyError('__mlframe_default_heatmap__')``. The plotly branch took the unknown-name path and fell back to
Viridis with a warning, which was right by accident.

Labels were drawn for every node, at full length, centred on the marker. Edges are capped; the labels never
were, so a friend graph at its default 200 nodes prints 200 names of ~35 characters over each other.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.renderers._shared_helpers import _NETWORK_MAX_LABELS, network_label_indices
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import FigureSpec, NetworkPanelSpec

N_NODES = 120


def _graph(n: int = N_NODES, **overrides) -> FigureSpec:
    """A friend-graph-shaped panel: many nodes, long generated names, no colormap named."""
    rng = np.random.default_rng(0)
    angle = rng.uniform(0, 2 * np.pi, n)
    radius = rng.uniform(0.2, 1.0, n)
    kwargs = dict(
        node_x=radius * np.cos(angle),
        node_y=radius * np.sin(angle),
        node_size=rng.uniform(40, 300, n),
        node_color=tuple(["#2ca02c"] * n),
        node_label=tuple(f"engineered_feature_{i}_ratio_log_v2" for i in range(n)),
        edge_src=rng.integers(0, n, 180),
        edge_dst=rng.integers(0, n, 180),
        edge_weight=rng.uniform(0, 1, 180),
        title="Feature friend graph",
    )
    kwargs.update(overrides)
    return FigureSpec(panels=((NetworkPanelSpec(**kwargs),),), figsize=(11.0, 8.0))


def test_matplotlib_renders_a_network_on_the_default_colormap():
    """The default is a sentinel; reaching matplotlib with it raised KeyError on every such panel."""
    fig = MatplotlibRenderer().render(_graph())
    try:
        assert fig.axes, "no axes were produced"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_plotly_renders_a_network_on_the_default_colormap():
    """Its fallback made it survive, but through the unknown-name path rather than the resolver."""
    fig = PlotlyRenderer().render(_graph())
    assert fig.data, "no traces were produced"


def test_only_the_biggest_nodes_are_named():
    """The helper's contract: cap the count, and spend it on the nodes the graph is drawn to show."""
    rng = np.random.default_rng(1)
    sizes = rng.uniform(1, 100, N_NODES)
    keep = network_label_indices(sizes)
    assert len(keep) == _NETWORK_MAX_LABELS, f"kept {len(keep)} labels, expected the cap {_NETWORK_MAX_LABELS}"
    kept_min = sizes[keep].min()
    dropped = np.setdiff1d(np.arange(N_NODES), keep)
    assert sizes[dropped].max() <= kept_min, "a dropped node is bigger than a kept one; the selection is not by size"
    assert keep == sorted(keep), "indices must stay in drawing order"


def test_a_small_graph_still_names_everything():
    """The cap must not cost a readable graph its labels."""
    assert network_label_indices(np.arange(5.0)) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_a_large_graph_prints_at_most_the_cap(backend):
    """End to end: the drawn text, not the helper."""
    spec = _graph()
    if backend == "matplotlib":
        fig = MatplotlibRenderer().render(spec)
        try:
            drawn = [t.get_text() for t in fig.axes[0].texts if t.get_text()]
        finally:
            import matplotlib.pyplot as plt

            plt.close(fig)
    else:
        fig = PlotlyRenderer().render(spec)
        node_traces = [tr for tr in fig.data if getattr(tr, "text", None) is not None]
        assert node_traces, "no trace carries node text"
        drawn = [t for t in node_traces[0].text if t]

    assert len(drawn) <= _NETWORK_MAX_LABELS, f"{backend} printed {len(drawn)} labels for {N_NODES} nodes"
    assert drawn, f"{backend} printed no labels at all"
    longest = max(len(d) for d in drawn)
    assert longest <= 26, f"{backend} drew a {longest}-char label; generated names must be truncated"
