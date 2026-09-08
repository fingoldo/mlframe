"""Capping the NUMBER of labels is not the same as making them readable.

Both renderers already limited how many point labels they drew, and both still printed those labels on top
of each other whenever the points they name are close together: a spectral embedding whose nodes collapse
into one region, and the low-probability corner of a reliability diagram. Nothing compared two labels to
each other. The halo makes the result read as a smudge rather than a clip, which is why it looked minor and
was in fact unreadable.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlframe.reporting.charts.spectral_embedding import compose_spectral_embedding_figure, spectral_embedding_panel
from mlframe.reporting.renderers._shared_helpers import non_colliding_label_indices
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import FigureSpec, ScatterPanelSpec

N_NODES = 200


def _graph():
    """A random graph dense enough that its spectral layout collapses into one blob."""
    rng = np.random.default_rng(0)
    return np.column_stack([rng.integers(0, N_NODES, 1200), rng.integers(0, N_NODES, 1200)])


def _crowded_reliability() -> FigureSpec:
    """Twelve bins crammed below 0.1 plus eight spread out, which is what a real reliability diagram looks like."""
    xs = np.concatenate([np.linspace(0.01, 0.09, 12), np.linspace(0.3, 0.95, 8)])
    panel = ScatterPanelSpec(
        x=xs, y=xs, title="reliability", xlabel="predicted", ylabel="observed",
        inline_labels=tuple((float(v), float(v), f"{v:.3f}") for v in xs),
    )
    return FigureSpec(panels=((panel,),), figsize=(6.0, 4.5))


def test_the_picker_keeps_every_label_when_they_all_fit():
    """Guard: de-collision must not thin an axis whose labels were never in each other's way."""
    xs = np.linspace(0.0, 1.0, 10)
    kept = non_colliding_label_indices(xs, xs, [str(i) for i in range(10)], fontsize=8, x_span=1.0, y_span=1.0, width_in=6.0, height_in=4.0)
    assert kept == list(range(10)), f"{10 - len(kept)} labels were dropped from a panel that had room for all ten"


def test_the_picker_drops_labels_that_land_on_each_other():
    """Forty labels inside one hundredth of the panel cannot all be drawn."""
    rng = np.random.default_rng(0)
    pts = rng.normal(0.0, 0.01, 40)
    kept = non_colliding_label_indices(pts, pts, [f"node_{i}" for i in range(40)], fontsize=8, x_span=1.0, y_span=1.0, width_in=6.0, height_in=4.0)
    assert 0 < len(kept) < 40, f"{len(kept)} of 40 labels kept in a cluster one hundredth of the panel wide"


def test_a_collapsed_graph_layout_does_not_print_a_smudge():
    """The rendered defect: twenty-five node names stacked into an illegible black mark."""
    fig = MatplotlibRenderer().render(compose_spectral_embedding_figure(N_NODES, _graph()))
    try:
        ax = next(a for a in fig.axes if a.collections)
        drawn = [(t.get_position(), t.get_text()) for t in ax.texts if t.get_text()]
        assert drawn, "no node was named at all"
        xs = [pos[0] for pos, _ in drawn]
        ys = [pos[1] for pos, _ in drawn]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-12)
        closest = min(abs(xs[i] - xs[j]) + abs(ys[i] - ys[j]) for i in range(len(drawn)) for j in range(i + 1, len(drawn))) if len(drawn) > 1 else span
        assert closest > span / 50.0, f"{len(drawn)} labels drawn with two of them {closest:.2e} apart across a {span:.2e} panel"
    finally:
        plt.close(fig)


def test_nodes_are_sized_by_degree_so_the_named_ones_are_the_hubs():
    """With every node the same size the label cap picked an arbitrary handful of indices to name."""
    edges = _graph()
    panel = spectral_embedding_panel(N_NODES, edges)
    sizes = np.asarray(panel.node_size, dtype=float)
    assert sizes.max() > sizes.min(), "every node is still the same size, so the label pick is arbitrary"
    deg = np.bincount(edges.ravel(), minlength=N_NODES)
    assert np.corrcoef(sizes, deg)[0, 1] > 0.99, "node size does not track degree"


def test_a_constant_edge_weight_draws_no_colour_scale():
    """A scale with one value on it encodes nothing, and matplotlib renders it with a bogus offset exponent."""
    fig = MatplotlibRenderer().render(compose_spectral_embedding_figure(N_NODES, _graph()))
    try:
        assert len([a for a in fig.axes if a.get_label() == "<colorbar>"]) == 0, "a colorbar was drawn over constant edge weights"
    finally:
        plt.close(fig)

    pfig = PlotlyRenderer().render(compose_spectral_embedding_figure(N_NODES, _graph()))
    flags = [t.marker.showscale for t in pfig.data if getattr(getattr(t, "marker", None), "showscale", None) is not None]
    assert not any(flags), f"plotly still shows the degenerate scale: {flags}"


@pytest.mark.parametrize("backend", ["matplotlib", "plotly"])
def test_crowded_inline_labels_are_thinned_on_both_backends(backend):
    """Twelve bins below 0.1 overprinted into one illegible block on both backends."""
    spec = _crowded_reliability()
    total = len(spec.panels[0][0].inline_labels)
    if backend == "matplotlib":
        fig = MatplotlibRenderer().render(spec)
        try:
            drawn = len([t for t in fig.axes[0].texts if t.get_text()])
        finally:
            plt.close(fig)
    else:
        fig = PlotlyRenderer().render(spec)
        drawn = len([a for a in fig.layout.annotations if a.text and a.text[0].isdigit()])
    assert 0 < drawn < total, f"{backend} drew {drawn} of {total} labels; the crowded bins are still on top of each other"


def test_the_spread_out_bins_keep_their_labels():
    """Guard: thinning must cost the crowded corner, not the readable half of the same chart."""
    spec = _crowded_reliability()
    fig = MatplotlibRenderer().render(spec)
    try:
        drawn = {t.get_text() for t in fig.axes[0].texts if t.get_text()}
    finally:
        plt.close(fig)
    assert "0.950" in drawn and "0.764" in drawn, f"a well-separated bin lost its label: {sorted(drawn)}"
