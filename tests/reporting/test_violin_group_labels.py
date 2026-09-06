"""Violin group labels got neither of the two guards the bar branches have.

A per-class probability violin on a 20-class problem with string class names put 20 labels of ~30 characters
under the axis at 30 degrees. They overlap into a staircase and take roughly 40% of the figure height, on
both backends. Bars truncate and thin their category labels; violins did neither.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import FigureSpec, ViolinPanelSpec

K = 20
STEM = "electronics_accessories_tier"
MAXLEN = 20


def _violin(k: int = K) -> FigureSpec:
    """A per-class violin with realistic long class names."""
    rng = np.random.default_rng(0)
    labels = tuple(f"{STEM}_{i}" for i in range(k))
    groups = tuple(rng.normal(i * 0.1, 1.0, 200) for i in range(k))
    panel = ViolinPanelSpec(groups=groups, group_labels=labels, title="Per-class score distribution", xlabel="class", ylabel="P(y=k|x)")
    return FigureSpec(panels=((panel,),), figsize=(9.0, 5.0))


def _matplotlib_labels(spec):
    """Tick labels actually drawn by matplotlib."""
    fig = MatplotlibRenderer().render(spec)
    try:
        return [t.get_text() for t in fig.axes[0].get_xticklabels() if t.get_text()]
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def _plotly_labels(spec):
    """Tick labels actually drawn: the axis subsamples the trace names, as matplotlib subsamples its ticks."""
    fig = PlotlyRenderer().render(spec)
    ticks = fig.layout.xaxis.ticktext
    if ticks:
        return list(ticks)
    return [tr.name for tr in fig.data if getattr(tr, "name", None)]


def test_the_fixture_needs_truncating():
    """Guard: with short names this test proves nothing."""
    assert len(f"{STEM}_0") > MAXLEN, "class names no longer exceed the cap"


@pytest.mark.parametrize("reader", [_matplotlib_labels, _plotly_labels], ids=["matplotlib", "plotly"])
def test_labels_are_truncated(reader):
    """Both backends, same cap: a label longer than this overlaps its neighbour at 30 degrees."""
    drawn = reader(_violin())
    assert drawn, "no group labels were produced"
    longest = max(len(d) for d in drawn)
    assert longest <= MAXLEN, f"a {longest}-char group label was drawn: {max(drawn, key=len)!r}"


def test_both_backends_draw_the_same_label_text():
    """One spec, one vocabulary.

    The COUNT can differ: each backend thins against the width its own layout engine gives the panel, and
    plotly's violin panel really is wider than matplotlib's. What must not differ is the label TEXT -- the
    truncation form, and which end of the name survives it -- nor which classes anchor the axis.
    """
    spec = _violin()
    mpl, plotly = _matplotlib_labels(spec), _plotly_labels(spec)
    assert set(mpl) <= set(plotly) or set(plotly) <= set(
        mpl
    ), f"the two backends truncate the same names differently: matplotlib {sorted(set(mpl) - set(plotly))}, plotly {sorted(set(plotly) - set(mpl))}"
    assert mpl[0] == plotly[0] and mpl[-1] == plotly[-1], f"the axes start/end on different classes: {mpl[0]}..{mpl[-1]} vs {plotly[0]}..{plotly[-1]}"


def test_the_truncated_labels_stay_distinguishable():
    """`tier_0` vs `tier_19` is the entire difference; a head-preserving cut would erase it."""
    drawn = _matplotlib_labels(_violin())
    assert len(set(drawn)) == len(drawn), f"truncation collapsed distinct classes onto one label: {drawn}"


def test_a_short_named_violin_is_left_alone():
    """The cap must not rewrite labels that already fit."""
    rng = np.random.default_rng(1)
    labels = ("a", "b", "c")
    panel = ViolinPanelSpec(groups=tuple(rng.normal(0, 1, 100) for _ in labels), group_labels=labels, title="t")
    drawn = _matplotlib_labels(FigureSpec(panels=((panel,),), figsize=(6.0, 4.0)))
    assert drawn == list(labels), f"short labels were altered: {drawn}"
