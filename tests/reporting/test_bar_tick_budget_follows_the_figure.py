"""Bar tick thinning was count-based while the figure size is also count-based, so the two cancelled.

`slice_finder` and `category_discriminability` both grow the figure half an inch per bar precisely so every
bar can be named. The renderers then dropped to ~20 labels past 25 categories regardless of size, leaving a
40-bar chart with twenty bars whose identity is unrecoverable -- the builder bought the height and the
renderer threw it away.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import BarPanelSpec, FigureSpec

TALL_K = 40


def _spec(k: int, figsize, orientation: str = "horizontal") -> FigureSpec:
    """A k-category bar panel at the requested figure size."""
    panel = BarPanelSpec(
        categories=tuple(f"slice_{i}" for i in range(k)), values=np.linspace(0.1, 0.9, k),
        title="t", orientation=orientation, xlabel="lift", ylabel="slice",
    )
    return FigureSpec(panels=((panel,),), figsize=figsize)


def _mpl_labels(spec, horizontal: bool = True):
    """Category-axis tick labels matplotlib actually drew."""
    fig = MatplotlibRenderer().render(spec)
    try:
        ticks = fig.axes[0].get_yticklabels() if horizontal else fig.axes[0].get_xticklabels()
        return [t.get_text() for t in ticks if t.get_text()]
    finally:
        plt.close(fig)


def _plotly_labels(spec, horizontal: bool = True):
    """Category-axis tick labels plotly actually drew."""
    fig = PlotlyRenderer().render(spec)
    axis = fig.layout.yaxis if horizontal else fig.layout.xaxis
    return list(axis.ticktext or ())


@pytest.mark.parametrize("reader", [_mpl_labels, _plotly_labels], ids=["matplotlib", "plotly"])
def test_a_figure_grown_for_its_bars_names_all_of_them(reader):
    """The builder's own sizing rule: 0.5in per bar, which is ten times what an 8pt label needs."""
    spec = _spec(TALL_K, (10.0, max(5.0, 0.5 * TALL_K + 2.0)))
    assert len(reader(spec)) == TALL_K, f"only {len(reader(spec))} of {TALL_K} bars are named on a figure sized to name every one"


@pytest.mark.parametrize("reader", [_mpl_labels, _plotly_labels], ids=["matplotlib", "plotly"])
def test_a_cramped_figure_still_thins(reader):
    """Guard: the fix must follow the axis in BOTH directions, not just stop thinning."""
    drawn = reader(_spec(60, (10.0, 4.0)))
    assert 0 < len(drawn) < 60, f"{len(drawn)} of 60 labels on a 4-inch axis is a smear, not an axis"


def test_upright_labels_are_rotated_only_when_they_do_not_fit():
    """Rotating buys labels, so it is worth doing -- but not on an axis that had room upright."""
    fig = MatplotlibRenderer().render(_spec(5, (10.0, 4.0), orientation="vertical"))
    try:
        assert {t.get_rotation() for t in fig.axes[0].get_xticklabels() if t.get_text()} == {0.0}, "a five-category axis was rotated for no reason"
    finally:
        plt.close(fig)

    fig = MatplotlibRenderer().render(_spec(60, (10.0, 4.0), orientation="vertical"))
    try:
        drawn = [t for t in fig.axes[0].get_xticklabels() if t.get_text()]
        assert {t.get_rotation() for t in drawn} == {45.0}, "sixty upright labels were left to overlap rather than rotated"
    finally:
        plt.close(fig)


def test_the_two_backends_thin_a_cramped_axis_to_the_same_count():
    """Same spec, same axis length, same pitch: the counts should not diverge on a shared layout."""
    spec = _spec(60, (10.0, 4.0), orientation="vertical")
    assert len(_mpl_labels(spec, horizontal=False)) == len(_plotly_labels(spec, horizontal=False))
