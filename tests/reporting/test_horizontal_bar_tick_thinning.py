"""Horizontal bar labels were thinned on matplotlib and on plotly's VERTICAL branch, but not on plotly's horizontal one.

A 200-row feature-importance / slice-finder / WoE chart therefore had a clean 20-label axis in the PNG and
an unreadable band of overlapping text in the HTML -- from one spec. The bars are one per category either
way; only the labels subsample.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers._shared_helpers import rotated_tick_pitch_in
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import BarPanelSpec, FigureSpec


def _spec(n: int, orientation: str = "horizontal") -> FigureSpec:
    """A bar panel with ``n`` generated feature names."""
    cats = tuple(f"engineered_feature_{i}_ratio_log" for i in range(n))
    panel = BarPanelSpec(categories=cats, values=np.linspace(1.0, 0.0, n), orientation=orientation, title="Feature importance")
    return FigureSpec(panels=((panel,),), figsize=(8.0, 14.0))


def _plotly_tick_count(spec, axis: str = "yaxis"):
    """Number of category ticks plotly will draw, or None when it draws them all."""
    ticks = getattr(PlotlyRenderer().render(spec).layout, axis).tickvals
    return None if ticks is None else len(ticks)


def _mpl_tick_count(spec, horizontal: bool = True) -> int:
    """Number of category ticks matplotlib draws."""
    fig = MatplotlibRenderer().render(spec)
    try:
        ax = fig.axes[0]
        return len(ax.get_yticks() if horizontal else ax.get_xticks())
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)


def test_a_long_horizontal_chart_is_thinned_on_plotly():
    """The defect: 200 labels drawn on top of each other.

    The budget is the axis's own length divided by what one label needs, not a flat cap -- a flat cap
    cancelled out the height the slice-finder builders buy per bar. 200 names never fit in 14 inches.
    """
    spec = _spec(200)
    count = _plotly_tick_count(spec)
    assert count is not None, "plotly still draws every category label on a 200-row chart"
    # plotly adds its margins ON TOP of figsize, so the plot area is the requested height.
    assert count <= spec.figsize[1] / rotated_tick_pitch_in(9, 0), f"plotly drew {count} labels in {spec.figsize[1]:.0f} inches; they cannot all fit"


def test_both_backends_thin_a_200_row_chart_hard():
    """One spec, one reading -- of the POLICY, not of the count.

    The counts differ by design: plotly adds its margins on top of ``figsize`` so its plot area is the full
    requested height, while matplotlib's figsize includes the margins. Each backend thins against the axis
    it actually has, which is the contract; pinning equality would pin one engine's margins into the other.
    """
    spec = _spec(200)
    counts = (_plotly_tick_count(spec), _mpl_tick_count(spec))
    assert all(c is not None and c < 200 for c in counts), f"a 200-row chart was left unthinned somewhere: {counts}"
    assert max(counts) / min(counts) < 1.5, f"the backends disagree by more than their layout difference explains: {counts}"


@pytest.mark.parametrize("n", [5, 12, 25])
def test_a_short_chart_keeps_every_label(n):
    """Thinning must not fire where the labels already fit."""
    count = _plotly_tick_count(_spec(n))
    assert count is None or count == n, f"plotly thinned a {n}-row chart to {count} labels"


def test_the_vertical_branch_is_unchanged():
    """Guard: the vertical orientation already thinned, and must keep doing so."""
    spec = _spec(200, orientation="vertical")
    count = _plotly_tick_count(spec, axis="xaxis")
    assert count is not None, "the vertical branch stopped thinning"
    assert count <= spec.figsize[0] / rotated_tick_pitch_in(9, 45), f"the vertical branch regressed: {count} labels in {spec.figsize[0]} inches"


def test_thinned_labels_are_still_truncated():
    """The two guards are independent; thinning must not drop the truncation."""
    fig = PlotlyRenderer().render(_spec(200))
    texts = list(fig.layout.yaxis.ticktext or ())
    assert texts, "no tick text was set"
    assert max(len(t) for t in texts) <= 40, f"a {max(len(t) for t in texts)}-char label survived: {max(texts, key=len)!r}"
