"""Horizontal bar labels were thinned on matplotlib and on plotly's VERTICAL branch, but not on plotly's horizontal one.

A 200-row feature-importance / slice-finder / WoE chart therefore had a clean 20-label axis in the PNG and
an unreadable band of overlapping text in the HTML -- from one spec. The bars are one per category either
way; only the labels subsample.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import _BAR_XTICK_KEEP, _BAR_XTICK_THIN_THRESHOLD, PlotlyRenderer
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
    """The defect: 200 labels drawn on top of each other."""
    count = _plotly_tick_count(_spec(200))
    assert count is not None, "plotly still draws every category label on a 200-row chart"
    assert count <= _BAR_XTICK_KEEP, f"plotly drew {count} labels, more than the {_BAR_XTICK_KEEP} the policy keeps"


def test_both_backends_thin_to_the_same_count():
    """One spec, one reading."""
    spec = _spec(200)
    assert _plotly_tick_count(spec) == _mpl_tick_count(spec), "the backends thin a 200-row chart differently"


@pytest.mark.parametrize("n", [5, 12, _BAR_XTICK_THIN_THRESHOLD])
def test_a_short_chart_keeps_every_label(n):
    """Thinning must not fire where the labels already fit."""
    count = _plotly_tick_count(_spec(n))
    assert count is None or count == n, f"plotly thinned a {n}-row chart to {count} labels"


def test_the_vertical_branch_is_unchanged():
    """Guard: the vertical orientation already thinned, and must keep doing so."""
    count = _plotly_tick_count(_spec(200, orientation="vertical"), axis="xaxis")
    assert count is not None and count <= _BAR_XTICK_KEEP, f"the vertical branch regressed: {count}"


def test_thinned_labels_are_still_truncated():
    """The two guards are independent; thinning must not drop the truncation."""
    fig = PlotlyRenderer().render(_spec(200))
    texts = list(fig.layout.yaxis.ticktext or ())
    assert texts, "no tick text was set"
    assert max(len(t) for t in texts) <= 40, f"a {max(len(t) for t in texts)}-char label survived: {max(texts, key=len)!r}"
