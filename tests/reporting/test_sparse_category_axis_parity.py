"""A bar panel with one to three categories keeps its bars narrow on BOTH backends.

Matplotlib widens the category axis to ``MIN_CATEGORY_SLOTS`` slots so a lone bar does not swell to fill the panel; plotly had no equivalent, so the same spec drew a full-width bar in the HTML report and a
narrow one in the PNG (reporting audit 2026-09-06, WALK-02). Both renderers now take the limits from ``sparse_category_limits``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlframe.reporting.renderers._shared_helpers import MIN_CATEGORY_SLOTS, sparse_category_limits
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import BarPanelSpec, FigureSpec


def _spec(n_categories: int, orientation: str) -> FigureSpec:
    """One bar panel with ``n_categories`` bars."""
    panel = BarPanelSpec(
        categories=tuple(f"c{i}" for i in range(n_categories)),
        values=np.linspace(0.2, 0.9, n_categories),
        title="t",
        xlabel="x",
        ylabel="y",
        orientation=orientation,
    )
    return FigureSpec(panels=((panel,),), figsize=(6.0, 4.0))


def test_the_limits_helper_is_the_documented_window():
    """One bar sits in the middle of ``MIN_CATEGORY_SLOTS`` slots; enough categories leave the axis alone."""
    assert sparse_category_limits(1) == (-MIN_CATEGORY_SLOTS / 2.0, MIN_CATEGORY_SLOTS / 2.0)
    lo, hi = sparse_category_limits(2)
    assert hi - lo == MIN_CATEGORY_SLOTS and (lo + hi) / 2.0 == 0.5
    assert sparse_category_limits(MIN_CATEGORY_SLOTS) is None and sparse_category_limits(0) is None


@pytest.mark.parametrize("n_categories", [1, 2, 3])
def test_vertical_bars_get_the_same_window_on_both_backends(n_categories):
    """The category (x) axis spans the same window in the plotly figure and the matplotlib axes."""
    want = sparse_category_limits(n_categories)
    fig = PlotlyRenderer().render(_spec(n_categories, "vertical"))
    assert tuple(fig.layout.xaxis.range) == pytest.approx(want)
    mfig = MatplotlibRenderer().render(_spec(n_categories, "vertical"))
    assert mfig.axes[0].get_xlim() == pytest.approx(want)
    plt.close(mfig)


@pytest.mark.parametrize("n_categories", [1, 2, 3])
def test_horizontal_bars_get_the_same_window_in_descending_order(n_categories):
    """The category (y) axis spans the window with the first category on top on both backends."""
    lo, hi = sparse_category_limits(n_categories)
    fig = PlotlyRenderer().render(_spec(n_categories, "horizontal"))
    assert tuple(fig.layout.yaxis.range) == pytest.approx((hi, lo))
    assert fig.layout.yaxis.autorange is False
    mfig = MatplotlibRenderer().render(_spec(n_categories, "horizontal"))
    assert mfig.axes[0].get_ylim() == pytest.approx((hi, lo))
    plt.close(mfig)


def test_enough_categories_leave_the_plotly_axis_alone():
    """Four bars or more are not padded: no explicit range, and a horizontal panel keeps its reversed autorange."""
    vfig = PlotlyRenderer().render(_spec(MIN_CATEGORY_SLOTS, "vertical"))
    assert vfig.layout.xaxis.range is None
    hfig = PlotlyRenderer().render(_spec(MIN_CATEGORY_SLOTS, "horizontal"))
    assert hfig.layout.yaxis.range is None and hfig.layout.yaxis.autorange == "reversed"
