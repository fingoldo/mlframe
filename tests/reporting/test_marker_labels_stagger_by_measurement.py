"""Alternating vline/vspan labels by INDEX fixes a pair and does nothing for three.

Rows 0, 1, 0 puts the first and third of three change points a few pixels apart straight back on top of
each other, which is exactly the threshold-sweep panel this staggering exists for (F1-optimal,
Youden-optimal and cost-optimal thresholds routinely land within a percent of each other). And measuring
the labels against the range of the MARKERS rather than of the axis made three points spanning 2% of a
unit axis look as though each had the whole panel to itself.
"""

from __future__ import annotations

import numpy as np

from mlframe.reporting.renderers._shared_helpers import stagger_label_rows
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import FigureSpec, LinePanelSpec

X = np.linspace(0.0, 1.0, 100)


def _panel(**kw) -> FigureSpec:
    """A threshold-sweep line panel carrying the requested markers."""
    return FigureSpec(panels=((LinePanelSpec(x=X, y=X, series_labels=("metric",), title="threshold sweep", xlabel="threshold", ylabel="value", **kw),),), figsize=(9.0, 4.0))


def _labelled(fig, needle: str):
    """``(text, yshift)`` for every annotation whose text contains ``needle``."""
    return sorted((a.text, int(a.yshift or 0)) for a in fig.layout.annotations if a.text and needle in a.text)


def test_the_helper_gives_close_labels_distinct_rows():
    """Three labels inside one percent of the panel cannot share a row."""
    rows = stagger_label_rows([0.50, 0.51, 0.52], ["regime A", "regime B", "regime C"], fontsize=8, x_span=1.0, width_in=8.0)
    assert len(set(rows)) == 3, f"three labels 1% of the panel apart share rows {rows}"


def test_the_helper_leaves_well_separated_labels_on_one_row():
    """Guard: stacking costs vertical room, so it must only happen where it is needed."""
    rows = stagger_label_rows([0.0, 0.4, 0.8], ["a", "b", "c"], fontsize=8, x_span=1.0, width_in=8.0)
    assert rows == [0, 0, 0], f"labels a third of the panel apart were stacked anyway: {rows}"


def test_three_close_vlines_do_not_share_a_row():
    """The rendered defect: three operating-point labels printed on top of each other."""
    fig = PlotlyRenderer().render(_panel(vlines=((0.50, "red", "F1-optimal"), (0.53, "blue", "Youden-optimal"), (0.56, "green", "cost-optimal"))))
    shifts = [s for _, s in _labelled(fig, "optimal")]
    assert len(set(shifts)) == 3, f"the three labels sit at {shifts}"


def test_a_far_away_vline_returns_to_the_first_row():
    """The stack is per-collision, not a running counter."""
    fig = PlotlyRenderer().render(_panel(vlines=((0.50, "red", "F1-optimal"), (0.53, "blue", "Youden-optimal"), (0.95, "gray", "saturation"))))
    far = dict(_labelled(fig, "saturation"))["saturation"]
    first = dict(_labelled(fig, "F1-optimal"))["F1-optimal"]
    assert far == first, f"a label at the far end of the axis was pushed to row {far} instead of {first}"


def test_the_vline_labels_hang_inside_the_plot_area():
    """Stacking UPWARD put the first row in the subplot title's strip: one collision traded for another."""
    fig = PlotlyRenderer().render(_panel(vlines=((0.50, "red", "F1-optimal"), (0.53, "blue", "Youden-optimal"))))
    anns = [a for a in fig.layout.annotations if a.text and "optimal" in a.text]
    assert anns, "no vline label was drawn"
    for ann in anns:
        assert ann.yanchor == "top" and float(ann.y) == 1.0, f"{ann.text!r} is anchored above the panel, where the title is"
        assert int(ann.yshift or 0) <= 0, f"{ann.text!r} shifts upward ({ann.yshift}), out of the plot area"


def test_close_vspans_do_not_share_a_row():
    """Two adjacent regimes is what a regime chart is FOR."""
    fig = PlotlyRenderer().render(_panel(vspans=((0.05, 0.10, "orange", 0.2, "regime one"), (0.11, 0.15, "purple", 0.2, "regime two"))))
    shifts = [s for _, s in _labelled(fig, "regime")]
    assert len(set(shifts)) == 2, f"the two regime labels sit at {shifts}"
