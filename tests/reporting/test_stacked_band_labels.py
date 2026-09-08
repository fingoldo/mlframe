"""Labels stamped above a panel must not all share one height.

Every vspan label and every vline label was placed at ``y=1, yref="y domain", yanchor="bottom"`` -- the same
horizontal line just above the plot area. Two adjacent regimes, or two change points a few pixels apart,
printed on top of each other. A regime chart exists to show adjacent regimes, so this is the normal case
rather than an edge one.

There are two vline code paths (a datetime-safe shape and plotly's own ``add_vline``) and both are covered:
the second was missed by the first version of the fix and still stacked its labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting.renderers.plotly import _STACKED_LABEL_ROWS, PlotlyRenderer
from mlframe.reporting.spec import FigureSpec, LinePanelSpec

COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")


def _panel(*, vspans=None, vlines=None, x=None) -> FigureSpec:
    """A line panel carrying adjacent bands and/or change points."""
    xs = np.linspace(0, 10, 200) if x is None else x
    panel = LinePanelSpec(x=xs, y=np.sin(np.linspace(0, 10, len(xs))), title="Regimes", vspans=vspans, vlines=vlines)
    return FigureSpec(panels=((panel,),), figsize=(9.0, 4.0))


def _label_shifts(fig, prefix: str):
    """``(text, yshift)`` for the stamped labels, in drawing order."""
    return [(a.text, a.yshift or 0) for a in fig.layout.annotations if a.text and a.text.startswith(prefix)]


def test_adjacent_regime_labels_do_not_share_one_height():
    """Bands whose labels would land on each other must go on different rows.

    The rows are chosen by MEASUREMENT now, not by alternating on the index, so the fixture has to put the
    bands close enough that their labels really do collide -- ``regime_0`` is about 0.6 data units wide on
    this axis, so bands a third of a unit apart overlap and bands two units apart do not.
    """
    spans = tuple((i * 0.3, i * 0.3 + 0.25, c, 0.15, f"regime_{i}") for i, c in enumerate(COLORS))
    shifts = _label_shifts(PlotlyRenderer().render(_panel(vspans=spans)), "regime_")
    assert len(shifts) == 4, f"expected four band labels, got {shifts}"
    for (name_a, ya), (name_b, yb) in zip(shifts, shifts[1:]):
        assert ya != yb, f"{name_a} and {name_b} are both at yshift={ya}, so they overprint"


def test_well_separated_bands_are_not_stacked():
    """Guard: stacking costs vertical room, so it must only happen where the labels actually collide."""
    spans = tuple((float(a), float(a + 2.0), c, 0.15, f"regime_{i}") for i, (a, c) in enumerate(zip([0, 3, 6, 9], COLORS)))
    shifts = {s for _, s in _label_shifts(PlotlyRenderer().render(_panel(vspans=spans)), "regime_")}
    assert len(shifts) == 1, f"bands three units apart were stacked across {len(shifts)} rows anyway"


def test_neighbouring_change_point_labels_do_not_share_one_height():
    """Two change points a few pixels apart is exactly the collision the finding describes."""
    vlines = tuple((float(v), "#333333", f"change_{i}") for i, v in enumerate([2.0, 2.3, 6.0]))
    shifts = _label_shifts(PlotlyRenderer().render(_panel(vlines=vlines)), "change_")
    assert len(shifts) == 3, f"expected three change-point labels, got {shifts}"
    assert shifts[0][1] != shifts[1][1], f"the two adjacent change points share yshift={shifts[0][1]}"


def test_the_datetime_vline_path_staggers_too():
    """A datetime x axis takes a different code path; it stacked its labels after the first fix."""
    x = pd.date_range("2026-01-01", periods=200, freq="D")
    vlines = tuple((x[i], "#333333", f"change_{k}") for k, i in enumerate((40, 44, 120)))
    shifts = _label_shifts(PlotlyRenderer().render(_panel(vlines=vlines, x=x)), "change_")
    assert len(shifts) == 3, f"expected three labels on the datetime path, got {shifts}"
    assert shifts[0][1] != shifts[1][1], f"datetime vline labels share yshift={shifts[0][1]}"


@pytest.mark.parametrize("n", [1, 2, 5])
def test_the_shift_cycles_over_a_bounded_number_of_rows(n):
    """Staggering must not walk the labels off the top of the figure as the count grows."""
    spans = tuple((float(i), float(i + 1), COLORS[i % len(COLORS)], 0.15, f"regime_{i}") for i in range(n))
    shifts = [s for _, s in _label_shifts(PlotlyRenderer().render(_panel(vspans=spans)), "regime_")]
    assert len(set(shifts)) <= _STACKED_LABEL_ROWS, f"labels use {len(set(shifts))} distinct rows, more than the {_STACKED_LABEL_ROWS} allowed"
