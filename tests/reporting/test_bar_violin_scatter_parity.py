"""Four more places one FigureSpec read differently on the two backends, and one that misread the data.

* The worst-K highlight ring is 4x the point's own AREA on matplotlib and was a constant 12px circle on
  plotly, so on a panel of large bubbles the ring vanished inside the point it marked.
* The violin inner box claimed 5th/95th-percentile whiskers on plotly and drew to the data range.
* A bar reference line's label was pinned to the panel corner on plotly, far from the line and usually on
  top of the longest bar.
* numpy sorts NaN LAST ascending, so a worst-first (descending) subgroup chart led with every unmeasurable
  group -- blank bars that keep their tick label, under a title reading "is nan x the global".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting.charts.error_analysis import segments_bar
from mlframe.reporting.charts.temporal import compose_target_acf_figure
from mlframe.reporting.colors import BAR_PRIMARY
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import BarPanelSpec, FigureSpec, ScatterPanelSpec, ViolinPanelSpec

BIG_POINT_AREA = 200.0


def test_the_worst_k_ring_grows_with_the_points_it_marks():
    """A ring smaller than its own point is not a highlight."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=120)
    panel = ScatterPanelSpec(
        x=x, y=x + rng.normal(scale=0.4, size=120), point_size=BIG_POINT_AREA, title="t",
        highlight_indices=np.array([0, 5, 9]), highlight_color="red",
    )
    fig = PlotlyRenderer().render(FigureSpec(panels=((panel,),), figsize=(6.0, 4.0)))
    ring = next(t for t in fig.data if t.name == "worst-K")
    point = next(t for t in fig.data if t.name != "worst-K")
    point_px = float(np.max(np.atleast_1d(point.marker.size)))
    assert float(ring.marker.size) > point_px, f"the ring is {ring.marker.size}px across, inside a {point_px}px point"


def test_a_small_point_still_gets_a_visible_ring():
    """Guard: scaling with the data must not shrink the ring to nothing on a fine scatter."""
    panel = ScatterPanelSpec(x=np.arange(10.0), y=np.arange(10.0), point_size=1.0, title="t", highlight_indices=np.array([0]), highlight_color="red")
    fig = PlotlyRenderer().render(FigureSpec(panels=((panel,),), figsize=(6.0, 4.0)))
    ring = next(t for t in fig.data if t.name == "worst-K")
    assert float(ring.marker.size) >= 8.0, f"the ring collapsed to {ring.marker.size}px"


def test_the_violin_box_whiskers_at_the_5th_and_95th_percentiles():
    """``quartilemethod`` picks how quartiles are COMPUTED; it never moved a whisker."""
    groups = tuple(np.random.default_rng(i).normal(size=400) for i in range(3))
    panel = ViolinPanelSpec(groups=groups, group_labels=("a", "b", "c"), title="t", show_box=True)
    fig = PlotlyRenderer().render(FigureSpec(panels=((panel,),), figsize=(8.0, 4.0)))
    boxes = [t for t in fig.data if t.type == "box"]
    assert len(boxes) == len(groups), f"expected one box per group, got {len(boxes)}"
    for box, g in zip(boxes, groups):
        assert float(box.lowerfence[0]) == pytest.approx(float(np.percentile(g, 5)))
        assert float(box.upperfence[0]) == pytest.approx(float(np.percentile(g, 95)))
        assert float(box.upperfence[0]) < float(np.max(g)), "the whisker reaches the data range, which is the defect"


def _bar_reference_annotation(orientation: str):
    """The reference-line label on a bar panel of the given orientation."""
    panel = BarPanelSpec(
        categories=("alpha", "beta", "gamma"), values=np.array([0.2, 0.5, 0.9]), title="a fairly long panel title",
        orientation=orientation, hline=(0.31, "darkorange", "global = 0.31"), xlabel="err", ylabel="grp",
    )
    fig = PlotlyRenderer().render(FigureSpec(panels=((panel,),), figsize=(7.0, 4.0)))
    return next(a for a in fig.layout.annotations if a.text and "global" in a.text)


def test_the_vertical_reference_label_sits_at_its_own_line_and_below_the_title():
    """Both halves matter: at the line, and clear of the strip the subplot title occupies."""
    ann = _bar_reference_annotation("horizontal")
    assert float(ann.x) == pytest.approx(0.31), f"the label is at x={ann.x}, not at the line it names"
    assert ann.yanchor == "top" and float(ann.y) == pytest.approx(1.0), "the label hangs above the plot area, where the panel title is"


def test_the_horizontal_reference_label_stays_on_its_own_value():
    """Guard: the vertical-bar case was already correct and must not move."""
    ann = _bar_reference_annotation("vertical")
    assert float(ann.y) == pytest.approx(0.31), f"the label is at y={ann.y}, not at the line it names"


def test_acf_and_pacf_are_the_same_colour():
    """Two panels of one measurement family; a colour difference the reader looks for a meaning in."""
    spec = compose_target_acf_figure(np.random.default_rng(0).normal(size=500))
    bars = [p for row in spec.panels for p in row if isinstance(p, BarPanelSpec)]
    assert len({p.colors for p in bars}) == 1, f"the ACF/PACF panels are drawn in {[p.colors for p in bars]}"
    assert all(p.colors == (BAR_PRIMARY,) for p in bars), "the panels do not use the shared single-series bar colour"


def test_unmeasurable_subgroups_do_not_lead_the_worst_first_chart():
    """A missing measurement is not the worst result, and it must not poison the global reference either."""
    df = pd.DataFrame({"g": ["a", "b", "c", "d"], "err": [0.4, np.nan, 0.9, 0.2], "count": [10, 5, 20, 7]})
    spec = segments_bar(df, group_col="g", metric_col="err", higher_is_worse=True)
    bar = spec.panels[0][0]
    assert "b" not in bar.categories, f"the NaN group is still charted: {bar.categories}"
    assert bar.categories[0] == "c", f"worst-first led with {bar.categories[0]!r}, not the largest error"
    assert np.isfinite(bar.values).all(), "a blank bar keeping its tick label reads as a measured zero"
    assert "nan" not in bar.title, f"the title reports a NaN ratio: {bar.title!r}"
    assert "1 subgroup" in spec.caption, "the dropped group is invisible; a silent drop reads as 'no problem here'"
