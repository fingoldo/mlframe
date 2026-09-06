"""Two ways a figure's own geometry worked against the labels it has to carry.

Many builders pass ``suptitle=""`` deliberately, and the layout engine was switched on by the presence of a
suptitle or caption alone -- so those figures got NO engine, and nothing reserved space for 45-degree tick
labels, a long category axis or a colorbar. ``savefig(bbox_inches="tight")`` rescued the saved file by
growing the canvas, which silently produces a figure that is not the requested figsize, and the interactive
``display(fig)`` path has no tight bbox at all: in a notebook those labels were simply clipped.

Separately, ``segments_bar`` grew only its WIDTH with the group count, leaving a fixed 5-inch height to
absorb rotated labels of arbitrary length.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlframe.reporting.charts.error_analysis import segments_bar
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.spec import BarPanelSpec, FigureSpec, HeatmapPanelSpec, LinePanelSpec

LONG_NAMES = tuple(f"a_long_group_name_{i}" for i in range(30))


def _engine_name(spec) -> str:
    """The layout engine matplotlib ends up with for this spec."""
    fig = MatplotlibRenderer().render(spec)
    try:
        return type(fig.get_layout_engine()).__name__
    finally:
        plt.close(fig)


def test_a_plain_line_figure_still_pays_for_no_layout_engine():
    """The engine is not free (~180ms on a rotated-bar figure), so it stays off where nothing needs it."""
    x = np.linspace(0.0, 1.0, 50)
    spec = FigureSpec(panels=((LinePanelSpec(x=x, y=x**2, series_labels=("a",), title="line"),),), figsize=(6.0, 4.0))
    assert _engine_name(spec) == "NoneType", "a figure with no axis furniture was given a layout engine"


@pytest.mark.parametrize(
    "panel",
    [
        BarPanelSpec(categories=LONG_NAMES, values=np.linspace(0, 1, 30), title="t", xtick_rotation=45.0),
        BarPanelSpec(categories=LONG_NAMES, values=np.linspace(0, 1, 30), title="t", orientation="horizontal"),
        HeatmapPanelSpec(matrix=np.zeros((4, 4)), row_labels=tuple("abcd"), col_labels=tuple("abcd"), title="t", colorbar_label="v"),
    ],
    ids=["rotated-ticks", "horizontal-bars", "heatmap-colorbar"],
)
def test_furniture_that_needs_room_turns_the_engine_on(panel):
    """Each of these reserves space that nothing else reserves for it."""
    spec = FigureSpec(panels=((panel,),), figsize=(6.0, 4.0))
    assert _engine_name(spec) == "ConstrainedLayoutEngine", "no layout engine, so this panel's labels are clipped in a notebook"


def test_rotated_labels_are_not_clipped_on_the_notebook_path():
    """The defect, on the path that has no tight bbox to rescue it."""
    spec = FigureSpec(
        panels=((BarPanelSpec(categories=LONG_NAMES, values=np.linspace(0, 1, 30), title="t", xtick_rotation=45.0, xlabel="group", ylabel="value"),),),
        figsize=(9.0, 4.0),
    )
    fig = MatplotlibRenderer().render(spec)
    try:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        lowest = min(t.get_window_extent(renderer).y0 for t in fig.axes[0].get_xticklabels() if t.get_text())
        assert lowest >= fig.get_window_extent().y0, f"a tick label starts at y={lowest:.1f}, below the figure's own bottom edge"
    finally:
        plt.close(fig)


def _segments(n_groups: int, name_len: int):
    """A worst-first subgroup chart with `n_groups` groups whose names are `name_len` characters long."""
    df = pd.DataFrame(
        {
            "g": [("x" * name_len) + str(i) for i in range(n_groups)],
            "err": np.linspace(0.1, 0.9, n_groups),
            "count": np.full(n_groups, 10),
        }
    )
    return segments_bar(df, group_col="g", metric_col="err", higher_is_worse=True, max_groups=60)


def test_the_segments_figure_stops_growing_sideways_forever():
    """Width grew without bound with the group count; a 30-inch letterbox is not a chart."""
    assert _segments(60, 20).figsize[0] <= 16.0, "the figure is still growing its width without a cap"


def test_the_segments_figure_grows_its_height_for_long_labels():
    """A 40-character name projects about two inches vertically at 45 degrees; the height has to allow for it."""
    short, long = _segments(30, 6), _segments(30, 40)
    assert long.figsize[1] > short.figsize[1], f"long names got the same height as short ones: {long.figsize} vs {short.figsize}"
    assert long.figsize[1] <= 12.0, "the height grew without a cap of its own"


def test_the_extra_height_is_roughly_what_the_labels_project():
    """Guard: growth has to track the label geometry, not just be a bigger constant."""
    short, long = _segments(30, 6), _segments(30, 40)
    grew = long.figsize[1] - short.figsize[1]
    assert grew < 40 * math.sin(math.radians(45.0)) / 6.0, f"the figure grew {grew:.2f}in, more than the labels can project"
