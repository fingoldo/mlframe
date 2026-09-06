"""Inline point labels on plotly had neither of matplotlib's two protections.

``colors.auto_text_color`` deliberately picks WHITE for a label sitting on a dark bubble. matplotlib pairs
that with a contrasting path-effect halo, so the text stays legible when it runs off the fill onto the
panel. plotly emitted a plain annotation with a fixed upward shift, so that white text landed on the white
panel background and vanished -- the exact failure auto_text_color exists to prevent, defeated by the
missing outline.

matplotlib also flips the anchors near a panel edge; plotly used one fixed anchor, so a point in the busy
bottom-left corner of a reliability diagram had its label clipped by the axis.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import FigureSpec, ScatterPanelSpec

CORNER_X = np.array([0.02, 0.50, 0.97])
CORNER_Y = np.array([0.03, 0.50, 0.96])


def _annotations(colors=("white", "black", "white")):
    """The inline-label annotations of a scatter with points at both edges and the middle."""
    panel = ScatterPanelSpec(
        x=CORNER_X,
        y=CORNER_Y,
        title="edges",
        inline_labels=tuple((float(a), float(b), f"bin_{i}") for i, (a, b) in enumerate(zip(CORNER_X, CORNER_Y))),
        inline_label_colors=tuple(colors),
    )
    fig = PlotlyRenderer().render(FigureSpec(panels=((panel,),), figsize=(6.0, 5.0)))
    return [a for a in fig.layout.annotations if a.text and a.text.startswith("bin_")]


def test_every_inline_label_gets_a_backing():
    """Without one, a white label on the white panel is invisible."""
    for ann in _annotations():
        assert ann.bgcolor, f"{ann.text} has no backing colour"


def test_the_backing_contrasts_with_the_text():
    """A white-on-white or black-on-black pairing would defeat the point of having one."""
    for ann in _annotations():
        text_is_white = str(ann.font.color).lower() == "white"
        backing_is_dark = ann.bgcolor.startswith("rgba(0,0,0")
        assert text_is_white == backing_is_dark, f"{ann.text}: font={ann.font.color} on backing={ann.bgcolor}"


def test_a_label_at_the_left_edge_is_anchored_away_from_it():
    """A fixed anchor put the text off the axis and it was clipped mid-word."""
    left, middle, _ = _annotations()
    assert left.xanchor == "left", f"the left-edge label is anchored {left.xanchor}, back over the axis"
    assert middle.xanchor == "right", f"an interior label should keep the default side, got {middle.xanchor}"


def test_a_label_at_the_top_edge_is_pushed_downwards():
    """The fixed +8 shift walked a top-edge label out of the panel."""
    *_, top = _annotations()
    assert top.yanchor == "top", f"the top-edge label is anchored {top.yanchor}"
    assert top.yshift < 0, f"the top-edge label is still shifted upward ({top.yshift})"


@pytest.mark.parametrize("colours", [("white", "white", "white"), ("black", "black", "black")])
def test_uniform_label_colours_still_get_matching_backings(colours):
    """The backing is chosen per label, not once for the panel."""
    anns = _annotations(colours)
    expected_dark = colours[0] == "white"
    for ann in anns:
        assert ann.bgcolor.startswith("rgba(0,0,0") == expected_dark, f"{ann.text}: {ann.font.color} on {ann.bgcolor}"
