"""A plotly colorbar title is drawn horizontally above a 12px bar, so a long one runs off the figure.

The bar is pinned to the right edge of its subplot under a fixed 40 px right margin. The codebase already
passes labels that do not fit -- ``error_analysis`` sends "mean error (darker = worse); cell number = rows
in cell" (55 chars) and ``interaction_strength`` sends "H (0 additive .. 1 pure interaction)".

The matplotlib twin needs no equivalent: its colorbar title is drawn VERTICALLY along the bar, where the
room available is the panel height, so the same label fits. That asymmetry is why this is a plotly-only fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.renderers._plotly_heatmap import _COLORBAR_TITLE_WRAP
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import FigureSpec, HeatmapPanelSpec

LONG_LABEL = "mean error (darker = worse); cell number = rows in cell"


def _spec(label: str) -> FigureSpec:
    """A heatmap carrying ``label`` on its colorbar."""
    rng = np.random.default_rng(0)
    matrix = rng.random((8, 10))
    panel = HeatmapPanelSpec(
        matrix=matrix,
        row_labels=tuple(f"segment_{i}" for i in range(8)),
        col_labels=tuple(f"bin_{i}" for i in range(10)),
        title="Error by segment and bin",
        colorbar_label=label,
    )
    return FigureSpec(panels=((panel,),), figsize=(9.0, 5.0))


def _colorbar_title(fig) -> str:
    """The colorbar title text off a rendered plotly figure."""
    bars = [tr for tr in fig.data if getattr(tr, "colorbar", None) is not None]
    assert bars, "no trace carries a colorbar"
    return bars[0].colorbar.title.text


def test_the_fixture_label_is_one_the_codebase_really_passes():
    """Guard: a label short enough to fit would make the rest of this file vacuous."""
    assert len(LONG_LABEL) > _COLORBAR_TITLE_WRAP * 2, f"the fixture label is only {len(LONG_LABEL)} chars"


def test_a_long_colorbar_title_is_wrapped_on_plotly():
    """Every line has to be short enough to sit above the bar rather than run past the margin."""
    title = _colorbar_title(PlotlyRenderer().render(_spec(LONG_LABEL)))
    assert "<br>" in title, f"the title was not wrapped: {title!r}"
    longest = max(len(line) for line in title.split("<br>"))
    assert longest <= _COLORBAR_TITLE_WRAP, f"a {longest}-char line survives wrapping: {title!r}"


def test_wrapping_preserves_every_word():
    """A wrap that drops content is worse than an overflow."""
    title = _colorbar_title(PlotlyRenderer().render(_spec(LONG_LABEL)))
    assert title.replace("<br>", " ").split() == LONG_LABEL.split(), f"wrapping changed the text: {title!r}"


@pytest.mark.parametrize("label", ["PSI", "rows in cell"])
def test_a_short_title_is_left_alone(label):
    """No gratuitous line breaks on labels that already fit."""
    assert _colorbar_title(PlotlyRenderer().render(_spec(label))) == label


def test_matplotlib_keeps_the_label_on_one_line():
    """Its title runs along the bar, so the room is the panel height and wrapping would only shorten it."""
    fig = MatplotlibRenderer().render(_spec(LONG_LABEL))
    try:
        texts = [t.get_text() for ax in fig.axes for t in ([ax.yaxis.label, ax.xaxis.label]) if t.get_text()]
        assert LONG_LABEL in texts, f"matplotlib no longer draws the colorbar label intact; got {texts}"
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
