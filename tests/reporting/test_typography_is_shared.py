"""One FigureSpec must not read as two documents.

Caption size, panel-title size and the caption wrap budget were declared separately in each renderer and
had drifted: captions at 7pt on matplotlib and 9pt on plotly, panel titles at 10 vs 11 under a comment
claiming they matched. Worse, plotly measured its caption wrap against 10pt while drawing at 9 and reused
the narrower SUPTITLE budget for the wider caption band, so a caption could overrun its reserved strip.
"""

from __future__ import annotations

import numpy as np

from mlframe.reporting.renderers import _shared_helpers as shared
from mlframe.reporting.renderers import matplotlib as mpl_mod
from mlframe.reporting.renderers import plotly as plotly_mod
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import FigureSpec, LinePanelSpec

CAPTION = "How to read: each line is one model; higher is better; the shaded band is the bootstrap interval at 95 percent, and points below the diagonal are miscalibrated."


def test_the_two_renderers_agree_on_every_typography_constant():
    """Sizes that differ between backends print the same figure as two documents."""
    for name, mpl_const, plotly_const in [
        ("panel title", mpl_mod._TITLE_FONTSIZE, plotly_mod._PANEL_TITLE_FONTSIZE),
        ("caption size", mpl_mod._CAPTION_FONTSIZE, plotly_mod._CAPTION_FONTSIZE),
        ("suptitle wrap", mpl_mod._SUPTITLE_WRAP_CHARS, plotly_mod._SUPTITLE_WRAP_CHARS),
    ]:
        assert mpl_const == plotly_const, f"{name} is {mpl_const} on matplotlib and {plotly_const} on plotly"


def test_the_constants_come_from_one_place():
    """Equal-by-coincidence is how they drifted the first time; the value has to have one home."""
    assert mpl_mod._TITLE_FONTSIZE is shared.PANEL_TITLE_FONTSIZE
    assert plotly_mod._PANEL_TITLE_FONTSIZE is shared.PANEL_TITLE_FONTSIZE
    assert mpl_mod._CAPTION_FONTSIZE is shared.CAPTION_FONTSIZE
    assert plotly_mod._CAPTION_FONTSIZE is shared.CAPTION_FONTSIZE


def _spec():
    """One labelled line panel carrying the long how-to-read caption."""
    x = np.linspace(0.0, 1.0, 20)
    return FigureSpec(panels=((LinePanelSpec(x=x, y=x, series_labels=("a",), title="a panel"),),), figsize=(8.0, 4.0), caption=CAPTION)


def test_plotly_draws_its_caption_at_the_size_it_wrapped_it_against():
    """A budget measured at one size and drawn at another is how a caption overruns its band."""
    fig = PlotlyRenderer().render(_spec())
    captions = [a for a in fig.layout.annotations if a.text and "How to read" in a.text]
    assert len(captions) == 1, f"expected one caption, got {[a.text for a in fig.layout.annotations]}"
    assert (
        captions[0].font.size == plotly_mod._CAPTION_FONTSIZE
    ), f"the caption is wrapped against {plotly_mod._CAPTION_FONTSIZE}pt but drawn at {captions[0].font.size}pt"


def test_both_backends_fold_the_caption_the_same_way():
    """Same text, same width, same size -- the reader must not get a two-line band on one and three on the other."""
    spec = _spec()
    plotly_lines = next(a for a in PlotlyRenderer().render(spec).layout.annotations if a.text and "How to read" in a.text).text.count("<br>") + 1
    fig = MatplotlibRenderer().render(spec)
    try:
        mpl_lines = next(t.get_text() for t in fig.texts if "How to read" in t.get_text()).count("\n") + 1
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
    assert plotly_lines == mpl_lines, f"the same caption folds onto {mpl_lines} lines on matplotlib and {plotly_lines} on plotly"
