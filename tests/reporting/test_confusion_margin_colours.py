"""The confusion-matrix marginal strips must mean the same thing on both backends.

The two strips answer different questions -- how many rows each TRUE class has, and how much volume the
model sent to each PREDICTED class -- so a reader has to be able to tell them apart. matplotlib painted
them blue and green; plotly painted BOTH with ``TREND_LINE``, a constant whose documented job is the
robust-fit overlay line. The same confusion matrix therefore came out blue-and-green as a PNG and two
identical oranges as HTML, with the orange colliding with the trend line's meaning elsewhere in the report.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np

from mlframe.reporting.colors import CONFUSION_COL_MARGIN, CONFUSION_ROW_MARGIN, TREND_LINE
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import ConfusionMarginsPanelSpec, FigureSpec

K = 5


def _spec() -> FigureSpec:
    """A confusion matrix with both marginal strips."""
    rng = np.random.default_rng(0)
    matrix = rng.random((K, K))
    matrix /= matrix.sum(1, keepdims=True)
    panel = ConfusionMarginsPanelSpec(
        matrix=matrix,
        row_labels=tuple(f"class_{i}" for i in range(K)),
        col_labels=tuple(f"class_{i}" for i in range(K)),
        row_margin=rng.integers(50, 500, K).astype(float),
        col_margin=rng.integers(50, 500, K).astype(float),
        row_margin_label="true support",
        col_margin_label="predicted volume",
        title="Confusion with margins",
    )
    return FigureSpec(panels=((panel,),), figsize=(8.0, 6.0))


def test_the_two_strips_are_different_colours():
    """One colour for both strips makes them indistinguishable, which is what plotly did."""
    assert CONFUSION_ROW_MARGIN != CONFUSION_COL_MARGIN, "the two marginal strips share one colour"


def test_neither_strip_borrows_the_trend_line_colour():
    """TREND_LINE means 'robust fit overlay' elsewhere in the same report."""
    assert CONFUSION_ROW_MARGIN != TREND_LINE and CONFUSION_COL_MARGIN != TREND_LINE


def test_plotly_paints_each_strip_its_own_colour():
    """The defect: both bar traces came out TREND_LINE."""
    fig = PlotlyRenderer().render(_spec())
    bar_colours = [tr.marker.color for tr in fig.data if tr.type == "bar"]
    assert len(bar_colours) == 2, f"expected two marginal strips, got {bar_colours}"
    assert set(bar_colours) == {CONFUSION_ROW_MARGIN, CONFUSION_COL_MARGIN}, f"plotly strip colours are {bar_colours}"


def test_both_backends_use_the_same_two_colours():
    """One spec, one reading -- the backends had drifted to different palettes here."""
    plotly_colours = {tr.marker.color for tr in PlotlyRenderer().render(_spec()).data if tr.type == "bar"}
    fig = MatplotlibRenderer().render(_spec())
    try:
        mpl_colours = {mcolors.to_hex(patch.get_facecolor()) for ax in fig.axes for patch in getattr(ax, "patches", [])}
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
    expected = {CONFUSION_ROW_MARGIN.lower(), CONFUSION_COL_MARGIN.lower()}
    assert {c.lower() for c in plotly_colours} == expected, f"plotly: {plotly_colours}"
    assert expected <= {c.lower() for c in mpl_colours}, f"matplotlib: {mpl_colours}"
