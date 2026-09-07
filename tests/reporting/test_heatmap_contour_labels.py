"""An inline contour label was written straight across the cell values it crossed.

The 0.10 / 0.25 triage contours are labelled inline, and an inline label follows its contour -- so on a grid
whose drift ramps left to right the label runs diagonally through several cells and lands on their numbers.
Seen in a 25x10 render: "significant 0.25" over two cells of the late-drift block. The cell values are the
more precise reading, so where both would compete the wording moves to the legend, in BOTH backends.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.renderers.plotly import PlotlyRenderer
from mlframe.reporting.spec import FigureSpec, HeatmapPanelSpec

CONTOURS = ((0.10, "orange", "dash", "moderate 0.1"), (0.25, "red", "solid", "significant 0.25"))
LABELS = {c[3] for c in CONTOURS}


def _panel(rows: int, cols: int, with_text: bool) -> HeatmapPanelSpec:
    """A grid that crosses both thresholds, so both contours actually exist and want labelling."""
    mat = np.linspace(0.0, 1.0, rows * cols).reshape(rows, cols)
    return HeatmapPanelSpec(
        matrix=mat,
        row_labels=tuple(f"f{i}" for i in range(rows)),
        col_labels=tuple(f"t{j}" for j in range(cols)),
        title="drift",
        cell_text=mat if with_text else None,
        text_format=".2f",
        threshold_contours=CONTOURS,
    )


def _mpl_texts(rows: int, cols: int, with_text: bool):
    """``(all text strings on the axes, all legend label strings)`` for a rendered matplotlib heatmap."""
    fig = MatplotlibRenderer().render(FigureSpec(panels=((_panel(rows, cols, with_text),),), figsize=(8.0, 6.0)))
    try:
        fig.canvas.draw()
        axes = next(a for a in fig.axes if a.images)
        legend = axes.get_legend()
        return {t.get_text() for t in axes.texts}, {t.get_text() for t in legend.get_texts()} if legend else set()
    finally:
        plt.close(fig)


def test_the_contour_label_is_not_written_over_the_cell_values():
    """The defect: with per-cell numbers drawn, an inline label sits on top of them."""
    texts, _ = _mpl_texts(8, 8, with_text=True)
    assert not (LABELS & texts), f"{LABELS & texts} was drawn inline across a grid that also carries cell values"


def test_the_threshold_is_still_named_when_the_inline_label_is_suppressed():
    """Suppressing the label must not lose the wording -- an anonymous squiggle is what this replaced."""
    _, legend = _mpl_texts(8, 8, with_text=True)
    assert LABELS <= legend, f"the legend names {legend}, losing {LABELS - legend}"


def test_a_grid_with_no_cell_values_keeps_the_inline_label():
    """Nothing competes for those pixels on a dense grid, and inline beats a legend for reading a contour."""
    texts, legend = _mpl_texts(40, 40, with_text=True)  # 1600 cells: past the cell-text ceiling
    assert LABELS & texts, "the inline label was dropped from a grid that draws no cell values"
    assert not legend, "a legend was added even though the inline label is drawn"


@pytest.mark.parametrize(("rows", "cols", "inline_expected"), [(8, 8, False), (40, 40, True)])
def test_plotly_suppresses_its_inline_label_on_the_same_grids(rows, cols, inline_expected):
    """Backend parity: one spec must not name its thresholds differently in HTML than in PNG."""
    fig = PlotlyRenderer().render(FigureSpec(panels=((_panel(rows, cols, True),),), figsize=(8.0, 6.0)))
    contours = [tr for tr in fig.data if tr.type == "contour"]
    assert contours, "no contour trace was drawn at all"
    for tr in contours:
        assert bool(tr.contours.showlabels) is inline_expected
        assert tr.name in LABELS, f"the threshold lost its name entirely: {tr.name!r}"
