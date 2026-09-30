"""Plotly subplot grid spacing: panels keep their requested height however many rows a figure has.

``vertical_spacing`` used to be a fixed 0.16 FRACTION of the plot height, so the gap grew with the row count and every
panel of a tall grid shrank (a 4-row binary report drew each panel at ~52% of its requested height, the PIT panel
included). These tests read the rendered layout's axis domains and assert on the pixels each panel actually gets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")

from mlframe.reporting.charts.binary import compose_binary_figure
from mlframe.reporting.charts.calibration_drift import build_calibration_drift_spec, calibration_drift
from mlframe.reporting.renderers import get_renderer
from mlframe.reporting.renderers._shared_helpers import PX_PER_INCH
from mlframe.reporting.spec import FigureSpec, LinePanelSpec

_CELL_H_IN = 4.0


def _row_heights_px(fig, spec) -> list:
    """Pixel height of every y-axis domain of the rendered figure, one per subplot."""
    plot_h_px = spec.figsize[1] * PX_PER_INCH
    lay = fig.layout.to_plotly_json()
    heights = []
    for k, ax in lay.items():
        if k.startswith("yaxis") and "domain" in ax and ax.get("overlaying") is None:
            d0, d1 = ax["domain"]
            heights.append((d1 - d0) * plot_h_px)
    return heights


def _line_grid(n_rows: int) -> FigureSpec:
    """Figure spec of n_rows by two identical line panels."""
    x = np.linspace(0, 1, 20)
    panel = LinePanelSpec(x=x, y=x * x, title="panel", xlabel="x", ylabel="y")
    return FigureSpec(panels=tuple((panel, panel) for _ in range(n_rows)), figsize=(12.0, _CELL_H_IN * n_rows))


def test_panel_height_does_not_shrink_with_row_count():
    """Panel height does not shrink with row count."""
    renderer = get_renderer("plotly")
    per_row = {}
    for n_rows in (2, 3, 5):
        spec = _line_grid(n_rows)
        per_row[n_rows] = min(_row_heights_px(renderer.render(spec), spec))
    # Under the fixed-fraction gap these were 336 / 272 / 144 px for a requested 400. A fixed-pixel gap still costs
    # each row (N-1)/N of one gap, so the loss is bounded by a single gap however many rows there are.
    assert per_row[5] >= 0.9 * per_row[2]
    assert per_row[5] >= 0.8 * _CELL_H_IN * PX_PER_INCH


def test_pit_panel_of_seven_panel_binary_report_keeps_its_height():
    """Pit panel of seven panel binary report keeps its height."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 4000)
    s = np.clip(0.3 * y + rng.random(4000) * 0.7, 0, 1)
    spec = compose_binary_figure(y, s, panels_template="ROC PR SCORE_DIST KS THRESHOLD GAIN PIT")
    assert len(spec.panels) == 4
    fig = get_renderer("plotly").render(spec)
    heights = _row_heights_px(fig, spec)
    cell_px = spec.figsize[1] * PX_PER_INCH / len(spec.panels)
    assert min(heights) >= 0.75 * cell_px


def test_caption_clears_bottom_row_axis_title():
    """Caption clears bottom row axis title."""
    spec = _line_grid(1)
    spec = FigureSpec(panels=spec.panels, figsize=spec.figsize, caption="How to read: something.")
    fig = get_renderer("plotly").render(spec)
    cap = [a for a in fig.layout.annotations if (a.text or "").startswith("How to read")]
    assert cap
    # Tick labels (~20px) plus a one-line x-axis title (~26px) sit below the axis; the old fixed -38px shift overlapped the title.
    assert -cap[0].yshift >= 46
    assert fig.layout.margin.b >= -cap[0].yshift + 10


def test_secondary_y_axis_title_does_not_collide_with_next_column():
    """Secondary y axis title does not collide with next column."""
    x = np.linspace(0, 1, 20)
    left = LinePanelSpec(x=x, y=(x, 1 - x), secondary_y=(False, True), secondary_ylabel="queue rate", xlabel="t", ylabel="metric")
    right = LinePanelSpec(x=x, y=x, xlabel="f", ylabel="captured")
    spec = FigureSpec(panels=((left, right),), figsize=(12.0, 4.0))
    fig = get_renderer("plotly").render(spec)
    gap_px = (fig.layout.xaxis2.domain[0] - fig.layout.xaxis.domain[1]) * (spec.figsize[0] * PX_PER_INCH - fig.layout.margin.l - fig.layout.margin.r)
    # Right-hand ticks + title of the left panel, then left ticks + title of the right panel: ~36+26 on each side.
    assert gap_px >= 120


def test_calibration_drift_figure_has_no_rangeslider():
    """Calibration drift figure has no rangeslider."""
    rng = np.random.default_rng(1)
    n = 3000
    y = rng.integers(0, 2, n)
    s = np.clip(0.3 * y + rng.random(n) * 0.7, 0, 1)
    res = calibration_drift(y, s, pd.date_range("2024-01-01", periods=n, freq="h"))
    fig = get_renderer("plotly").render(build_calibration_drift_spec(res))
    lay = fig.layout.to_plotly_json()
    assert not any(lay[k].get("rangeslider", {}).get("visible") for k in lay if k.startswith("xaxis"))
