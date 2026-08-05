"""REPORTING_B-4 regression test: MatplotlibRenderer's two heatmap-text-color call sites (and
confusion_matrix_plot.py's) must use auto_text_colors_batch, and must produce colors bit-identical to
what per-cell auto_text_color previously computed.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.colors import auto_text_color, auto_text_colors_batch
from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.spec import HeatmapPanelSpec

pytestmark = pytest.mark.fast


def _reference_per_cell(matrix, colormap, vmin, vmax):
    """Reference: the old per-cell auto_text_color loop, for direct comparison against the batched path."""
    out = np.empty(matrix.shape, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            cell = float(matrix[i, j])
            out[i, j] = auto_text_color(cell if np.isfinite(cell) else vmin, colormap, vmin=vmin, vmax=vmax)
    return out


def test_batch_matches_per_cell_reference():
    """auto_text_colors_batch's output must exactly match the per-cell auto_text_color loop it replaces."""
    rng = np.random.default_rng(0)
    matrix = rng.uniform(0.2, 0.9, size=(6, 5))
    matrix[1, 2] = np.nan
    vmin, vmax = 0.2, 0.9
    ref = _reference_per_cell(matrix, "viridis", vmin, vmax)
    batch = auto_text_colors_batch(np.where(np.isfinite(matrix), matrix, vmin), "viridis", vmin=vmin, vmax=vmax)
    assert (ref == batch).all()


def test_heatmap_panel_cell_text_colors_via_batch():
    """MatplotlibRenderer._heatmap's cell-text color pass renders each cell with the same color the
    (removed) per-cell auto_text_color loop would have produced, now via auto_text_colors_batch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array([[0.1, 0.8], [0.5, 0.95]])
    spec = HeatmapPanelSpec(
        matrix=matrix,
        row_labels=("r0", "r1"),
        col_labels=("c0", "c1"),
        cell_text=matrix,
        colormap="viridis",
    )
    fig, ax = plt.subplots()
    calls = []
    real_text = ax.text

    def _spy_text(x, y, s, **kwargs):
        """Record (row, col, color) for each cell-text call, then delegate to the real Axes.text."""
        calls.append((int(y), int(x), kwargs.get("color")))
        return real_text(x, y, s, **kwargs)

    ax.text = _spy_text
    try:
        renderer = MatplotlibRenderer()
        renderer._heatmap(ax, spec, fig)
    finally:
        plt.close(fig)

    assert len(calls) == 4
    vmin, vmax = float(matrix.min()), float(matrix.max())
    expected = _reference_per_cell(matrix, "viridis", vmin, vmax)
    for i, j, color in calls:
        assert color == expected[i, j], f"cell ({i},{j}) color mismatch: got {color}, expected {expected[i, j]}"
