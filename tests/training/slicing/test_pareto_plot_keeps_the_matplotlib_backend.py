"""Rendering the Pareto plot writes the file without switching the process-wide matplotlib backend (tracker 2026-11-02 wave 16, IMP16-9)."""

import os

import matplotlib
import numpy as np

from mlframe.training.slicing._slice_pareto_plot import _render_matplotlib


def test_png_is_written_and_the_backend_is_unchanged(tmp_path):
    """matplotlib.use('Agg', force=True) used to replace a notebook's inline backend for the rest of the session."""
    matplotlib.use("pdf", force=True)
    before = matplotlib.get_backend()
    path = str(tmp_path / "pareto.png")
    means, stds = np.linspace(1.0, 0.5, 20), np.linspace(0.1, 0.3, 20)
    _render_matplotlib(path, means, stds, [0, 5, 19], 5, {"0.9": 10}, title="t", monitor_metric_name="rmse", direction="min")
    assert os.path.getsize(path) > 1000
    assert matplotlib.get_backend() == before
