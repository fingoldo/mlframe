"""A deliberate falsy ``bin_width`` reaches the drawing call; only ``None`` means unset."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.reporting.renderers.matplotlib import MatplotlibRenderer
from mlframe.reporting.spec import HistogramPanelSpec


@pytest.mark.parametrize("bin_width, expected", [(0.0, 0.0), (None, None)])
def test_histogram_drawing_call_gets_the_spec_bin_width(monkeypatch, bin_width, expected):
    """``bin_width=0.0`` must reach ``ax.bar`` as 0.0, while ``None`` lets the renderer derive its own width."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recorded: dict = {}
    fig, ax = plt.subplots()
    real_bar = ax.bar

    def _bar(*args, **kwargs):
        """Record the width the renderer passed to the drawing call, then draw as usual."""
        recorded["width"] = kwargs.get("width")
        return real_bar(*args, **kwargs)

    monkeypatch.setattr(ax, "bar", _bar)
    centers = np.linspace(0.0, 1.0, 5)
    spec = HistogramPanelSpec(values=np.array([1.0, 2.0, 3.0, 2.0, 1.0]), bin_centers=centers, bin_width=bin_width)
    MatplotlibRenderer()._histogram(ax, spec)
    plt.close(fig)

    assert "width" in recorded, "the renderer never called ax.bar"
    derived_width_is_positive = recorded["width"] is not None and recorded["width"] > 0.0
    assert derived_width_is_positive if expected is None else recorded["width"] == pytest.approx(expected), recorded
