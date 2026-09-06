"""A nominal class label must not be drawn on a diverging continuous colour scale.

The separability scatter passed the raw class vector into "coolwarm". A diverging scale says its extremes
are opposite ends of ONE quantity, so three classes read as low / middle / high rather than as three
categories -- and the middle class landed on the pale white midpoint, nearly invisible against the panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.reporting.charts.engineered_separability import compose_separability_figure
from mlframe.reporting.spec import ScatterPanelSpec

DIVERGING = {"coolwarm", "rdbu", "rdbu_r", "bwr", "seismic", "spectral"}


def _panel(n_classes: int = 3) -> ScatterPanelSpec:
    """The separability scatter for a problem with ``n_classes`` classes."""
    rng = np.random.default_rng(0)
    n = 1200
    y = rng.integers(0, n_classes, n)
    X = pd.DataFrame({"feat_a": rng.normal(y * 1.5, 1, n), "feat_b": rng.normal(-y * 0.8, 1, n), "feat_c": rng.normal(0, 1, n)})
    spec = compose_separability_figure(X, y, list(X.columns))
    return next(p for row in spec.panels for p in row if isinstance(p, ScatterPanelSpec))


def test_the_class_overlay_is_not_a_diverging_scale():
    """The defect: a nominal label encoded as 'extremes of a quantity'."""
    assert _panel().colormap.lower() not in DIVERGING, f"class labels are drawn on {_panel().colormap}, a diverging scale"


@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_the_scale_is_pinned_to_whole_class_bands(n_classes):
    """Pinning keeps class k in the middle of band k, so two runs colour the same class the same way."""
    panel = _panel(n_classes)
    assert panel.color_vmin == -0.5, f"lower bound is {panel.color_vmin}, so class 0 is not centred in its band"
    assert panel.color_vmax == n_classes - 0.5, f"upper bound is {panel.color_vmax} for {n_classes} classes"


def test_the_colours_are_contiguous_codes_not_raw_labels():
    """Raw labels with gaps (say 0, 2, 7) would spread the classes unevenly across the scale."""
    panel = _panel(3)
    codes = np.unique(np.asarray(panel.point_color))
    assert codes.tolist() == [0.0, 1.0, 2.0], f"class codes are {codes.tolist()}, not contiguous"


def test_the_colorbar_still_says_how_many_classes_there_are():
    """The reader has to know the scale is categorical and how many bands to expect."""
    assert "3" in _panel(3).colorbar_label, f"the colourbar label does not state the class count: {_panel(3).colorbar_label!r}"
