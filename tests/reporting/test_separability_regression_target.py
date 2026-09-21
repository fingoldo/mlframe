"""A regression target must not be drawn as 141 "classes" with Fisher J=nan (a production chart did exactly that)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.reporting.charts.engineered_separability import compose_separability_figure


def test_continuous_target_uses_percentile_colouring_and_rank_scores():
    rng = np.random.default_rng(0)
    a = rng.lognormal(3, 1, 3000)
    y = np.log1p(a) + rng.normal(scale=0.3, size=3000)
    spec = compose_separability_figure(pd.DataFrame({"f0": a, "f1": rng.normal(size=3000)}), y, features=["f0", "f1"])
    panel = spec.panels[0][0]
    assert "class" not in panel.colorbar_label
    assert "Fisher" not in panel.title and "nan" not in panel.title
    assert "Spearman" in panel.title
    assert np.nanmin(panel.point_color) >= 0 and np.nanmax(panel.point_color) <= 1


def test_class_target_keeps_fisher_score():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    spec = compose_separability_figure(pd.DataFrame({"f0": y + rng.normal(size=2000), "f1": rng.normal(size=2000)}), y, features=["f0", "f1"])
    assert "Fisher" in spec.panels[0][0].title
