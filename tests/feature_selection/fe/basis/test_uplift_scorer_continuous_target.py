"""``score_features_by_mi_uplift`` must bin a continuous target, not truncate it to integers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._orthogonal_univariate_fe import score_features_by_mi_uplift


def test_uplift_scorer_keeps_signal_of_unit_interval_target() -> None:
    """A target in [0, 1) was cast to a single class, so every engineered MI was 0."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(3000)
    y = (x**2) / (x**2).max() * 0.99
    raw = pd.DataFrame({"x": x})
    eng = pd.DataFrame({"x__He2": x**2 - 1.0})
    scores = score_features_by_mi_uplift(raw, eng, y)
    assert float(scores["engineered_mi"].iloc[0]) > 0.5
