"""best_trivial_pair returns None (as documented) when the batch MI kernel yields no finite score."""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters import fe_baselines
from mlframe.feature_selection.filters.hermite_fe import shared


def test_best_trivial_pair_all_nan_mi_returns_none(monkeypatch):
    """An all-NaN MI batch gives None, not a truthy (None, None, nan) tuple."""
    monkeypatch.setattr(shared, "plugin_mi_classif_batch_njit", lambda X, y, nb: np.full(X.shape[1], np.nan))
    rng = np.random.default_rng(0)
    x_a, x_b = rng.normal(size=200), rng.normal(size=200)
    y = (rng.random(200) > 0.5).astype(np.int64)
    assert fe_baselines.best_trivial_pair(x_a, x_b, y, discrete_target=True) is None
