"""Confirming the next screen candidates ahead on threads leaves the selection unchanged."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _fit(window, monkeypatch):
    """Selected columns of a fit with the given prefetch window, with the fit cache cleared."""
    monkeypatch.setenv("MLFRAME_SCREEN_CONFIRM_PREFETCH", str(window))
    r = np.random.default_rng(3)
    n = 4000
    X = pd.DataFrame(r.standard_normal((n, 12), dtype=np.float32), columns=[f"x{i}" for i in range(12)])
    s = X.x0 + 0.8 * X.x1 - 0.6 * X.x2 + 0.3 * X.x3
    y = pd.Series((s + 0.7 * r.standard_normal(n) > 0).astype(np.int8))
    MRMR._FIT_CACHE.clear()
    return list(MRMR(verbose=0, random_seed=0, n_jobs=1, full_npermutations=6, min_nonzero_confidence=0.9, fe_max_steps=0).fit(X, y).get_feature_names_out())


def test_prefetch_window_keeps_the_selection(monkeypatch):
    """Window 4 and window 0 select the same columns in the same order."""
    assert _fit(4, monkeypatch) == _fit(0, monkeypatch)
