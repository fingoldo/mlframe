"""Regression test for the site-3 joblib audit fix (2026-07-19): ``screen_predictors``'s
``evaluate_candidates`` threading pool must never be constructed, regardless of ``n_workers``.

Isolated/warmed/best-of-3+ measurement at this call site's realistic scales found the pool never wins over
serial: m=10 candidates -> 0.03x, m=320 (wellbore-scale) -> 0.72-0.73x, m=820/n_workers=8 -> 0.81x. The fix
makes ``_screen_predictors.py`` always set ``workers_pool = None``; the retired consuming branch, its
environment switch and joblib itself are now removed from the screen altogether. ``n_workers`` remains an accepted knob (the
Fleuret recheck still cuts its permutation budget into one segment per worker) -- this test pins that no joblib pool exists in the screen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._screen_predictors as _sp
from mlframe.feature_selection.filters.mrmr import MRMR


def _wide_xy(n=600, p=30, seed=1):
    """Small wide fixture, cheap enough to fit quickly with n_workers>1 requested."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    y = ((X["f0"].to_numpy() + X["f1"].to_numpy() - X["f2"].to_numpy() + 0.1 * rng.normal(size=n)) > 0).astype(int)
    return X, y


def test_screen_predictors_never_builds_evaluate_candidates_pool():
    """``MRMR(n_workers=4).fit(...)`` constructs no ``joblib.Parallel`` pool in the screen although ``n_workers > 1`` is requested: the screening modules import no joblib and the fit still
    selects the planted columns."""
    import mlframe.feature_selection.filters._confirm_predictor as _cp
    import mlframe.feature_selection.filters.fleuret as _fl

    for mod in (_sp, _cp, _fl):
        assert not hasattr(mod, "Parallel"), f"{mod.__name__} must not import joblib.Parallel"
    X, y = _wide_xy()
    m = MRMR(n_workers=4, max_runtime_mins=1.0)
    m.fit(X, y)
    assert {"f0", "f1", "f2"} <= {str(m.feature_names_in_[i]) for i in m.support_}


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
