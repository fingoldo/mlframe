"""Regression test for the site-3 joblib audit fix (2026-07-19): ``screen_predictors``'s
``evaluate_candidates`` threading pool must never be constructed, regardless of ``n_workers``.

Isolated/warmed/best-of-3+ measurement at this call site's realistic scales found the pool never wins over
serial: m=10 candidates -> 0.03x, m=320 (wellbore-scale) -> 0.72-0.73x, m=820/n_workers=8 -> 0.81x. The fix
makes ``_screen_predictors.py`` always set ``workers_pool = None`` (never call ``joblib.Parallel``) and adds
a defense-in-depth ``_confirm_predictor._EVALUATE_CANDIDATES_POOL_ENABLED = False`` gate around the
consuming branch. ``n_workers`` remains an accepted knob (other call sites, e.g. the Fleuret conditional-
confirmation gate, still branch on it) -- this test only pins that THIS specific pool is never built.
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


def test_screen_predictors_never_builds_evaluate_candidates_pool(monkeypatch):
    """``MRMR(n_workers=4).fit(...)`` must never construct a ``joblib.Parallel`` pool for the
    ``evaluate_candidates`` dispatch, even though ``n_workers > 1`` is explicitly requested."""

    def _boom(*args, **kwargs):
        """Sensor stub: any call proves the retired evaluate_candidates pool branch fired."""
        raise AssertionError("Parallel() must never be constructed for the evaluate_candidates dispatch (site 3, retired)")

    monkeypatch.setattr(_sp, "Parallel", _boom)

    X, y = _wide_xy()
    MRMR(n_workers=4, max_runtime_mins=1.0).fit(X, y)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "--tb=short", "-x", "--no-cov"])
