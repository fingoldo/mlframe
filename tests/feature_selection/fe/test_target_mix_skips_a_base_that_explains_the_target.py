"""The target-mix second pass runs no child fit for a base that leaves only noise in the mixed target."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _target_mix_fe as tm


def _frame(n: int = 6000, seed: int = 0):
    """A target that is a design column plus ``noise_sd`` of noise, and the frame the second pass would fit a child on."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"p": rng.random(n), "q": rng.random(n)})
    u = 3.0 * X["p"].to_numpy() + np.sin(5.0 * X["q"].to_numpy())
    return X, u, rng


def _run(noise_sd: float, monkeypatch):
    """Run the second pass with a child fit that records its calls; returns how many children were fitted."""
    X, u, rng = _frame()
    y = u + noise_sd * rng.standard_normal(len(u))
    fitted = []

    class _Child:
        """A child that records the fit and proposes nothing."""

        _engineered_features_: list = []
        _engineered_recipes_: list = []

        def fit(self, X_, y_):
            """Record the call."""
            fitted.append(len(y_))

    monkeypatch.setattr(tm, "_child", lambda estimator: _Child())
    monkeypatch.setattr("mlframe.feature_selection.filters._target_mix_gate.target_has_structure", lambda *a, **k: True)
    tm.second_pass_recipes(object(), X, y, {"u": u}, max_bases=2, scan_rows=20000, min_relative_gain=0.001)
    return len(fitted)


def test_a_base_that_explains_all_but_noise_runs_no_child_fit(monkeypatch):
    """Residual variance far below 0.1% of the target variance: nothing is left to find, so no child is fitted."""
    assert _run(noise_sd=0.0005, monkeypatch=monkeypatch) == 0


def test_a_base_that_leaves_real_variance_still_fits_a_child(monkeypatch):
    """A residual of a few percent of the target variance is worth the second pass."""
    assert _run(noise_sd=0.5, monkeypatch=monkeypatch) == 1
