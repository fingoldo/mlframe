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


def test_the_child_fit_runs_with_the_analytic_null_lowered_to_its_validated_floor(monkeypatch):
    """Inside the child fit the analytic null applies from 20000 rows; outside it the default threshold is back."""
    from mlframe.feature_selection.filters._analytic_mi_null import _ANALYTIC_NULL_MIN_N_DEFAULT, analytic_null_applicable, analytic_null_min_n

    monkeypatch.delenv("MLFRAME_MI_ANALYTIC_NULL_MIN_N", raising=False)
    seen = []
    X, u, rng = _frame()
    y = u + 0.5 * rng.standard_normal(len(u))

    class _Child:
        """A child that records the analytic threshold it runs under."""

        _engineered_features_: list = []
        _engineered_recipes_: list = []

        def fit(self, X_, y_):
            """Record the threshold and whether 20000 rows are served by the analytic null."""
            seen.append((analytic_null_min_n(), analytic_null_applicable(20_000, 10, 10)))

    monkeypatch.setattr(tm, "_child", lambda estimator: _Child())
    monkeypatch.setattr("mlframe.feature_selection.filters._target_mix_gate.target_has_structure", lambda *a, **k: True)
    assert analytic_null_min_n() == _ANALYTIC_NULL_MIN_N_DEFAULT and not analytic_null_applicable(20_000, 10, 10)
    tm.second_pass_recipes(object(), X, y, {"u": u}, max_bases=1, scan_rows=20000, min_relative_gain=0.001)
    assert seen == [(20_000, True)]
    assert analytic_null_min_n() == _ANALYTIC_NULL_MIN_N_DEFAULT  # restored after the fit


def test_an_explicit_environment_threshold_still_wins_inside_the_child(monkeypatch):
    """A user who set the threshold keeps it: the lowering never overrides an explicit setting."""
    from mlframe.feature_selection.filters._analytic_mi_null import analytic_null_min_n, analytic_null_min_n_lowered

    monkeypatch.setenv("MLFRAME_MI_ANALYTIC_NULL_MIN_N", "50000")
    with analytic_null_min_n_lowered(20_000):
        assert analytic_null_min_n() == 50_000
