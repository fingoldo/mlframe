"""The target-mix second pass fits no child once the fit's soft ``max_runtime_mins`` deadline has passed."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _target_mix_fe as tm
from mlframe.feature_selection.filters._fe_deadline import fe_deadline_scope, timer


def _children_fitted(deadline, monkeypatch) -> int:
    """Run the second pass under ``deadline`` with a child that records its fits; returns how many children were fitted."""
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame({"p": rng.random(n), "q": rng.random(n)})
    u = 3.0 * X["p"].to_numpy() + np.sin(5.0 * X["q"].to_numpy())
    y = u + 0.5 * rng.standard_normal(n)
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
    with fe_deadline_scope(deadline):
        tm.second_pass_recipes(object(), X, y, {"u": u}, max_bases=2, scan_rows=20000, min_relative_gain=0.001)
    return len(fitted)


def test_no_child_is_fitted_after_the_deadline(monkeypatch):
    """A spent budget stops the pass before the child fit, which would otherwise run the whole nested MRMR."""
    assert _children_fitted(timer() - 1.0, monkeypatch) == 0


def test_a_child_is_fitted_before_the_deadline_and_without_one(monkeypatch):
    """The gate is inert while time remains and when no budget is set."""
    assert _children_fitted(timer() + 3600.0, monkeypatch) == 1
    assert _children_fitted(None, monkeypatch) == 1
