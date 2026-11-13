"""A fit nested inside another fit (the target-mix child) neither extends nor ends the enclosing fit's FE deadline.

The child estimator is a clone that keeps max_runtime_mins, so its fit published a fresh deadline measured from its own start,
and the finally of MRMR.fit then cleared the thread's deadline outright: after the first child fit the parent had no budget at all.
"""

from __future__ import annotations

from timeit import default_timer as timer

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._fe_deadline as D
from mlframe.feature_selection.filters.mrmr import MRMR


def _small_fit(**kwargs):
    """Fit a tiny MRMR with no FE steps, so the fit's own deadline handling is all that runs."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.standard_normal(300), "b": rng.standard_normal(300)})
    y = pd.Series((X["a"] + 0.1 * rng.standard_normal(300) > 0).astype(int), name="y")
    MRMR(verbose=0, fe_max_steps=0, **kwargs).fit(X, y)


def test_nested_fit_leaves_the_enclosing_deadline_in_force():
    """After a fit inside an enclosing deadline scope, the enclosing deadline is still the one in force."""
    outer = timer() + 3600.0
    with D.fe_deadline_scope(outer):
        _small_fit(max_runtime_mins=1.0)
        assert D.current_fe_deadline() == outer


def test_nested_fit_with_a_later_own_budget_is_bounded_by_the_enclosing_deadline(monkeypatch):
    """A fit whose own budget ends later than the enclosing deadline publishes the enclosing deadline."""
    published = []
    real = D.set_fe_deadline
    monkeypatch.setattr(D, "set_fe_deadline", lambda dl: (published.append(dl), real(dl))[1])
    outer = timer() + 30.0
    with D.fe_deadline_scope(outer):
        _small_fit(max_runtime_mins=60.0)
    assert published[0] == outer


def test_top_level_fit_leaves_no_deadline_behind():
    """A fit with no enclosing deadline leaves none behind once it returns."""
    D.clear_fe_deadline()
    _small_fit(max_runtime_mins=1.0)
    assert D.current_fe_deadline() is None
