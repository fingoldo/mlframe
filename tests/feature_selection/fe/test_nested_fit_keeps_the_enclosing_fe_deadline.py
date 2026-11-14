"""A fit nested inside another fit (the target-mix child) neither extends nor ends the enclosing fit's FE deadline, and a top-level fit ignores a stale one.

The child estimator is a clone that keeps max_runtime_mins, so its fit published a fresh deadline measured from its own start,
and the finally of MRMR.fit then cleared the thread's deadline outright: after the first child fit the parent had no budget at all.
Restoring whatever was on the thread is not enough either: a top-level fit would then inherit an expired timestamp that earlier code left behind.
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
    """Inside an enclosing fit's boundary, a fit returns with the enclosing deadline still the one in force."""
    outer = timer() + 3600.0
    with D.fit_deadline_boundary():
        D.set_fe_deadline(outer)
        _small_fit(max_runtime_mins=1.0)
        assert D.current_fe_deadline() == outer
    assert D.current_fe_deadline() is None


def test_nested_fit_with_a_later_own_budget_is_bounded_by_the_enclosing_deadline(monkeypatch):
    """A fit whose own budget ends later than the enclosing deadline publishes the enclosing deadline."""
    outer = timer() + 30.0
    published = []
    with D.fit_deadline_boundary():
        D.set_fe_deadline(outer)
        real = D.set_fe_deadline
        monkeypatch.setattr(D, "set_fe_deadline", lambda dl: (published.append(dl), real(dl))[1])
        _small_fit(max_runtime_mins=60.0)
        monkeypatch.setattr(D, "set_fe_deadline", real)
    assert published[0] == outer


def test_top_level_fit_leaves_no_deadline_behind():
    """A fit with no enclosing deadline leaves none behind once it returns."""
    D.clear_fe_deadline()
    _small_fit(max_runtime_mins=1.0)
    assert D.current_fe_deadline() is None


def test_top_level_fit_ignores_a_stale_deadline_left_on_the_thread(monkeypatch):
    """An expired timestamp that earlier code published and never cleared must not bound the next top-level fit: it runs on its own budget and removes it afterwards."""
    published = []
    real = D.set_fe_deadline
    D.set_fe_deadline(timer() - 1.0)
    try:
        monkeypatch.setattr(D, "set_fe_deadline", lambda dl: (published.append(dl), real(dl))[1])
        _small_fit(max_runtime_mins=60.0)
        monkeypatch.setattr(D, "set_fe_deadline", real)
        assert published[0] > timer()
        assert D.current_fe_deadline() is None
    finally:
        D.clear_fe_deadline()
