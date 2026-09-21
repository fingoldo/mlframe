"""The FE wall-clock budget stops every family stage, and the FE step's optional tails are skipped once it is spent.

``_fe_budget_ok`` was consulted at four of the ~35 cascade stages, so a spent ``max_runtime_mins`` still let the other ~31 families start;
and the FE step's escalation / additive-fusion / stability-vote tails ran to completion regardless of the deadline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _fe_deadline
from mlframe.feature_selection.filters.mrmr import MRMR


@pytest.fixture(autouse=True)
def _clear_deadline():
    """No deadline leaks into or out of a test in this module."""
    _fe_deadline.set_fe_deadline(None)
    yield
    _fe_deadline.set_fe_deadline(None)


def _frame(n: int = 600, seed: int = 0):
    """A small frame with a categorical group column, so the grouped-agg family has something to aggregate."""
    rng = np.random.default_rng(seed)
    value = rng.normal(size=n)
    grp = rng.choice(["a", "b", "c"], size=n)
    X = pd.DataFrame({"value": value, "other": rng.normal(size=n), "grp": pd.Categorical(grp)})
    y = pd.Series((value + (grp == "a") * 0.8 + 0.3 * rng.normal(size=n) > 0).astype(np.int64), name="y")
    return X, y


def test_tail_budget_helper_reports_only_when_the_deadline_passed(caplog):
    """The tail guard is False with no deadline and False before it, True after it, and says which stage it skipped."""
    from mlframe.feature_selection.filters._mrmr_fe_step._step_score import _fe_tail_budget_spent

    assert _fe_tail_budget_spent("auto-escalation") is False
    _fe_deadline.set_fe_deadline(_fe_deadline.timer() + 3600.0)
    assert _fe_tail_budget_spent("auto-escalation") is False, "a budget still in the future must not skip the stage"
    _fe_deadline.set_fe_deadline(0.0)  # an absolute deadline in the distant past
    with caplog.at_level(logging.INFO):
        assert _fe_tail_budget_spent("auto-escalation", verbose=1) is True
    assert any("auto-escalation" in r.getMessage() for r in caplog.records), [r.getMessage() for r in caplog.records]


def test_family_stage_does_not_start_once_the_budget_is_spent(monkeypatch):
    """A family that only ever checked its enable flag (grouped-agg) is not entered when max_runtime_mins is already spent."""
    import mlframe.feature_selection.filters._grouped_agg_fe as gagg

    calls = {"n": 0}
    real = gagg.hybrid_grouped_agg_fe

    def _spy(*args, **kwargs):
        """Count entries into the family."""
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(gagg, "hybrid_grouped_agg_fe", _spy)
    X, y = _frame()
    MRMR._FIT_CACHE.clear()
    MRMR(
        random_seed=0, n_jobs=1, verbose=0, fe_max_steps=1, full_npermutations=3, baseline_npermutations=2,
        fe_grouped_agg_enable=True, max_runtime_mins=1e-9,
    ).fit(X, y)
    assert calls["n"] == 0, "the grouped-agg family started although the wall-clock budget was already spent"


def test_family_stage_runs_when_the_budget_is_generous(monkeypatch):
    """Control: the same family is entered with a large budget, so the gate is not simply disabling it."""
    import mlframe.feature_selection.filters._grouped_agg_fe as gagg

    calls = {"n": 0}
    real = gagg.hybrid_grouped_agg_fe

    def _spy(*args, **kwargs):
        """Count entries into the family."""
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(gagg, "hybrid_grouped_agg_fe", _spy)
    X, y = _frame(seed=1)
    MRMR._FIT_CACHE.clear()
    MRMR(
        random_seed=0, n_jobs=1, verbose=0, fe_max_steps=1, full_npermutations=3, baseline_npermutations=2,
        fe_grouped_agg_enable=True, max_runtime_mins=120.0,
    ).fit(X, y)
    assert calls["n"] >= 1, "fixture precondition: the grouped-agg family must run when the budget allows it"
