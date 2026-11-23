"""Cluster 42 regression: a missing gamma is a clear error, a two-step partial opt-in is honoured, and a solver failure is not reported as infeasible (audit 2026-11-20, findings 42-F3, 42-F5, 42-F6)."""

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.votenrank.leaderboard import _cw
from mlframe.votenrank.leaderboard.leaderboard_impl import Leaderboard


def test_optimality_gap_without_gamma_is_a_clear_error():
    """elect_all with an empty optimality_gap settings dict names the missing gamma (42-F6)."""
    table = pd.DataFrame({"t1": [1.0, 2.0], "t2": [3.0, 1.0]}, index=["m1", "m2"])
    with pytest.raises(ValueError, match="requires 'gamma'"):
        Leaderboard(table).elect_all(use_methods={"optimality_gap": {}})


def test_optimality_gap_with_gamma_still_runs():
    """Control: an explicit gamma runs (42-F6)."""
    table = pd.DataFrame({"t1": [1.0, 2.0], "t2": [3.0, 1.0]}, index=["m1", "m2"])
    out = Leaderboard(table).elect_all(use_methods={"optimality_gap": {"gamma": [2]}})
    assert len(out) >= 1


def test_partial_opt_in_reaches_two_step_group_rankings():
    """A parent with allow_partial=True is not rejected by its own groups' partial guard (42-F3)."""
    table = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [2.0, 1.0, 3.0], "c": [3.0, 3.0, 1.0]}, index=["m1", "m2", "m3"])
    lb = Leaderboard(table, allow_partial=True)
    result = lb.two_step_ranking("borda", {"g1": ["a", "b"], "g2": ["c"]})
    assert len(result) == 3


def test_solver_failure_is_logged_not_silent(monkeypatch, caplog):
    """A non-infeasible linprog status is logged as a solver failure (42-F5)."""

    def _fake_linprog(*a, **k):
        """Return a failed solve with status 4 (numerical difficulties)."""
        return {"success": False, "status": 4, "message": "numerical difficulties", "x": None}

    monkeypatch.setattr(_cw, "linprog", _fake_linprog)
    obj = Leaderboard(pd.DataFrame({"t1": [1.0, 2.0], "t2": [2.0, 1.0]}, index=["m1", "m2"]))
    with caplog.at_level(logging.WARNING, logger=_cw.logger.name):
        out = _cw._find_weights_for_majority_graph(obj, [("m2", "m1")])
    assert out is None
    assert any("solver failed" in r.getMessage() for r in caplog.records)
