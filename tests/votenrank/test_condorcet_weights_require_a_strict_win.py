"""A weighting that only ties the rival is not a Condorcet-winning weighting."""

from __future__ import annotations

import pandas as pd

from mlframe.votenrank import Leaderboard


def test_identical_models_have_no_condorcet_weights():
    """Two models with identical scores tie under every weighting, so neither can be the Condorcet winner."""
    table = pd.DataFrame({"t1": [3.0, 3.0, 1.0], "t2": [2.0, 2.0, 1.0]}, index=["a", "b", "c"])
    lb = Leaderboard(table)
    assert lb.find_weights_for_condorcet("a") is None
    assert lb.split_models_by_feasibility()["feasible"] == []


def test_dominating_model_still_has_weights():
    """A model that wins on both tasks keeps its strict-win weighting."""
    table = pd.DataFrame({"t1": [3.0, 2.0, 1.0], "t2": [3.0, 2.0, 1.0]}, index=["a", "b", "c"])
    weights = Leaderboard(table).find_weights_for_condorcet("a")
    assert weights is not None and abs(sum(weights.values()) - 1.0) < 1e-9
