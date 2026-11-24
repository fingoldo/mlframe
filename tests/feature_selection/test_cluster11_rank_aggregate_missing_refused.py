"""Cluster 11 regression: a feature missing from the rank aggregate is refused, not padded with the minimum score (audit 2026-11-20, finding 11-F16)."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid import _arms_rank_aggregation as ra


def _frame():
    """Three named columns and a target."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["a", "b", "c"])
    y = rng.integers(0, 2, size=40)
    return X, y


def test_missing_aggregate_score_is_refused(monkeypatch):
    """If the aggregate omits a feature, the arm raises KeyError naming it instead of inventing a score (11-F16)."""
    monkeypatch.setattr(ra, "base_score_table", lambda X, y, random_state=0: pd.DataFrame(index=list(X.columns)))
    monkeypatch.setattr(ra, "aggregate_ranks", lambda table, rule: ({"a": 1.0, "b": 2.0}, rule))
    X, y = _frame()
    with pytest.raises(KeyError, match="'c'"):
        ra.RankAggregationArm(k=2)._compute(X, y)


def test_complete_aggregate_is_kept(monkeypatch):
    """Control: when every feature has a score the arm runs and keeps the top k (11-F16)."""
    monkeypatch.setattr(ra, "base_score_table", lambda X, y, random_state=0: pd.DataFrame(index=list(X.columns)))
    monkeypatch.setattr(ra, "aggregate_ranks", lambda table, rule: ({"a": 1.0, "b": 3.0, "c": 2.0}, rule))
    X, y = _frame()
    out = ra.RankAggregationArm(k=2)._compute(X, y)
    assert out["support"].tolist() == [False, True, True]
