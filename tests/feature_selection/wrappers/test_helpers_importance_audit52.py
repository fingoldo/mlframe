"""Regression tests: string-column discreteness probe, non-finite vote scores, CPI dtype round trip and CPI failure logging."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.wrappers import _helpers_importance as hi
from mlframe.feature_selection.wrappers._enums import VotesAggregation
from mlframe.utils.log_throttle import reset_throttle_counts


def test_is_discrete_v2_handles_string_column_without_unbound_local():
    """A non-numeric column used to leave ``mask`` unbound and raise UnboundLocalError."""
    col = np.array(["a", "b"] * 20, dtype=object)
    assert hi._is_discrete_v2(col) is True


def test_nan_vote_score_ranks_last_not_as_zero(monkeypatch):
    """A NaN aggregate score must sort after every finite score, including negative ones."""

    class _FakeLeaderboard:
        """Leaderboard stand-in returning fixed scores."""

        def __init__(self, *a, **k):
            """Ignore the table."""

        def borda_ranking(self):
            """Scores with one NaN, one negative, one positive."""
            return pd.Series({"nan_f": np.nan, "neg_f": -1.0, "pos_f": 2.0})

    monkeypatch.setattr(hi, "Leaderboard", _FakeLeaderboard)
    out = hi.get_actual_features_ranking({"r1": {"nan_f": 1.0, "neg_f": 2.0, "pos_f": 3.0}}, VotesAggregation.Borda)
    assert out == ["pos_f", "neg_f", "nan_f"]


class _FlakyModel:
    """score() succeeds on the baseline call and raises afterwards."""

    def __init__(self):
        """Start with no calls."""
        self.calls = 0

    def score(self, X, y):
        """Baseline ok, permuted calls raise."""
        self.calls += 1
        if self.calls == 1:
            return 0.5
        raise RuntimeError("boom")


@pytest.mark.parametrize("p", [1, 3])
def test_cpi_score_failure_is_logged_at_warning(caplog, p):
    """A model.score() crash on permuted X must be visible at WARNING, not debug."""
    reset_throttle_counts()
    rng = np.random.default_rng(0)
    X = rng.random((150, p))
    with caplog.at_level(logging.WARNING, logger=hi.logger.name):
        imp = hi._conditional_permutation_importance(_FlakyModel(), X, rng.random(150), n_repeats=2)
    assert np.isnan(imp).all()
    assert any("model.score() failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)


class _OkModel:
    """Constant-score model."""

    def score(self, X, y):
        """Always 0.5."""
        return 0.5


def test_cpi_conditioning_tree_failure_is_logged_at_warning(caplog):
    """A conditioning-tree fit failure (NaN in the regressed column) must be visible at WARNING."""
    reset_throttle_counts()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 3))
    X[0, 0] = np.nan
    with caplog.at_level(logging.WARNING, logger=hi.logger.name):
        imp = hi._conditional_permutation_importance(_OkModel(), X, rng.random(150), n_repeats=1)
    assert np.isnan(imp[0])
    assert any("conditioning tree fit failed" in r.getMessage() and r.levelno == logging.WARNING for r in caplog.records)
