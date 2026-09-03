"""The removal search used one variable as both the running maximum and the acceptance bar.

`best_score` started at `current_score` and was raised by every accepted candidate, so a candidate was compared
against the best one BEFORE it rather than against the current feature set. With `tol > 0` that made the search
drop whichever removal cleared the bar first instead of the argmax, and -- because "first" is column order --
the surviving feature set depended on the order of `X.columns`. Two frames holding the same features in a
different order returned different selections.

`tol` documents a minimum improvement over the CURRENT set; it is a property of the step, not a handicap
applied between candidates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

from mlframe.feature_selection.greedy_backward_elimination import greedy_backward_elimination

# Score for each column subset, keyed by the set of surviving columns. Removing "q" is the best move by a clear
# margin, but "p" is examined first and clears the bar, after which "q" no longer beats "p" by `tol`.
_SCORES = {
    frozenset("pqrs"): 0.0,
    frozenset("qrs"): 1.0,  # dropping p
    frozenset("prs"): 1.3,  # dropping q -- the argmax
    frozenset("pqs"): 0.5,  # dropping r
    frozenset("pqr"): 0.2,  # dropping s
}

_TOL = 0.4


class _SubsetScorer(BaseEstimator):
    """Predicts a constant that depends only on WHICH columns it was fitted on, so the CV score is a lookup."""

    def fit(self, X, y):
        """Record the column subset; there is nothing to learn."""
        self.cols_ = frozenset(X.columns)
        return self

    def predict(self, X):
        """The subset's tabulated score, repeated for every row."""
        return np.full(len(X), _SCORES[frozenset(X.columns)])


def _scoring(y_true, y_pred):
    """Hand the tabulated constant straight back; higher is better, as the selector expects."""
    return float(y_pred[0])


def _frame(order: str = "pqrs") -> pd.DataFrame:
    """Six rows of noise; only the column NAMES matter to `_SubsetScorer`."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.random(6) for c in order})


def _run(order: str = "pqrs", tol: float = _TOL, scores=None):
    """One elimination round (`min_features=3` on four columns), returning the survivors."""
    global _SCORES
    saved = _SCORES
    if scores is not None:
        _SCORES = scores
    try:
        return greedy_backward_elimination(
            _SubsetScorer(),
            _frame(order),
            np.arange(6, dtype=float),
            _scoring,
            cv=KFold(n_splits=2),
            min_features=3,
            tol=tol,
        )
    finally:
        _SCORES = saved


class TestTheBestRemovalWins:
    """The defect, stated directly."""

    def test_it_drops_the_column_whose_removal_scores_highest(self):
        """Dropping "q" scores 1.3 and dropping "p" scores 1.0, so "q" is the move."""
        assert set(_run()) == {"p", "r", "s"}, "a lower-scoring removal was preferred because it was examined first"

    def test_the_result_does_not_depend_on_column_order(self):
        """The sharpest form: same features, different order, and the old code answered differently."""
        assert set(_run("pqrs")) == set(_run("srqp")) == set(_run("rpsq"))

    def test_every_permutation_agrees(self):
        """Order-independence is not a property of two lucky orders."""
        import itertools

        results = {frozenset(_run("".join(p))) for p in itertools.permutations("pqrs")}
        assert len(results) == 1, f"the survivors vary with column order: {sorted(map(sorted, results))}"


class TestTolStillGates:
    """`tol` must keep meaning "improve on the current set by this much", which the fix must not loosen."""

    def test_no_removal_is_accepted_when_none_clears_tol(self):
        """The best available move is +0.3 against a tol of 0.4, so the set is returned intact."""
        weak = {frozenset("pqrs"): 0.0, frozenset("qrs"): 0.3, frozenset("prs"): 0.2, frozenset("pqs"): 0.1, frozenset("pqr"): 0.0}
        assert set(_run(scores=weak)) == set("pqrs")

    def test_an_exactly_equal_improvement_is_not_enough(self):
        """`tol` is a strict bar; +0.4 against tol 0.4 does not clear it."""
        borderline = {frozenset("pqrs"): 0.0, frozenset("qrs"): 0.4, frozenset("prs"): 0.4, frozenset("pqs"): 0.4, frozenset("pqr"): 0.4}
        assert set(_run(scores=borderline)) == set("pqrs")

    def test_with_the_default_tol_any_improvement_is_taken(self):
        """The documented `tol=0.0` behaviour, still driven by the argmax."""
        assert set(_run(tol=0.0)) == {"p", "r", "s"}

    def test_a_strictly_worse_set_is_never_returned(self):
        """Every removal hurts, so the full set survives regardless of tol."""
        harmful = {frozenset("pqrs"): 1.0, frozenset("qrs"): 0.1, frozenset("prs"): 0.2, frozenset("pqs"): 0.3, frozenset("pqr"): 0.4}
        assert set(_run(tol=0.0, scores=harmful)) == set("pqrs")


def test_min_features_is_still_respected():
    """The loop bound is unchanged by the fix; a guard against trading one bug for another."""
    assert len(_run()) == 3


@pytest.mark.parametrize("order", ["pqrs", "qpsr", "sqrp"])
def test_the_survivors_are_returned_in_the_frames_own_order(order):
    """Documented contract: "in original order"."""
    survivors = _run(order)
    assert survivors == [c for c in order if c in set(survivors)]
