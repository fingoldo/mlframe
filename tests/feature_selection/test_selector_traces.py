"""Three selectors computed a ranking on every round and returned a bare set.

Each of these wrappers evaluates candidates, picks the best, and throws the comparison away. The set it
returns carries no order, so every ranking metric -- average precision, recovery at a matched budget,
anything that reads more than membership -- has to skip the selector entirely. The information was there;
nothing kept it.

The traces are opt-in (`return_trace=True`) so no existing caller changes shape, and each test below pins
both halves of that: the trace says what the rounds did, and the plain call still returns exactly what it
returned before.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mlframe.feature_selection.greedy_backward_elimination import EliminationStep, greedy_backward_elimination
from mlframe.feature_selection.zero_importance_pruning import PruningRound, iterative_zero_importance_pruning

ROWS = 400


def _bed(seed: int = 0, dead_columns: int = 0) -> Any:
    """Return `(X, y)` with two informative columns, some probes, and optionally some constant ones."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"c{i}": rng.normal(size=ROWS) for i in range(6)})
    for index in range(dead_columns):
        frame[f"dead{index}"] = 0.0
    score = 1.5 * frame["c0"] + 1.0 * frame["c1"]
    labels = (rng.random(ROWS) < 1.0 / (1.0 + np.exp(-score))).astype(np.int64)
    return frame, labels


def test_backward_elimination_still_returns_a_bare_set_by_default() -> None:
    """The trace is opt-in; a caller that did not ask for it must see exactly what it saw before."""
    frame, labels = _bed()

    kept = greedy_backward_elimination(LogisticRegression(max_iter=500), frame, labels, roc_auc_score, min_features=2)

    assert isinstance(kept, list)
    assert all(not isinstance(item, EliminationStep) for item in kept)


def test_backward_elimination_trace_does_not_change_what_is_kept() -> None:
    """Asking for the trace must not change the search, or the trace would describe a different run."""
    frame, labels = _bed()

    plain = greedy_backward_elimination(LogisticRegression(max_iter=500), frame, labels, roc_auc_score, min_features=2)
    kept, _trace = greedy_backward_elimination(LogisticRegression(max_iter=500), frame, labels, roc_auc_score, min_features=2, return_trace=True)

    assert kept == plain


def test_backward_elimination_trace_records_one_step_per_accepted_removal() -> None:
    """A trace shorter or longer than the accepted removals would misstate the elimination order."""
    frame, labels = _bed()

    kept, trace = greedy_backward_elimination(LogisticRegression(max_iter=500), frame, labels, roc_auc_score, min_features=2, return_trace=True)

    assert len(trace) == frame.shape[1] - len(kept)
    assert all(isinstance(step, EliminationStep) for step in trace)


def test_backward_elimination_trace_counts_down_and_never_drops_a_survivor() -> None:
    """The trace IS the ranking: a survivor appearing in it would make the order contradict the result."""
    frame, labels = _bed()

    kept, trace = greedy_backward_elimination(LogisticRegression(max_iter=500), frame, labels, roc_auc_score, min_features=2, return_trace=True)

    dropped = [step.dropped for step in trace]
    assert len(set(dropped)) == len(dropped), "a column was recorded as dropped twice"
    assert not (set(dropped) & set(kept)), "a surviving column appears in the elimination trace"
    assert [step.n_remaining for step in trace] == sorted((step.n_remaining for step in trace), reverse=True)


def test_backward_elimination_drops_a_pure_probe_before_an_informative_column() -> None:
    """The order has to mean something: the earliest drops should be the columns that carry least.

    Not a claim that the selector is good -- a claim that the recorded order is the search's own order
    rather than column order, which is what makes it usable as a ranking.
    """
    frame, labels = _bed()

    _kept, trace = greedy_backward_elimination(LogisticRegression(max_iter=500), frame, labels, roc_auc_score, min_features=2, return_trace=True)

    assert len(trace) > 0, "the search dropped nothing, so there is no order to check"
    assert trace[0].dropped not in ("c0", "c1"), f"the first removal was {trace[0].dropped!r}, one of the two informative columns"


def test_pruning_still_returns_a_bare_set_by_default() -> None:
    """Same opt-in contract for the second selector."""
    frame, labels = _bed(dead_columns=2)

    kept = iterative_zero_importance_pruning(RandomForestClassifier(n_estimators=20, random_state=0), frame, labels, roc_auc_score, max_rounds=3)

    assert isinstance(kept, list)


def test_pruning_trace_does_not_change_what_is_kept() -> None:
    """The trace must observe the search, not steer it."""
    frame, labels = _bed(dead_columns=2)
    args = (RandomForestClassifier(n_estimators=20, random_state=0), frame, labels, roc_auc_score)

    plain = iterative_zero_importance_pruning(*args, max_rounds=3)
    kept, _trace = iterative_zero_importance_pruning(*args, max_rounds=3, return_trace=True)

    assert kept == plain


def test_pruning_trace_names_what_each_round_dropped() -> None:
    """A round reporting a size but not the columns cannot be turned into a ranking afterwards."""
    frame, labels = _bed(dead_columns=3)

    _kept, trace = iterative_zero_importance_pruning(RandomForestClassifier(n_estimators=20, random_state=0), frame, labels, roc_auc_score, max_rounds=3, return_trace=True)

    assert trace, "a bed with three constant columns must produce at least one pruning round"
    assert all(isinstance(round_, PruningRound) for round_ in trace)
    first_dropped = set(trace[0].dropped)
    assert {"dead0", "dead1", "dead2"} <= first_dropped, f"the constant columns were not dropped first; round one dropped {sorted(first_dropped)}"


def test_pruning_trace_marks_which_round_became_the_best_set() -> None:
    """The returned set is the BEST of the rounds, not the last, which is a maximum over noisy draws.

    Recording which rounds improved is what makes that optimism measurable rather than assumed: the
    winner's-curse column needs the internal optimum and the round it came from.
    """
    frame, labels = _bed(dead_columns=3)

    _kept, trace = iterative_zero_importance_pruning(RandomForestClassifier(n_estimators=20, random_state=0), frame, labels, roc_auc_score, max_rounds=4, return_trace=True)

    flags: List[bool] = [round_.became_best for round_ in trace]
    assert len(flags) > 0, "the pruning recorded no rounds"
    assert all(isinstance(flag, bool) for flag in flags)
    scores = [round_.score_after for round_ in trace]
    assert len(scores) == len(flags)
    assert all(np.isfinite(score) for score in scores), "a round recorded a non-finite score, which no comparison can use"


def test_pruning_rounds_shrink_the_surviving_set() -> None:
    """A round that dropped nothing should not be recorded, or the trace would imply progress it did not make."""
    frame, labels = _bed(dead_columns=2)

    _kept, trace = iterative_zero_importance_pruning(RandomForestClassifier(n_estimators=20, random_state=0), frame, labels, roc_auc_score, max_rounds=4, return_trace=True)

    assert trace, "the pruning run recorded no rounds"
    for round_ in trace:
        assert len(round_.dropped) > 0, "a recorded round dropped no columns"
    sizes = [round_.n_remaining for round_ in trace]
    assert sizes == sorted(sizes, reverse=True)


def test_rfecv_adapter_prefers_a_strict_consensus_order_over_the_all_ties_ranking() -> None:
    """`ranking_` gives EVERY survivor rank 1, and a cut over one enormous tie group is decided by the tie-break.

    That is not hypothetical: an earlier version of the atlas reported a wrapper recovering a parity bed
    perfectly on all twenty seeds, and the result was the tie-break inheriting the bed's column order.
    """
    from mlframe.feature_selection._benchmarks.fs_hybrid._arms import _rfecv_rank_vector

    names = ["a", "b", "c", "d"]

    class _Fitted:
        """A fitted selector exposing both attributes, with the all-ties one first in the old lookup order."""

        ranking_ = np.array([1, 1, 1, 2])
        consensus_ranking_ = ["c", "a", "d", "b"]

    ranks = _rfecv_rank_vector(_Fitted(), names, selected=["a", "b", "c"])

    assert len(set(ranks.tolist())) == len(names), "the consensus order was ignored and the all-ties ranking used instead"
    assert ranks[names.index("c")] < ranks[names.index("a")] < ranks[names.index("d")]


def test_rfecv_adapter_still_falls_back_when_there_is_no_consensus_order() -> None:
    """The ordinary path does not set the consensus attribute, and the adapter has to keep working there."""
    from mlframe.feature_selection._benchmarks.fs_hybrid._arms import _rfecv_rank_vector

    names = ["a", "b", "c"]

    class _Fitted:
        """A fitted selector from the path that sets only the sklearn-style ranking."""

        ranking_ = np.array([1, 1, 3])

    ranks = _rfecv_rank_vector(_Fitted(), names, selected=["a", "b"])

    assert ranks.tolist() == [1.0, 1.0, 3.0]


def test_rfecv_adapter_ignores_an_empty_consensus_order() -> None:
    """An attribute present but empty must not be preferred over a usable ranking."""
    from mlframe.feature_selection._benchmarks.fs_hybrid._arms import _rfecv_rank_vector

    names = ["a", "b"]

    class _Fitted:
        """A fitted selector whose consensus attribute exists but was never filled."""

        ranking_ = np.array([1, 2])
        consensus_ranking_: List[str] = []

    assert _rfecv_rank_vector(_Fitted(), names, selected=["a"]).tolist() == [1.0, 2.0]


@pytest.mark.parametrize("dead", [0, 2])
def test_neither_trace_leaks_into_the_returned_set(dead: int) -> None:
    """A trace object appearing among the selected columns would break every consumer downstream."""
    frame, labels = _bed(dead_columns=dead)

    kept, _trace = iterative_zero_importance_pruning(RandomForestClassifier(n_estimators=20, random_state=0), frame, labels, roc_auc_score, max_rounds=2, return_trace=True)

    assert all(isinstance(column, str) for column in kept)
