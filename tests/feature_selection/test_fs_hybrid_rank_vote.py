"""The rank-aggregation arm: does combining several rankings beat the best one, or is that just plausible?

The claim needs the components in the same leaderboard as the aggregate, which is what putting this in the
roster buys. These tests pin the properties that make the comparison fair -- full coverage, unlike base
scorers, an abstaining scorer that does not vote rather than voting for everything equally -- not the
outcome, which is what the run is for.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection._benchmarks.fs_hybrid._arms import build_arm_roster
from mlframe.feature_selection._benchmarks.fs_hybrid._arms_rank_aggregation import COPELAND_MAX_FEATURES, VOTING_RULES, RankAggregationArm, aggregate_ranks, base_score_table

ROWS = 500


def _bed(seed: int = 0) -> Any:
    """Return `(X, y)` with three informative columns and nine probes."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"s{i}": rng.normal(size=ROWS) for i in range(3)})
    for index in range(9):
        frame[f"n{index:02d}"] = rng.normal(size=ROWS)
    score = 1.6 * frame["s0"] + 1.1 * frame["s1"] + 0.8 * frame["s2"]
    labels = (rng.random(ROWS) < 1.0 / (1.0 + np.exp(-score))).astype(np.int64)
    return frame, labels


def test_the_score_table_has_one_row_per_feature_and_several_unlike_scorers() -> None:
    """A vote over scorers that agree everywhere is a slower copy of any one of them."""
    frame, labels = _bed()

    table = base_score_table(frame, labels, random_state=0)

    assert list(table.index) == [str(column) for column in frame.columns]
    assert table.shape[1] >= 3, "too few base scorers for a vote to mean anything"
    correlations = table.corr(method="spearman").to_numpy()
    off_diagonal = correlations[~np.eye(correlations.shape[0], dtype=bool)]
    assert float(np.min(off_diagonal)) < 0.99, "every base scorer produces the same ranking, so the vote is decorative"


def test_a_scorer_that_cannot_run_abstains_rather_than_voting_for_everything_equally(monkeypatch: pytest.MonkeyPatch) -> None:
    """A column of zeros is a scorer ranking every feature the same, which dilutes every real opinion."""
    import mlframe.feature_selection._benchmarks.fs_hybrid._arms_rank_aggregation as module

    def _broken(*_args: Any, **_kwargs: Any) -> np.ndarray:
        """Stand in for a scorer that fails on this bed."""
        raise RuntimeError("this scorer cannot run here")

    monkeypatch.setattr(module, "_tree_importance", _broken)
    frame, labels = _bed()

    table = base_score_table(frame, labels, random_state=0)

    assert "tree" not in table.columns
    assert table.shape[1] >= 2


def test_every_scorer_failing_is_an_error_rather_than_an_empty_vote(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty table would aggregate to a ranking of nothing, which is worse than refusing."""
    import mlframe.feature_selection._benchmarks.fs_hybrid._arms_rank_aggregation as module

    def _broken(*_args: Any, **_kwargs: Any) -> np.ndarray:
        """Stand in for a scorer that fails on this bed."""
        raise RuntimeError("cannot run")

    for name in ("_f_statistic", "_mutual_information", "_abs_spearman", "_tree_importance"):
        monkeypatch.setattr(module, name, _broken)
    frame, labels = _bed()

    with pytest.raises(RuntimeError, match="nothing to aggregate"):
        base_score_table(frame, labels, random_state=0)


@pytest.mark.parametrize("rule", VOTING_RULES)
def test_every_rule_ranks_every_feature(rule: str) -> None:
    """Partial coverage would make the aggregate incomparable to the filters it is built from."""
    frame, labels = _bed()
    table = base_score_table(frame, labels, random_state=0)

    scores, used = aggregate_ranks(table, rule)

    assert used == rule
    assert set(scores.index) == set(table.index)


def test_an_unknown_rule_is_refused_rather_than_defaulted() -> None:
    """Reporting one aggregation under another's name is exactly what this arm exists to compare."""
    frame, labels = _bed()

    with pytest.raises(ValueError, match="unknown voting rule"):
        aggregate_ranks(base_score_table(frame, labels), "instant-runoff")


def test_copeland_falls_back_above_its_width_cap_and_says_so() -> None:
    """A pairwise majority graph is quadratic; running it silently for an hour is not an option."""
    rng = np.random.default_rng(0)
    wide = pd.DataFrame(rng.normal(size=(3, COPELAND_MAX_FEATURES + 5)).T, columns=["a", "b", "c"])

    _scores, used = aggregate_ranks(wide, "copeland")

    assert used == "borda", "the fallback happened silently or not at all"


def test_the_arm_reports_a_continuous_score_covering_every_feature() -> None:
    """`continuous` is a promise about coverage; an arm declaring it without a full score vector is fatal."""
    frame, labels = _bed()

    result = RankAggregationArm(k=4, rule="borda", random_state=0).run(frame, labels)

    assert result.score_kind == "continuous"
    assert result.score is not None and result.score.shape == (frame.shape[1],)
    assert np.all(np.isfinite(result.score))


def test_the_arm_keeps_exactly_its_budget() -> None:
    """A matched-K comparison is only matched if the arm honours the budget it was given."""
    frame, labels = _bed()

    result = RankAggregationArm(k=4, rule="borda", random_state=0).run(frame, labels)

    assert int(np.asarray(result.support, dtype=bool).sum()) == 4


def test_the_arm_finds_the_informative_columns_on_a_bed_every_component_can_solve() -> None:
    """Not a claim the vote is better -- a floor. An aggregate losing to its own components here is broken."""
    frame, labels = _bed()
    names = [str(column) for column in frame.columns]

    result = RankAggregationArm(k=3, rule="borda", random_state=0).run(frame, labels)
    selected = {names[index] for index, keep in enumerate(np.asarray(result.support, dtype=bool)) if keep}

    assert len(selected & {"s0", "s1", "s2"}) >= 2, f"the aggregate kept {sorted(selected)} on a bed with three obvious signals"


def test_the_arm_records_which_rule_it_actually_used() -> None:
    """The requested rule and the used one differ at width, and a report cannot tell them apart otherwise."""
    frame, labels = _bed()

    provenance = RankAggregationArm(k=3, rule="dowdall", random_state=0).run(frame, labels).provenance

    assert provenance["rule_requested"] == "dowdall"
    assert provenance["rule_used"] == "dowdall"
    assert provenance["scorers"]


def test_the_arm_is_in_the_roster_so_its_components_are_beside_it() -> None:
    """The comparison is the whole point: an aggregate benchmarked apart from its parts proves nothing."""
    roster = build_arm_roster(20, k=5, random_state=0)

    assert "rank-vote" in roster
    for component in ("skb-f", "univariate-mi"):
        assert component in roster, f"the vote's component {component!r} is not in the same roster"


def test_the_arm_charges_itself_for_the_one_model_it_fits() -> None:
    """Three of its scorers fit no model and one fits a forest; reporting zero would make it look free."""
    frame, labels = _bed()

    result = RankAggregationArm(k=3, rule="borda", random_state=0).run(frame, labels)

    assert result.n_model_fits is not None and result.n_model_fits >= 1
