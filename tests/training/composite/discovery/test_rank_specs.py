"""Spec rankings go through ``rank_specs``, which orders one kind of score and refuses to mix kinds.

The cross-target budget once sorted RMSE fractions against MI nats, and the per-target equivalence dedup still did: of
two equivalent composites it kept the one with the larger raw gain, whichever unit that gain was in.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlframe.training.composite.discovery._score import MixedScoresError, Score, rank_specs


class _Spec:
    """A stand-in composite spec: a name and a transform name, all rank_specs reads."""
    def __init__(self, name, transform="diff"):
        self.name, self.transform_name = name, transform


def test_one_kind_ranks_with_non_finite_last_and_the_chosen_tie_rule():
    """Scores of one kind sort ascending (or descending), non-finite last, ties broken by the chosen rule."""
    specs = [_Spec("b"), _Spec("a"), _Spec("c"), _Spec("d")]
    vals = {"a": 1.0, "b": 1.0, "c": math.nan, "d": 0.5}
    key = lambda s: Score(vals[s.name], "y_rmse", "cv", "tiny_cv", s.transform_name)
    assert [s.name for s in rank_specs(specs, key)] == ["d", "a", "b", "c"]
    assert [s.name for s in rank_specs(specs, key, tiebreak=None)] == ["d", "b", "a", "c"]
    assert [s.name for s in rank_specs(specs, key, descending=True)] == ["a", "b", "d", "c"]


@pytest.mark.parametrize("other", [Score(1.0, "mi_nats", "cv", "tiny_cv"), Score(1.0, "y_rmse", "honest_holdout", "tiny_cv"),
                                   Score(1.0, "y_rmse", "cv", "honest_rmse")])
def test_a_second_unit_split_or_estimator_is_refused(other):
    """Scores that differ in unit, split or estimator cannot be ranked together."""
    specs = [_Spec("a"), _Spec("b")]
    with pytest.raises(MixedScoresError):
        rank_specs(specs, lambda s: Score(1.0, "y_rmse", "cv", "tiny_cv") if s.name == "a" else other)


def test_a_score_measured_for_another_transform_is_refused():
    """A score measured for a different transform than the spec's is refused."""
    with pytest.raises(MixedScoresError, match="measured for"):
        rank_specs([_Spec("multi", "linear_residual_multi")], lambda s: Score(0.2, "mi_nats", "screen", "mi_gain", "linear_residual"))


def test_the_equivalence_dedup_keeps_the_budgets_choice_not_the_larger_raw_gain():
    """An honest-RMSE composite outranks an MI-only one in the budget; of an equivalent pair the dedup keeps it."""
    from mlframe.training.core._phase_composite_discovery_dedup import prune_equivalent_composite_specs

    rng = np.random.default_rng(0)
    n = 500
    y = rng.normal(size=n)
    t_a = y - rng.normal(size=n)
    t_by_name = {"A": t_a, "B": 2.0 * t_a + 1.0}
    pending = [{"tt": "regression", "name": "A", "gain": 0.05, "rmse_gain": 0.05},  # RMSE fraction, honest holdout
               {"tt": "regression", "name": "B", "gain": 0.30, "rmse_gain": None}]  # MI nats
    drops = prune_equivalent_composite_specs(specs=[{"name": "A"}, {"name": "B"}], t_by_name=t_by_name, y_full=y, train_idx=None,
                                             pending=pending, metadata={}, target_type="regression", target_name="y")
    assert set(drops) == {"B"}, drops
