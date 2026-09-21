"""The tripwire: an unsupervised control must not be able to recover any bed's answer key.

Sorting columns by their raw variance never looks at the target. In an additively generated structural
causal model it nonetheless recovers the causal order, because variance accumulates with depth in
topological order (Reisach, Seiler and Drton, NeurIPS 2021, "Beware of the Simulated DAG"). A generator
that leaves this channel open produces beds where every method appears to succeed and the success is an
artefact of how the data was drawn.

The generator standardises every column to unit variance to close the channel. This is the check that it
stayed closed -- across every registered bed, not just the ones anyone remembered to look at. A bed where
the control wins is a broken bed, and the failure names it.

A bed may declare an exemption in the registry, with a reason, when its unequal scales are part of a
published design rather than an accident. No bed currently needs one, and the exemptions that were
registered on that assumption did not survive measurement. Declared exemptions are checked too: one that
turns out to be unnecessary is a claim nobody verified, and it would quietly widen the hole for the next
bed that copies it.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

from mlframe.data.datasets import scenarios as scenario_registry
from mlframe.data.datasets.generator import generate

#: Rows per bed. Small enough that seventeen beds are one fast test, large enough that a variance ranking is
#: not itself noise.
ROWS = 1500

#: Seeds averaged over. One seed is not enough and the first version of this test used one: when every
#: column is standardised to unit variance the ranking is decided by floating-point noise, so on a bed with
#: two informative columns out of thirty-two a single draw picks one of them by luck about one time in eight.
#: That reads as a leak. Averaging over several seeds separates luck from an actual scale channel, which is
#: the only thing this check is about.
SEEDS = (0, 1, 2, 3, 4)

#: How far above the base rate a variance ranking may score before it counts as recovery. Generous: the
#: question is "does the control WIN", not "is it exactly at chance", and a tight bound would turn this
#: tripwire into a flaky test that people learn to ignore.
RECOVERY_MARGIN = 0.25


def _recovery_at(name: str, seed: int) -> Tuple[float, float]:
    """Return ``(recovered_share, base_rate)`` for one seed of one bed."""
    scenario = scenario_registry.get(name)
    takes_rows = "n_samples" in scenario.builder.__code__.co_varnames
    spec = scenario.build(seed=seed, n_samples=ROWS) if takes_rows else scenario.build(seed=seed)
    frame = generate(spec).frame
    answer = {str(edge.source) for edge in spec.edges}
    if not answer:
        return 0.0, 0.0

    numeric = frame.select_dtypes(include=["number"])
    top = set(numeric.var(axis=0).sort_values(ascending=False).index[: len(answer)].astype(str))
    return float(len(top & answer) / len(answer)), float(len(answer) / max(numeric.shape[1], 1))


def _variance_recovery(name: str) -> Tuple[float, float]:
    """Return ``(mean_recovered_share, base_rate)`` for ranking a bed's columns by raw variance.

    The control takes the top ``k`` columns by variance, where ``k`` is the size of the bed's declared
    answer key, and reports what share of that key it captured. The base rate is what the same-sized random
    selection would capture, so the two are directly comparable. Averaged over seeds, because a single draw
    of a bed whose columns all carry unit variance is a coin toss rather than a measurement.
    """
    measured = [_recovery_at(name, seed) for seed in SEEDS]
    return float(np.mean([pair[0] for pair in measured])), float(np.mean([pair[1] for pair in measured]))


@pytest.mark.parametrize("name", sorted(scenario_registry.names()))
def test_variance_sorting_does_not_recover_the_answer_key(name: str) -> None:
    """Every bed that has not declared itself varsortable must keep the unsupervised control at chance."""
    scenario = scenario_registry.get(name)
    recovered, base_rate = _variance_recovery(name)

    if scenario.varsortable:
        pytest.skip(f"{name} declares varsortable=True: {scenario.varsortable_reason}")
    assert recovered <= base_rate + RECOVERY_MARGIN, f"{name}: variance sorting recovered {recovered:.3f} of the answer key against a base rate of {base_rate:.3f}, so the bed leaks its truth through column scale"


def test_declared_varsortable_beds_really_are_varsortable() -> None:
    """An exemption nobody verified is a hole the next bed copies without needing a reason of its own.

    No bed currently needs one, which was itself a measured correction: Friedman's second and third
    functions were registered as exempt on the assumption that their published four-orders-of-magnitude
    range imbalance would leak the answer key through column scale. It does not, because their probes are
    drawn over the same wide range as their informative columns -- variance sorting there recovers 0.25 of
    the key against a 0.40 base rate, which is WORSE than chance. The exemptions were withdrawn.
    """
    declared = [scenario.name for scenario in scenario_registry.SCENARIOS if scenario.varsortable]

    for name in declared:
        recovered, base_rate = _variance_recovery(name)
        assert recovered > base_rate, f"{name} claims a varsortability exemption it does not need (recovered {recovered:.3f} against base rate {base_rate:.3f}); an unnecessary exemption is an unchecked one"


def test_varsortable_flag_requires_a_reason() -> None:
    """The flag waives the suite's strongest anti-rigging check, so it may not be set silently."""
    from mlframe.data.datasets.scenarios import Scenario
    from mlframe.data.datasets.scenarios._null import null_spec

    with pytest.raises(ValueError, match="without a reason"):
        Scenario(name="x", family="null", builder=null_spec, expected_to_break=(), purpose="p", varsortable=True)


def test_every_registered_bed_names_arms_it_expects_to_defeat() -> None:
    """A bed expecting nothing to break cannot produce a negative result, which is what this suite is for."""
    silent: List[str] = [scenario.name for scenario in scenario_registry.SCENARIOS if not scenario.expected_to_break]

    assert not silent, f"these beds declare no expectations, so their forecast cannot be scored: {silent}"


def test_every_arm_named_anywhere_is_named_by_at_least_two_beds() -> None:
    """One bed naming an arm is an anecdote; the pre-registration requires two before a prediction counts."""
    counts: dict = {}
    for scenario in scenario_registry.SCENARIOS:
        for arm in scenario.expected_to_break:
            counts[arm] = counts.get(arm, 0) + 1

    lonely = sorted(arm for arm, count in counts.items() if count < 2)
    assert not lonely, f"these arms are predicted to break on only one bed each: {lonely}"
