"""Meta-tests for the scenario library: the anti-rigging mechanics, and each bed's declared property.

The library is written by the author of one of the methods it judges, so the tests that matter most here are
not about arithmetic. They check that the suite CAN produce a negative result: that every arm is expected to
fail somewhere, that the registry on disk is the registry in the code, and that each bed actually has the
property its entry claims -- a bed whose parity is not parity, or whose null is not null, would let a method
pass a test it never took.
"""

from __future__ import annotations

import collections
from typing import Dict

import numpy as np
import pytest

from mlframe.data.datasets import scenarios
from mlframe.data.datasets.generator import generate
from mlframe.data.datasets.ground_truth import PRIMARY_TARGET_SET

# Small enough to keep the file quick, large enough that a marginal association of 0.05 is not noise.
SMOKE_ROWS = 2500


@pytest.fixture(scope="module")
def datasets() -> Dict[str, object]:
    """Generate every registered scenario once at a reduced row count."""
    out: Dict[str, object] = {}
    for name in scenarios.names():
        spec = scenarios.get(name).build(seed=0)
        out[name] = generate(spec.model_copy(update={"n_samples": SMOKE_ROWS}))
    return out


class TestRegistryDiscipline:
    """The mechanics that stop a hand-picked suite from being unfalsifiable."""

    def test_every_scenario_declares_a_purpose_and_a_key(self) -> None:
        """A bed with no stated purpose cannot be argued with, and one with no key scores against nothing."""
        for scenario in scenarios.SCENARIOS:
            assert scenario.purpose.strip(), scenario.name
            assert scenario.primary_target_set == PRIMARY_TARGET_SET, scenario.name

    def test_every_named_arm_is_expected_to_break_somewhere_twice(self) -> None:
        """One bed per arm is a coincidence; two is a claim the suite can be held to."""
        counts = collections.Counter(arm for scenario in scenarios.SCENARIOS for arm in scenario.expected_to_break)
        thin = {arm: count for arm, count in counts.items() if count < 2}
        assert not thin, f"arms expected to break in fewer than two beds: {thin}"

    def test_the_suite_can_produce_a_negative_result(self) -> None:
        """A library where nothing is expected to fail cannot report a failure."""
        assert any(scenario.expected_to_break for scenario in scenarios.SCENARIOS)

    def test_the_committed_lock_matches_the_code(self) -> None:
        """A bed edited after its lock was committed is a post-hoc change, and must be visible as one."""
        assert scenarios.lock_differences() == []

    def test_an_unknown_name_fails_loudly(self) -> None:
        """A typo in a scenario name is a missing bed, not an empty run."""
        with pytest.raises(KeyError, match="unknown scenario"):
            scenarios.get("no_such_bed")


class TestBedsHaveTheClaimedProperty:
    """Each family's defining property, checked on the generated data rather than assumed."""

    @pytest.mark.parametrize("name", ["null_p100", "null_p1000"])
    def test_null_beds_are_actually_null(self, name: str, datasets: Dict[str, object]) -> None:
        """Chance-level ceiling and an empty answer key: anything selected here is a false positive."""
        dataset = datasets[name]
        assert dataset.calibration["bayes_auc"] == pytest.approx(0.5, abs=1e-9)  # type: ignore[attr-defined]
        assert dataset.truth.primary_target_set().members == ()  # type: ignore[attr-defined]

    def test_the_parity_bed_has_no_marginal_signal(self, datasets: Dict[str, object]) -> None:
        """If an operand were marginally visible, the bed would not test what it claims to test."""
        dataset = datasets["xor3"]
        for name in ("p0", "p1", "p2"):
            corr = abs(float(np.corrcoef(dataset.frame[name].to_numpy(), dataset.truth.true_prob)[0, 1]))  # type: ignore[attr-defined]
            assert corr < 0.06, f"{name} is marginally visible at {corr:.3f}"

    def test_the_decoy_bed_gives_a_marginal_method_something_to_take(self, datasets: Dict[str, object]) -> None:
        """The decoy must be genuinely attractive, or the bed cannot distinguish the two failure modes."""
        dataset = datasets["xor3_plus_marginal_decoy"]
        decoy = abs(float(np.corrcoef(dataset.frame["decoy"].to_numpy(), dataset.truth.true_prob)[0, 1]))  # type: ignore[attr-defined]
        operand = abs(float(np.corrcoef(dataset.frame["p0"].to_numpy(), dataset.truth.true_prob)[0, 1]))  # type: ignore[attr-defined]
        assert decoy > 10 * operand

    def test_exact_redundancy_is_recorded_as_collapsible(self, datasets: Dict[str, object]) -> None:
        """One representative suffices, and the truth record has to say so, or scoring will demand all five."""
        groups = datasets["redundant_exact_k5"].truth.redundancy_groups  # type: ignore[attr-defined]
        assert len(groups) == 1 and groups[0].exact is True and groups[0].rank == 1

    def test_private_delta_redundancy_is_recorded_as_not_collapsible(self, datasets: Dict[str, object]) -> None:
        """The same cluster shape, the opposite correct behaviour; the record is what separates them."""
        groups = datasets["latent_replicates_private_delta"].truth.redundancy_groups  # type: ignore[attr-defined]
        assert len(groups) == 1 and groups[0].exact is False and groups[0].rank > 1

    def test_the_two_redundancy_beds_look_the_same_to_a_correlation_matrix(self, datasets: Dict[str, object]) -> None:
        """This is why a correlation-clustering selector gets one of them wrong: it cannot tell them apart."""
        exact = datasets["redundant_exact_k5"].frame[["c0", "c1", "c2"]].corr().to_numpy()  # type: ignore[attr-defined]
        private = datasets["latent_replicates_private_delta"].frame[["r0", "r1", "r2"]].corr().to_numpy()  # type: ignore[attr-defined]
        off_diagonal = ~np.eye(3, dtype=bool)
        assert exact[off_diagonal].min() > 0.5
        assert private[off_diagonal].min() > 0.5

    def test_the_spouse_is_in_the_blanket_and_the_cause_is_too(self, datasets: Dict[str, object]) -> None:
        """The bed exists for the spouse: a blanket member no marginal method can reach."""
        members = set(datasets["mb_spouse_collider"].truth.primary_target_set().members)  # type: ignore[attr-defined]
        assert {"x_cause", "spouse", "collider"} <= members

    def test_the_spouse_is_marginally_invisible(self, datasets: Dict[str, object]) -> None:
        """Blanket membership through a collider is exactly the case a marginal ranking cannot see."""
        dataset = datasets["mb_spouse_collider"]
        corr = abs(float(np.corrcoef(dataset.frame["spouse"].to_numpy(), dataset.truth.true_prob)[0, 1]))  # type: ignore[attr-defined]
        assert corr < 0.06

    def test_the_spouse_becomes_visible_once_the_collider_is_conditioned_on(self, datasets: Dict[str, object]) -> None:
        """The whole point of the bed, and the check that the declared edges were actually realised.

        A previous version drew the collider as an independent column: the graph claimed a collider, the
        data held noise, the spouse was marginally invisible for the trivial reason that it was unrelated to
        anything, and every structural assertion still passed. The partial coefficient is what distinguishes
        a realised collider from a declared one.
        """
        dataset = datasets["mb_spouse_collider"]
        y = np.asarray(dataset.target)  # type: ignore[attr-defined]
        spouse = dataset.frame["spouse"].to_numpy()  # type: ignore[attr-defined]
        collider = dataset.frame["collider"].to_numpy()  # type: ignore[attr-defined]

        marginal = abs(float(np.corrcoef(spouse, y)[0, 1]))
        design = np.column_stack([spouse, collider, np.ones(y.size)])
        partial = float(np.linalg.lstsq(design, y.astype(float), rcond=None)[0][0])
        assert marginal < 0.05, f"the spouse should be marginally invisible, got {marginal:.3f}"
        assert abs(partial) > 0.15, f"the spouse should be informative given the collider, got {partial:.3f}"

    def test_the_mediator_chain_is_a_realised_chain(self, datasets: Dict[str, object]) -> None:
        """Cause, mechanism and proxy must all be associated with the target, or the bed is three probes."""
        dataset = datasets["mediator_chain_with_proxy"]
        y = np.asarray(dataset.target).astype(float)  # type: ignore[attr-defined]
        correlations = {name: abs(float(np.corrcoef(dataset.frame[name].to_numpy(), y)[0, 1])) for name in ("x_cause", "mediator", "proxy")}  # type: ignore[attr-defined]
        assert all(value > 0.2 for value in correlations.values()), correlations
        assert correlations["mediator"] > correlations["x_cause"], correlations

    def test_the_mediator_screens_off_its_cause(self, datasets: Dict[str, object]) -> None:
        """Under the blanket the mediator suffices, which is what makes the answer key decisive here."""
        members = set(datasets["mediator_chain_with_proxy"].truth.primary_target_set().members)  # type: ignore[attr-defined]
        assert "mediator" in members
        assert "x_cause" not in members


class TestCalibration:
    """Every non-null bed ships the difficulty it declares."""

    @pytest.mark.parametrize("name", ["linear_k5_p50", "redundant_exact_k5", "latent_replicates_private_delta", "xor3", "mb_spouse_collider"])
    def test_the_declared_ceiling_is_reached(self, name: str, datasets: Dict[str, object]) -> None:
        """The calibration bisects on the shipped probabilities, so the declared number must hold."""
        dataset = datasets[name]
        requested = dataset.calibration["requested"]  # type: ignore[attr-defined]
        assert requested is not None
        assert dataset.calibration["bayes_auc"] == pytest.approx(float(requested), abs=0.02)  # type: ignore[attr-defined]

    def test_a_scenario_is_reproducible_from_its_name_alone(self) -> None:
        """Two builds of the same scenario at the same seed hash identically, or the lock means nothing."""
        first = scenarios.get("xor3").build(seed=0)
        second = scenarios.get("xor3").build(seed=0)
        assert first.content_hash() == second.content_hash()

    def test_a_different_seed_keeps_the_structure_and_changes_the_draw(self) -> None:
        """The seed is data, not structure: only the root seed field may differ between two seeds' specs."""
        first = scenarios.get("xor3").build(seed=0)
        second = scenarios.get("xor3").build(seed=1)
        assert first.content_hash() != second.content_hash()
        assert first.model_dump(exclude={"root_seed"}) == second.model_dump(exclude={"root_seed"})
