"""Behavioural tests for the synthetic dataset generator.

These check the properties the generator exists to provide, not its arithmetic: determinism, invariance to
editing a spec, a ceiling that means what it says once corruption is applied, and the two structures the
scenario library is built around -- private-delta redundancy, where averaging a cluster destroys signal, and
parity, where the operands are individually invisible.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest

from mlframe.data.datasets._target import bayes_auc, bayes_brier, shift_to_prevalence
from mlframe.data.datasets.generator import generate
from mlframe.data.datasets.spec import (
    CeilingTarget,
    DatasetSpec,
    EdgeSpec,
    FeatureSpec,
    LatentSpec,
    LinkSpec,
    NoiseSpec,
    TargetSpec,
)


def _linear_spec(name: str = "linear", n: int = 3000, seed: int = 0, extra: Tuple[FeatureSpec, ...] = (), insert_at: int = -1) -> DatasetSpec:
    """A small linear bed, optionally with extra columns inserted at a given position."""
    features = [FeatureSpec(name="x1"), FeatureSpec(name="x2"), FeatureSpec(name="n1"), FeatureSpec(name="n2")]
    if extra:
        position = len(features) if insert_at < 0 else insert_at
        features[position:position] = list(extra)
    return DatasetSpec(
        name=name,
        n_samples=n,
        root_seed=seed,
        features=tuple(features),
        targets=(
            TargetSpec(
                name="y",
                prevalence=0.35,
                link=LinkSpec(kind="logistic", coefficients={"x1": 1.0, "x2": 0.7}),
                calibrate_to=CeilingTarget(metric="auc", value=0.80),
            ),
        ),
        edges=(EdgeSpec(source="x1", target="y"), EdgeSpec(source="x2", target="y")),
    )


class TestDeterminism:
    """The generator is a pure function of its spec, and stays one when the spec is edited."""

    def test_same_spec_gives_bit_identical_data(self) -> None:
        """Two calls with the same spec return the same rows, not merely the same distribution."""
        first, second = generate(_linear_spec()), generate(_linear_spec())
        np.testing.assert_array_equal(first.frame.to_numpy(), second.frame.to_numpy())
        np.testing.assert_array_equal(first.target.to_numpy(), second.target.to_numpy())

    def test_appending_a_column_leaves_the_others_untouched(self) -> None:
        """Adding a feature at the end must not redraw the columns that were already there."""
        base = generate(_linear_spec())
        widened = generate(_linear_spec(extra=(FeatureSpec(name="added"),)))
        for column in base.frame.columns:
            np.testing.assert_array_equal(base.frame[column].to_numpy(), widened.frame[column].to_numpy())

    def test_inserting_a_column_in_the_middle_also_leaves_them_untouched(self) -> None:
        """Name-addressed streams: a mid-spec insertion is the case positional seeding gets wrong."""
        base = generate(_linear_spec())
        widened = generate(_linear_spec(extra=(FeatureSpec(name="inserted"),), insert_at=1))
        for column in base.frame.columns:
            np.testing.assert_array_equal(base.frame[column].to_numpy(), widened.frame[column].to_numpy())

    def test_a_different_root_seed_gives_different_data(self) -> None:
        """Determinism must not have collapsed into ignoring the seed."""
        base, other = generate(_linear_spec(seed=0)), generate(_linear_spec(seed=1))
        assert not np.array_equal(base.frame["x1"].to_numpy(), other.frame["x1"].to_numpy())


class TestVarsortability:
    """Standardisation is a defence against a control arm winning for the wrong reason."""

    def test_every_numeric_column_has_unit_variance(self) -> None:
        """Sorting by variance must carry no information about the causal order."""
        dataset = generate(_linear_spec())
        for column in dataset.frame.columns:
            assert dataset.frame[column].std(ddof=0) == pytest.approx(1.0, abs=0.05), column

    def test_the_removed_scale_is_recorded_not_discarded(self) -> None:
        """A scenario studying varsortability can put the scale back, so it must be kept."""
        dataset = generate(_linear_spec(extra=(FeatureSpec(name="wide", family="lognormal"),)))
        scale = dataset.truth.features["wide"].pre_standardization_scale
        assert scale is not None and scale > 1.0


class TestCeiling:
    """The declared difficulty has to be the difficulty of the data that ships."""

    def test_calibration_hits_the_requested_auc(self) -> None:
        """The link scale is bisected until the Bayes AUC of the realised probabilities matches."""
        dataset = generate(_linear_spec())
        assert dataset.calibration["achieved_auc"] == pytest.approx(0.80, abs=0.01)
        assert bayes_auc(dataset.truth.true_prob) == pytest.approx(0.80, abs=0.01)

    def test_label_noise_is_inside_the_calibration_not_after_it(self) -> None:
        """A bed that flips 5% of its labels still ships the ceiling it declared."""
        spec = _linear_spec(name="noisy")
        noisy = spec.model_copy(
            update={
                "targets": (spec.targets[0].model_copy(update={"noise": NoiseSpec(kind="uniform_flip", rate=0.05, true_prob_update="uniform_flip")}),),
            }
        )
        dataset = generate(noisy)
        assert bayes_auc(dataset.truth.true_prob) == pytest.approx(0.80, abs=0.01)

    def test_a_harder_target_needs_a_smaller_scale(self) -> None:
        """Difficulty is monotone in the calibrated scale, which is what makes the sweep readable."""
        easy = generate(_linear_spec())
        spec = _linear_spec(name="harder")
        harder = generate(spec.model_copy(update={"targets": (spec.targets[0].model_copy(update={"calibrate_to": CeilingTarget(metric="auc", value=0.65)}),)}))
        assert harder.calibration["scale"] < easy.calibration["scale"]

    def test_prevalence_is_reached_without_dropping_rows(self) -> None:
        """The intercept moves; the sample does not, so imbalance is not confounded with sample size."""
        dataset = generate(_linear_spec(n=4000))
        assert len(dataset.frame) == 4000
        assert float(dataset.target.mean()) == pytest.approx(0.35, abs=0.03)


class TestBayesQuantities:
    """The exact ceiling computation, on cases whose answers are known by hand."""

    def test_perfect_separation_scores_one(self) -> None:
        """Probabilities of zero and one rank perfectly."""
        assert bayes_auc(np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(1.0)

    def test_a_constant_probability_is_chance(self) -> None:
        """Every ordering is a tie, so the ceiling is one half, not undefined and not one."""
        assert bayes_auc(np.full(50, 0.3)) == pytest.approx(0.5)

    def test_brier_is_the_mean_of_p_times_one_minus_p(self) -> None:
        """The Bayes Brier score has a closed form, and it is this one."""
        p = np.array([0.1, 0.5, 0.9])
        assert bayes_brier(p) == pytest.approx(float(np.mean(p * (1 - p))))

    def test_prevalence_shift_moves_the_mean_and_not_the_ranking(self) -> None:
        """Shifting the intercept must not reorder rows, or it would change the ceiling it is holding."""
        score = np.linspace(-3, 3, 200)
        shifted, _ = shift_to_prevalence(score, 0.1)
        assert float(np.mean(shifted)) == pytest.approx(0.1, abs=0.01)
        assert np.all(np.diff(shifted) > 0)


class TestPrivateDeltaRedundancy:
    """The structure the whole cluster-collapse argument rests on."""

    @staticmethod
    def _spec() -> DatasetSpec:
        """Three reflections of one latent, whose PRIVATE parts drive the target."""
        return DatasetSpec(
            name="private_delta",
            n_samples=4000,
            root_seed=0,
            features=(FeatureSpec(name="r1"), FeatureSpec(name="r2"), FeatureSpec(name="r3"), FeatureSpec(name="n1")),
            latents=(LatentSpec(name="z", reflections=("r1", "r2", "r3"), loadings=(1.0, 1.0, 1.0), distinct_sd=0.8),),
            targets=(
                TargetSpec(
                    name="y",
                    prevalence=0.5,
                    link=LinkSpec(
                        kind="logistic",
                        coefficients={"z::delta::r1": 1.0, "z::delta::r2": -0.62, "z::delta::r3": 0.39},
                    ),
                    calibrate_to=CeilingTarget(metric="auc", value=0.80),
                ),
            ),
            edges=(EdgeSpec(source="z", target="r1", kind="redundant_noisy"), EdgeSpec(source="z", target="r2", kind="redundant_noisy"), EdgeSpec(source="z", target="r3", kind="redundant_noisy"), EdgeSpec(source="r1", target="y"), EdgeSpec(source="r2", target="y"), EdgeSpec(source="r3", target="y")),
        )

    def test_the_group_is_recorded_as_inexact(self) -> None:
        """Private parts mean the members are not interchangeable, and the truth has to say so."""
        dataset = generate(self._spec())
        groups = dataset.truth.redundancy_groups
        assert len(groups) == 1
        assert groups[0].exact is False and groups[0].rank > 1

    def test_averaging_the_cluster_destroys_more_signal_than_any_member_carries(self) -> None:
        """The cluster mean must not be a sufficient statistic for what drives the target.

        This is the property an earlier version of this scenario silently lost by weighting the private
        parts equally: the mean then preserved exactly the driving combination, aggregation cost nothing
        measurable, and the bed proved the opposite of what it claimed.
        """
        dataset = generate(self._spec())
        frame, probability = dataset.frame, dataset.truth.true_prob
        members = frame[["r1", "r2", "r3"]].to_numpy()
        mean_column = members.mean(axis=1)
        mean_corr = abs(float(np.corrcoef(mean_column, probability)[0, 1]))
        best_member = max(abs(float(np.corrcoef(members[:, i], probability)[0, 1])) for i in range(3))
        assert mean_corr < best_member
        assert mean_corr < 0.2

    def test_the_private_parts_are_not_emitted_as_columns(self) -> None:
        """Handing the deltas to a selector would hand it the answer."""
        dataset = generate(self._spec())
        assert not any("delta" in str(column) for column in dataset.frame.columns)


class TestParity:
    """Parity is the case that separates methods; a product term is not parity."""

    @staticmethod
    def _spec() -> DatasetSpec:
        """Three operands whose sign parity drives the target, plus probes."""
        return DatasetSpec(
            name="parity",
            n_samples=6000,
            root_seed=0,
            features=(FeatureSpec(name="p1"), FeatureSpec(name="p2"), FeatureSpec(name="p3"), FeatureSpec(name="n1")),
            targets=(
                TargetSpec(
                    name="y",
                    prevalence=0.5,
                    link=LinkSpec(kind="parity", interactions=(("p1", "p2", "p3"),), interaction_weights=(1.0,)),
                    calibrate_to=CeilingTarget(metric="auc", value=0.85),
                ),
            ),
            edges=(EdgeSpec(source="p1", target="y"), EdgeSpec(source="p2", target="y"), EdgeSpec(source="p3", target="y")),
        )

    def test_each_operand_is_marginally_uninformative(self) -> None:
        """Zero marginal correlation with the target probability is what makes the bed hard."""
        dataset = generate(self._spec())
        for name in ("p1", "p2", "p3"):
            corr = abs(float(np.corrcoef(dataset.frame[name].to_numpy(), dataset.truth.true_prob)[0, 1]))
            assert corr < 0.05, f"{name} is marginally visible at corr={corr:.3f}"

    def test_the_operands_are_jointly_decisive(self) -> None:
        """Their sign parity separates the probability into two well-separated groups."""
        dataset = generate(self._spec())
        signs = np.sign(dataset.frame[["p1", "p2", "p3"]].to_numpy())
        parity = signs.prod(axis=1)
        gap = abs(dataset.truth.true_prob[parity > 0].mean() - dataset.truth.true_prob[parity < 0].mean())
        one_operand = dataset.frame["p1"].to_numpy() > 0
        single_gap = abs(dataset.truth.true_prob[one_operand].mean() - dataset.truth.true_prob[~one_operand].mean())
        assert gap > 0.15
        # The comparison, not the absolute size, is the claim: the joint split separates the probability
        # while any single operand's split does essentially nothing.
        assert gap > 20 * single_gap

    def test_the_ceiling_is_reached_despite_the_marginal_blindness(self) -> None:
        """A bed nothing marginal can solve still has a well-defined, calibrated ceiling."""
        dataset = generate(self._spec())
        assert not math.isnan(dataset.calibration["bayes_auc"])
        assert dataset.calibration["bayes_auc"] == pytest.approx(0.85, abs=0.02)


class TestCorruption:
    """A corruption has to move the recorded probabilities, or the ceiling silently stops being true."""

    def test_a_flip_moves_the_recorded_probabilities_towards_one_half(self) -> None:
        """Flipping labels without updating true_prob is the failure this design forbids."""
        spec = _linear_spec(name="flipped")
        clean = generate(spec)
        flipped = generate(
            spec.model_copy(
                update={"targets": (spec.targets[0].model_copy(update={"noise": NoiseSpec(kind="uniform_flip", rate=0.2, true_prob_update="uniform_flip"), "calibrate_to": None}),)}
            )
        )
        assert float(np.std(flipped.truth.true_prob)) < float(np.std(clean.truth.true_prob))
