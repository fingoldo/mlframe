"""Tests for the oracle: the achievable ceiling, and reference mutual information.

The ceiling is checked against hand-computable laws, because it is the denominator every regret figure in
the benchmark is measured against -- an error there propagates into every comparison silently. The MI side
is checked for the properties that make it usable at all: an exact value where one exists, disagreement
between estimator families reported rather than averaged away, and a loud failure when the oracle would
otherwise be scoring an arm against the arm's own estimator.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mlframe.data.datasets._oracle import (
    ORACLE_ESTIMATORS,
    assert_estimators_disjoint,
    binary_entropy,
    exact_ceiling,
    reference_mi,
)
from mlframe.data.datasets.generator import generate
from mlframe.data.datasets.scenarios import get as get_scenario


class TestExactCeiling:
    """Laws whose optimum can be computed by hand."""

    def test_a_coin_flip_law_has_the_ceiling_of_a_coin_flip(self) -> None:
        """Every row at p=0.5: Brier 0.25, log-loss ln 2, accuracy 0.5, and no ranking to speak of."""
        p = np.full(500, 0.5)
        assert exact_ceiling(p, "brier").value == pytest.approx(0.25)
        assert exact_ceiling(p, "logloss").value == pytest.approx(math.log(2))
        assert exact_ceiling(p, "accuracy").value == pytest.approx(0.5)
        assert exact_ceiling(p, "auc").value == pytest.approx(0.5)

    def test_a_deterministic_law_is_perfectly_predictable(self) -> None:
        """Probabilities in {0, 1} leave nothing to be uncertain about."""
        p = np.array([0.0] * 50 + [1.0] * 50)
        assert exact_ceiling(p, "brier").value == pytest.approx(0.0)
        assert exact_ceiling(p, "accuracy").value == pytest.approx(1.0)
        assert exact_ceiling(p, "auc").value == pytest.approx(1.0)

    def test_a_deterministic_law_says_its_log_loss_ceiling_is_only_attainable_in_the_limit(self) -> None:
        """Zero is the right number and an unreachable one; a ceiling that does not say so misleads."""
        ceiling = exact_ceiling(np.array([0.0] * 50 + [1.0] * 50), "logloss")
        assert ceiling.value == pytest.approx(0.0)
        assert any("attainable" in caveat for caveat in ceiling.caveats)

    def test_the_ceiling_is_exact_rather_than_estimated(self) -> None:
        """It is an expectation under a known law, so there is no sampling error to report."""
        ceiling = exact_ceiling(np.linspace(0.05, 0.95, 400), "brier")
        assert ceiling.method == "closed_form"
        assert ceiling.se == 0.0
        assert ceiling.conditional_on == "realized_X"

    def test_a_thin_minority_is_flagged(self) -> None:
        """The ceiling stays exact; what gets noisy is anything measured against it."""
        p = np.concatenate([np.full(9900, 0.001), np.full(100, 0.9)])
        assert any("minority" in caveat for caveat in exact_ceiling(p, "auc").caveats)

    def test_a_value_outside_zero_one_is_refused(self) -> None:
        """Something that is not a probability must not be quietly clipped into one."""
        with pytest.raises(ValueError, match="inside"):
            exact_ceiling(np.array([0.5, 1.5]), "brier")

    def test_an_unknown_metric_is_refused(self) -> None:
        """A typo must not fall through to a default metric and answer a different question."""
        with pytest.raises(ValueError, match="unsupported ceiling metric"):
            exact_ceiling(np.full(10, 0.5), "f1")

    def test_binary_entropy_takes_the_zero_limit(self) -> None:
        """`0 log 0` is zero here, not a warning and not a NaN travelling into a mean."""
        assert binary_entropy(np.array([0.0, 1.0, 0.5])).tolist() == pytest.approx([0.0, 0.0, math.log(2)])


class TestCeilingThroughGroundTruth:
    """The accessor callers actually use."""

    def _dataset(self) -> object:
        """A small calibrated linear bed."""
        spec = get_scenario("linear_k5_p50").build(seed=0).model_copy(update={"n_samples": 1500})
        return generate(spec)

    def test_it_agrees_with_the_calibration_that_produced_the_bed(self) -> None:
        """The bed was calibrated by bisecting on this quantity, so the two must be the same number."""
        dataset = self._dataset()
        assert dataset.truth.ceiling("auc").value == pytest.approx(dataset.calibration["bayes_auc"], abs=1e-9)  # type: ignore[attr-defined]
        assert dataset.truth.ceiling("brier").value == pytest.approx(dataset.calibration["bayes_brier"], abs=1e-9)  # type: ignore[attr-defined]

    def test_it_is_memoised_per_metric(self) -> None:
        """The accessor is documented as expensive; a caller asking twice must not pay twice."""
        truth = self._dataset().truth  # type: ignore[attr-defined]
        assert truth.ceiling("auc") is truth.ceiling("auc")
        assert truth.ceiling("brier") is not truth.ceiling("auc")

    def test_a_truth_record_without_a_law_refuses_rather_than_estimating(self) -> None:
        """An estimate off the labels is a sample statistic, not the achievable ceiling."""
        from mlframe.data.datasets.ground_truth import GroundTruth

        with pytest.raises(ValueError, match="no true_prob"):
            GroundTruth(features={}).ceiling("auc")


class TestReferenceMI:
    """Estimates, labelled as estimates."""

    def _bed(self) -> object:
        """A bed with a strong signal, a weak one and a probe."""
        spec = get_scenario("linear_k5_p50").build(seed=0).model_copy(update={"n_samples": 3000})
        return generate(spec)

    def test_a_signal_column_carries_more_information_than_a_probe(self) -> None:
        """The bundle is only useful if its exact value orders the columns the way the law does."""
        dataset = self._bed()
        columns = {name: dataset.frame[name].to_numpy() for name in ("s0", "n000")}  # type: ignore[attr-defined]
        bundles = reference_mi(columns, np.asarray(dataset.target), true_prob=dataset.truth.true_prob)  # type: ignore[attr-defined]
        assert bundles["s0"].exact is not None and bundles["n000"].exact is not None
        assert bundles["s0"].exact.value > 5 * bundles["n000"].exact.value

    def test_estimator_families_disagree_and_the_spread_is_reported(self) -> None:
        """Averaging the families away would hide exactly the uncertainty the bundle exists to show."""
        dataset = self._bed()
        bundles = reference_mi({"s0": dataset.frame["s0"].to_numpy()}, np.asarray(dataset.target), true_prob=dataset.truth.true_prob)  # type: ignore[attr-defined]
        bundle = bundles["s0"]
        assert len({estimate.estimator for estimate in bundle.estimates}) >= 3
        assert bundle.spread() > 0.0

    def test_a_wide_spread_marks_the_bundle_unreliable(self) -> None:
        """When the estimators disagree by more than the measured effects, no single number is defensible."""
        dataset = self._bed()
        bundles = reference_mi({"s0": dataset.frame["s0"].to_numpy()}, np.asarray(dataset.target), true_prob=dataset.truth.true_prob)  # type: ignore[attr-defined]
        bundle = bundles["s0"]
        assert bundle.unreliable == (bundle.spread() > 0.02)
        if bundle.unreliable:
            assert bundle.caveats

    def test_without_a_law_there_is_no_exact_value_but_still_estimates(self) -> None:
        """Reference MI degrades to estimates on data this package did not generate, and says so."""
        rng = np.random.default_rng(0)
        column = rng.normal(size=800)
        labels = (rng.random(800) < 0.5).astype(int)
        bundle = reference_mi({"x": column}, labels)["x"]
        assert bundle.exact is None and bundle.estimates

    def test_the_accessor_memoises_per_column_set(self) -> None:
        """Two different column sets are two different questions and must not share one memo slot."""
        dataset = self._bed()
        labels = np.asarray(dataset.target)  # type: ignore[attr-defined]
        first = {"s0": dataset.frame["s0"].to_numpy()}  # type: ignore[attr-defined]
        second = {"s1": dataset.frame["s1"].to_numpy()}  # type: ignore[attr-defined]
        truth = dataset.truth  # type: ignore[attr-defined]
        assert truth.mi_reference(first, labels) is truth.mi_reference(first, labels)
        assert truth.mi_reference(second, labels) is not truth.mi_reference(first, labels)


class TestEstimatorDisjointness:
    """The oracle must not score an arm against the arm's own estimator."""

    def test_a_shared_backend_is_refused(self) -> None:
        """Otherwise the rank-correlation metric measures the arm agreeing with itself."""
        with pytest.raises(ValueError, match="must not share an estimator family"):
            assert_estimators_disjoint(["equal_width_plugin", "ksg"])

    def test_disjoint_backends_pass(self) -> None:
        """The arms' own binned backend is a different family from every estimator named here."""
        assert assert_estimators_disjoint(["mlframe_binned_njit", "ksg", "sklearn_mutual_info"]) is None
        assert "equal_width_plugin" in ORACLE_ESTIMATORS
