"""Targets that are not a coin flip, and the one construction that puts partial truth on real data.

Both extend what the suite can conclude rather than adding more of what it already had. The non-binary
targets change what a selector's job IS; the matched-probe injection gives the real leg a one-sided answer
key where it previously had none.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from mlframe.data.datasets import scenarios as scenario_registry
from mlframe.data.datasets._targets_nonbinary import categorical_ceiling, count_ceiling, count_mean, ordinal_probabilities, rotate_weights
from mlframe.data.datasets.from_real import PROBE_PREFIX, inject_matched_probes, probe_false_discovery_rate
from mlframe.data.datasets.generator import generate
from mlframe.data.datasets.scenarios._targets import SHARED_WEIGHTS, count_spec, multiclass_spec, ordinal_spec

ROWS = 2000


def test_rotated_weights_keep_the_same_columns_and_magnitudes() -> None:
    """Redrawing per class would make class count and signal strength vary together, confounding both."""
    per_class = rotate_weights({"a": 1.0, "b": 0.5, "c": 0.25}, n_classes=3)

    assert per_class[0] == {}, "the first class is the reference and must carry no weights of its own"
    assert per_class[1:]
    for weights in per_class[1:]:
        assert set(weights) == {"a", "b", "c"}
        assert sorted(weights.values()) == [0.25, 0.5, 1.0]


def test_rotated_weights_differ_between_classes() -> None:
    """Identical vectors would make the bed ordinal wearing a multiclass name."""
    per_class = rotate_weights({"a": 1.0, "b": 0.5, "c": 0.25}, n_classes=4)

    arrangements = {tuple(sorted(weights.items())) for weights in per_class[1:]}
    assert len(arrangements) == len(per_class) - 1, "two classes depend on the columns in the same arrangement"


def test_ordinal_probabilities_sum_to_one_and_stay_non_negative() -> None:
    """A cut-point set that produced a negative probability would be neither ordinal nor multiclass."""
    score = np.linspace(-3.0, 3.0, 500)

    law = ordinal_probabilities(score, np.array([-1.0, 0.0, 1.2]))

    assert law.shape == (500, 4)
    np.testing.assert_allclose(law.sum(axis=1), 1.0, atol=1e-12)
    assert float(law.min()) >= 0.0


def test_ordinal_cut_points_must_be_increasing() -> None:
    """Unordered cuts silently produce a different bed; the order IS what makes the classes ordered."""
    with pytest.raises(ValueError, match="strictly increasing"):
        ordinal_probabilities(np.zeros(10), np.array([1.0, 0.5]))


def test_ordinal_classes_really_are_ordered_along_the_score() -> None:
    """If the class means did not move monotonically with the score, the bed would not be ordinal at all."""
    generated = generate(ordinal_spec(seed=0, n_samples=ROWS))
    frame, labels = generated.frame, generated.target.to_numpy()
    score = sum(weight * frame[name].to_numpy() for name, weight in SHARED_WEIGHTS.items())

    means = [float(np.mean(score[labels == value])) for value in sorted(set(labels))]
    assert means == sorted(means), f"class means along the latent are not ordered: {np.round(means, 3).tolist()}"


def test_count_mean_clips_the_exponent_rather_than_the_mean() -> None:
    """Clipping the mean would flatten the relationship; clipping the exponent keeps it monotone everywhere."""
    values = count_mean(np.array([-100.0, 0.0, 1.0, 100.0]), clip=5.0)

    assert values[0] == pytest.approx(np.exp(-5.0))
    assert values[-1] == pytest.approx(np.exp(5.0))
    assert values[1] < values[2] < values[3]


def test_categorical_ceiling_is_exact_on_a_law_with_a_known_answer() -> None:
    """A deterministic law has zero achievable log-loss and perfect achievable accuracy; anything else is a bug."""
    certain = np.array([[1.0, 0.0], [0.0, 1.0]])

    ceiling = categorical_ceiling(certain)

    assert ceiling["log_loss"] == pytest.approx(0.0, abs=1e-9)
    assert ceiling["accuracy"] == pytest.approx(1.0)
    assert ceiling["brier"] == pytest.approx(0.0)


def test_categorical_ceiling_on_a_uniform_law_is_the_entropy_of_the_uniform() -> None:
    """The other end: nothing can beat chance on a law that is chance, and the number says exactly that."""
    uniform = np.full((100, 4), 0.25)

    ceiling = categorical_ceiling(uniform)

    assert ceiling["log_loss"] == pytest.approx(float(np.log(4.0)))
    assert ceiling["accuracy"] == pytest.approx(0.25)


def test_count_ceiling_is_not_zero() -> None:
    """A perfect count model still has residual deviance; reporting zero puts every real model at infinity."""
    ceiling = count_ceiling(np.full(50, 4.0))

    assert ceiling["poisson_deviance"] > 0.0
    assert ceiling["mean"] == pytest.approx(4.0)


@pytest.mark.parametrize("builder,expected_classes", [(multiclass_spec, 3), (ordinal_spec, 4)])
def test_categorical_beds_produce_every_class(builder: Any, expected_classes: int) -> None:
    """A bed whose fourth class never appears is a three-class bed with a four-class name."""
    generated = generate(builder(seed=0, n_samples=ROWS))

    assert len(set(generated.target.to_numpy())) == expected_classes


def test_categorical_beds_carry_the_full_per_row_law() -> None:
    """The ceiling is exact only because the law is known per row; storing the labels alone would lose it."""
    generated = generate(multiclass_spec(seed=0, n_samples=ROWS))

    law = np.asarray(generated.truth.true_mean)
    assert law.shape == (ROWS, 3)
    np.testing.assert_allclose(law.sum(axis=1), 1.0, atol=1e-10)


def test_non_binary_beds_leave_the_binary_probability_unset() -> None:
    """`true_prob` is documented as the binary law; filling it for a four-class bed would be a false claim."""
    for builder in (multiclass_spec, ordinal_spec, count_spec):
        generated = generate(builder(seed=0, n_samples=500))
        assert generated.truth.true_prob is None


def test_non_binary_beds_say_what_kind_they_are_in_their_caveats() -> None:
    """A reader who does not know the target kind cannot read the ceiling that travels with it."""
    caveats = " ".join(generate(count_spec(seed=0, n_samples=500)).truth.caveats)

    assert "Poisson" in caveats


def test_the_count_bed_is_heteroscedastic_as_a_poisson_must_be() -> None:
    """Variance growing with the mean is the property that makes a constant-variance scorer mis-weighted."""
    generated = generate(count_spec(seed=0, n_samples=4000))
    counts = generated.target.to_numpy().astype(np.float64)
    means = np.asarray(generated.truth.true_mean)

    low = counts[means <= np.median(means)]
    high = counts[means > np.median(means)]
    assert float(np.var(high)) > 2.0 * float(np.var(low)), "the high-mean rows are no more variable than the low-mean ones"


def test_an_unimplemented_target_kind_raises_rather_than_drawing_a_binary_label() -> None:
    """A binary label under a multilabel name is worse than an error, because nothing downstream would notice."""
    spec = multiclass_spec(seed=0, n_samples=200)
    broken = spec.model_copy(update={"targets": (spec.targets[0].model_copy(update={"kind": "multilabel"}),)})

    with pytest.raises(ValueError, match="no non-binary implementation"):
        generate(broken)


@pytest.mark.parametrize("name", ["multiclass_k3", "ordinal_k4", "count_poisson"])
def test_every_non_binary_bed_is_registered_with_a_prediction(name: str) -> None:
    """A bed nobody registered cannot run, and one predicting nothing cannot be scored."""
    assert name in scenario_registry.names()
    assert scenario_registry.get(name).expected_to_break


def _real_like_frame(seed: int = 0) -> Any:
    """Return a frame whose columns have the shapes real data has: heavy tails, a point mass, a plain column."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "heavy": rng.standard_t(3, 600),
            "sparse": np.where(rng.random(600) < 0.6, 0.0, rng.normal(size=600)),
            "plain": rng.normal(size=600),
        }
    )
    labels = (rng.random(600) < 1.0 / (1.0 + np.exp(-frame["plain"].to_numpy()))).astype(np.int64)
    return frame, labels


def test_injected_probes_reproduce_their_donor_column_s_marginal() -> None:
    """A probe drawn from a standard normal beside heavy-tailed real columns is findable without the target.

    That is the failure this construction exists to avoid: the selector never has to look at the label,
    the false-positive rate comes out flattering, and the bed has measured nothing.
    """
    frame, labels = _real_like_frame()

    bed = inject_matched_probes(frame, labels, n_probes=3, seed=0, source="demo")

    donor_kurtosis = float(pd.Series(frame["heavy"]).kurt())
    probe_kurtosis = float(pd.Series(bed.frame[f"{PROBE_PREFIX}0000"]).kurt())
    assert probe_kurtosis == pytest.approx(donor_kurtosis, rel=0.6), f"the probe's tail ({probe_kurtosis:.2f}) does not match its donor's ({donor_kurtosis:.2f})"


def test_injected_probes_reproduce_a_point_mass() -> None:
    """A parametric fit would smooth the mass away and leave the probe separable by its shape alone."""
    frame, labels = _real_like_frame()

    bed = inject_matched_probes(frame, labels, n_probes=3, seed=0, source="demo")

    donor_share = float((frame["sparse"].to_numpy() == 0.0).mean())
    probe_share = float((bed.frame[f"{PROBE_PREFIX}0001"].to_numpy() == 0.0).mean())
    assert probe_share == pytest.approx(donor_share, abs=0.08)


def test_injected_probes_carry_no_relationship_to_the_target() -> None:
    """They are irrelevant BY CONSTRUCTION, which is the only thing this bed knows for certain."""
    frame, labels = _real_like_frame()

    bed = inject_matched_probes(frame, labels, n_probes=6, seed=0, source="demo")

    correlations = [abs(float(np.corrcoef(bed.frame[name], labels)[0, 1])) for name in bed.probe_columns]
    assert max(correlations) < 0.2, f"an injected probe correlates {max(correlations):.3f} with the target"


def test_the_truth_refuses_to_claim_an_answer_key_it_does_not_have() -> None:
    """A `base` key would make the harness compute recall against columns whose relevance is unknown."""
    frame, labels = _real_like_frame()

    truth = inject_matched_probes(frame, labels, n_probes=2, seed=0, source="demo").truth()

    assert "base" not in truth
    assert truth["recall_is_unmeasurable"] is True
    assert truth["known_irrelevant"] and truth["unknown_relevance"]


def test_the_false_discovery_bound_counts_only_the_known_bad() -> None:
    """It is a LOWER bound: some selected real columns are probably wrong too, and nothing here can say which."""
    frame, labels = _real_like_frame()
    bed = inject_matched_probes(frame, labels, n_probes=4, seed=0, source="demo")

    bound = probe_false_discovery_rate(["plain", bed.probe_columns[0], bed.probe_columns[1]], bed.probe_columns)

    assert bound == pytest.approx(2.0 / 3.0)
    assert probe_false_discovery_rate([], bed.probe_columns) is None


def test_injection_refuses_a_frame_with_no_numeric_marginal_to_match() -> None:
    """A probe with no marginal to copy would have to be invented and would be findable by its shape."""
    frame = pd.DataFrame({"label_only": pd.Categorical(["a", "b"] * 5)})

    with pytest.raises(ValueError, match="no numeric column"):
        inject_matched_probes(frame, np.zeros(10), n_probes=2, seed=0, source="demo")


def test_injection_leaves_the_caller_s_frame_untouched() -> None:
    """A caller reusing the original frame for a second bed must not find the first bed's probes in it."""
    frame, labels = _real_like_frame()
    before = list(frame.columns)

    inject_matched_probes(frame, labels, n_probes=3, seed=0, source="demo")

    assert list(frame.columns) == before
