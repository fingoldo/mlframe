"""Behaviour of the ground-truth record types (``mlframe.data.datasets.ground_truth``)."""

import dataclasses

import numpy as np
import pytest

from mlframe.data.datasets.ground_truth import (
    PRIMARY_TARGET_SET,
    Ceiling,
    Edge,
    FeatureRole,
    FeatureTruth,
    GroundTruth,
    MIBundle,
    MIEstimate,
    RedundancyGroup,
    TargetSet,
)


def _truth(**overrides):
    """Build a small GroundTruth record, overriding named fields.

    Args:
        **overrides: Fields to replace in the default three-feature record.

    Returns:
        The constructed :class:`GroundTruth`.
    """
    kwargs = dict(
        features={
            "x1": FeatureTruth(role=FeatureRole.CAUSAL_PARENT, cost=2.0),
            "x2": FeatureTruth(role=FeatureRole.SPOUSE),
            "p1": FeatureTruth(role=FeatureRole.PROBE),
        },
        target_sets={
            PRIMARY_TARGET_SET: TargetSet(
                name=PRIMARY_TARGET_SET,
                members=("x1", "x2"),
                classes=(("x1",), ("x2",)),
                definition="parents, children and spouses of the target",
                unique=True,
            )
        },
        graph=(Edge(source="x1", target="y"),),
    )
    kwargs.update(overrides)
    return GroundTruth(**kwargs)


def test_ground_truth_is_a_frozen_dataclass_not_a_pydantic_model():
    """Frozen dataclass, so an ndarray field costs neither a copy nor a validation opt-out."""
    truth = _truth(true_prob=np.linspace(0.0, 1.0, 5))
    assert dataclasses.is_dataclass(truth)
    with pytest.raises(dataclasses.FrozenInstanceError):
        truth.target_name = "z"
    assert not hasattr(truth, "model_fields")


def test_true_prob_array_is_stored_without_copying():
    """The probability vector is held by reference; copying a 100M-row array is not an option here."""
    probabilities = np.linspace(0.0, 1.0, 100)
    assert _truth(true_prob=probabilities).true_prob is probabilities


def test_memo_field_is_excluded_from_equality():
    """Two records that differ only in what has been memoised are the same truth."""
    first, second = _truth(), _truth()
    second._memo["ceiling:auc"] = 0.9
    assert first == second


def test_roles_returns_a_copy():
    """Mutating the returned role map cannot reach into the frozen record."""
    truth = _truth()
    roles = truth.roles()
    roles["x1"] = FeatureRole.PROBE
    assert truth.features["x1"].role is FeatureRole.CAUSAL_PARENT


def test_names_with_role_filters_in_feature_order():
    """Role lookups are reported in the record's own column order."""
    assert _truth().names_with_role(FeatureRole.PROBE) == ("p1",)
    assert _truth().names_with_role(FeatureRole.CHILD) == ()


def test_feature_role_is_a_string_enum():
    """Roles serialise as plain strings, so a manifest needs no conversion step."""
    assert FeatureRole.M_COLLIDER == "m_collider"
    assert str(FeatureRole.PROBE.value) == "probe"


def test_primary_target_set_is_markov_blanket_and_missing_is_an_error():
    """The primary key is the blanket; a missing key raises rather than silently falling back.

    A silent fallback to another set would change the winner: on a spouse/collider scenario the blanket and
    the causal-parent keys crown opposite arms on identical data.
    """
    assert PRIMARY_TARGET_SET == "markov_blanket"
    assert _truth().primary_target_set().members == ("x1", "x2")
    with pytest.raises(KeyError):
        _truth(target_sets={}).primary_target_set()


def test_target_set_covers_scores_by_class_not_by_equality():
    """A non-unique key is scored by covering each equivalence class, never by exact set equality."""
    target_set = TargetSet(
        name="minimal_sufficient",
        members=("x1", "r1"),
        classes=(("x1",), ("r1", "r2", "r3")),
        definition="one representative per class",
        unique=False,
    )
    assert target_set.covers(("x1", "r2"))
    assert target_set.covers(("x1", "r3", "r1"))
    assert not target_set.covers(("x1",))


def test_mi_estimate_carries_its_provenance():
    """An MI number is never bare: the estimator, bin count and sample size travel with it."""
    estimate = MIEstimate(value=0.31, estimator="ksg_k3", n_bins=None, n_samples=1_000_000)
    assert (estimate.estimator, estimate.n_samples) == ("ksg_k3", 1_000_000)
    with pytest.raises(dataclasses.FrozenInstanceError):
        estimate.value = 0.9


def test_mi_bundle_spread_measures_estimator_disagreement():
    """The spread across estimators is the error bar; a single estimate has no measurable disagreement."""
    estimates = tuple(MIEstimate(value=value, estimator=f"bins{bins}", n_bins=bins) for value, bins in ((0.30, 5), (0.42, 50)))
    assert MIBundle(estimates=estimates).spread() == pytest.approx(0.12)
    assert MIBundle(estimates=estimates[:1]).spread() == 0.0


def test_mi_bundle_can_declare_itself_unreliable():
    """When the spread swamps the effect, the honest output is a suppression flag, not a number."""
    bundle = MIBundle(estimates=(), unreliable=True, caveats=("spread exceeds the measured effect",))
    assert bundle.unreliable and bundle.caveats


def test_ceiling_records_method_and_conditioning():
    """A ceiling states whether it is closed-form or MC, and on what it is conditional."""
    ceiling = Ceiling(value=0.83, se=0.002, method="mc", conditional_on="realized_X", n_oracle=1_000_000)
    assert (ceiling.method, ceiling.conditional_on, ceiling.metric) == ("mc", "realized_X", "auc")


def test_redundancy_group_distinguishes_exact_from_noisy():
    """Rank and exactness decide whether collapsing a group is free or destroys information."""
    exact = RedundancyGroup(members=("r1", "r2"), rank=1, exact=True)
    private = RedundancyGroup(members=("r1", "r2"), rank=2, exact=False)
    assert exact.rank == 1 and exact.exact
    assert private.rank == 2 and not private.exact


def test_expensive_truth_is_not_computed_in_the_constructor():
    """Statistical truth sits behind accessors, so building a record stays cheap."""
    assert _truth()._memo == {}


def test_the_ceiling_refuses_a_record_that_carries_no_law():
    """The oracle computes it from `true_prob`; without one there is nothing to compute it FROM.

    This used to assert `NotImplementedError` -- the accessor was a declared stub. Now that the oracle
    exists, the contract worth pinning is the refusal: a ceiling estimated from labels is a sample statistic
    wearing the name of an achievable bound, and returning one would be worse than raising.
    """
    with pytest.raises(ValueError, match="no true_prob"):
        _truth().ceiling("auc")


def test_the_ceiling_is_computed_and_memoised_when_the_law_is_present():
    """With a law attached the accessor answers, and answers once."""
    probability = np.linspace(0.05, 0.95, 200)
    truth = _truth(true_prob=probability)
    ceiling = truth.ceiling("brier")
    assert ceiling.value == pytest.approx(float(np.mean(probability * (1 - probability))))
    assert truth.ceiling("brier") is ceiling
    assert truth._memo


def test_reference_mi_needs_the_columns_it_is_about():
    """A truth record holds the law and the structure, not the data; the frame is passed in deliberately."""
    truth = _truth(true_prob=np.linspace(0.1, 0.9, 100))
    rng = np.random.default_rng(0)
    columns = {"x1": rng.normal(size=100)}
    labels = (rng.random(100) < 0.5).astype(int)
    bundle = truth.mi_reference(columns, labels)
    assert set(bundle) == {"x1"} and bundle["x1"].estimates
