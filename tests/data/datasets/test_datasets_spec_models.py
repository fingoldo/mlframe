"""Validation behaviour of the dataset specification models (``mlframe.data.datasets.spec``)."""

import numpy as np
import pytest
from pydantic import ValidationError

from mlframe.data.datasets.spec import (
    CeilingTarget,
    DatasetSpec,
    EdgeSpec,
    FeatureSpec,
    GateSpec,
    LatentSpec,
    LinkSpec,
    NoiseSpec,
    Prior,
    TargetSpec,
)


def test_subspecs_are_frozen_and_forbid_extras():
    """Immutability keeps a spec from being edited after the run that used it; extra=forbid catches typos."""
    feature = FeatureSpec(name="x1")
    with pytest.raises(ValidationError):
        feature.cost = 2.0
    with pytest.raises(ValidationError):
        FeatureSpec(name="x1", coste=2.0)


def test_content_hash_is_stable_and_order_insensitive():
    """The cache key is canonical sorted-key JSON, so dict insertion order cannot fork the cache."""
    first = FeatureSpec(name="x1", params={"loc": 0.0, "scale": 2.0})
    second = FeatureSpec(name="x1", params={"scale": 2.0, "loc": 0.0})
    assert first.content_hash() == second.content_hash()
    assert first.content_hash() != FeatureSpec(name="x1", params={"loc": 1.0, "scale": 2.0}).content_hash()
    assert len(first.content_hash()) == 32


def test_feature_spec_has_cost_from_the_start():
    """``cost`` is present in v1: retrofitting it later changes every spec hash and voids every cached dataset."""
    assert FeatureSpec(name="x1").cost == 1.0
    assert FeatureSpec(name="x1", cost=0.25).cost == 0.25
    with pytest.raises(ValidationError):
        FeatureSpec(name="x1", cost=-1.0)


def test_feature_spec_ties_levels_to_categorical_dtype():
    """A categorical column must declare its levels explicitly, and a numeric one must not."""
    assert FeatureSpec(name="c", dtype="category", levels=("a", "b")).levels == ("a", "b")
    with pytest.raises(ValidationError):
        FeatureSpec(name="c", dtype="category")
    with pytest.raises(ValidationError):
        FeatureSpec(name="x", dtype="float", levels=("a",))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "uniform"},
        {"kind": "uniform", "low": 1.0, "high": 0.5},
        {"kind": "loguniform", "low": 0.0, "high": 1.0},
        {"kind": "choice"},
        {"kind": "uniform", "low": 0.0, "high": 1.0, "choices": (1.0,)},
    ],
)
def test_prior_rejects_unsampleable_support(kwargs):
    """Every malformed support is rejected at construction, not at sampling time."""
    with pytest.raises(ValidationError):
        Prior(**kwargs)


@pytest.mark.parametrize(
    "prior,low,high",
    [
        (Prior(kind="uniform", low=0.5, high=0.99), 0.5, 0.99),
        (Prior(kind="loguniform", low=0.01, high=0.5), 0.01, 0.5),
        (Prior(kind="uniform_int", low=2, high=8), 2, 8),
        (Prior(kind="choice", choices=(2.0, 5.0)), 2.0, 5.0),
    ],
)
def test_prior_samples_inside_its_support_and_is_deterministic(prior, low, high):
    """Draws stay in support and are a pure function of the generator's seed."""
    values = [prior.sample(np.random.default_rng(11)) for _ in range(3)]
    assert values[0] == values[1] == values[2]
    assert low <= values[0] <= high


def test_prior_uniform_int_is_integral_and_inclusive():
    """The integer family draws integral values and can reach both bounds."""
    prior = Prior(kind="uniform_int", low=2, high=4)
    drawn = {prior.sample(np.random.default_rng(seed)) for seed in range(200)}
    assert drawn <= {2.0, 3.0, 4.0}
    assert drawn == {2.0, 3.0, 4.0}


def test_knob_accepts_both_a_value_and_a_prior():
    """A knob may be pinned or declared; a declared prior is what makes a sweep auditable."""
    assert LatentSpec(name="z", distinct_sd=0.3).distinct_sd == 0.3
    swept = LatentSpec(name="z", distinct_sd=Prior(kind="uniform", low=0.0, high=1.0))
    assert isinstance(swept.distinct_sd, Prior)


def test_gate_spec_requires_a_usable_interval():
    """A gate with no bound describes no region; an inverted one describes an empty one."""
    assert GateSpec(column="x1", low=0.0).low == 0.0
    with pytest.raises(ValidationError):
        GateSpec(column="x1")
    with pytest.raises(ValidationError):
        GateSpec(column="x1", low=1.0, high=0.0)


def test_latent_spec_requires_one_loading_per_reflection():
    """A loading vector of the wrong length is a silent mis-specification of the latent structure."""
    assert LatentSpec(name="z", reflections=("r1", "r2"), loadings=(0.9, 0.8)).loadings == (0.9, 0.8)
    with pytest.raises(ValidationError):
        LatentSpec(name="z", reflections=("r1", "r2"), loadings=(0.9,))


def test_link_spec_validates_interaction_arity_and_weights():
    """An interaction needs two operands, and weights must match the term count when supplied."""
    link = LinkSpec(kind="parity", interactions=(("x1", "x2"),), interaction_weights=(1.0,))
    assert link.interactions[0] == ("x1", "x2")
    with pytest.raises(ValidationError):
        LinkSpec(interactions=(("x1",),))
    with pytest.raises(ValidationError):
        LinkSpec(interactions=(("x1", "x2"),), interaction_weights=(1.0, 2.0))


def test_noise_spec_refuses_a_corruption_without_a_true_prob_update():
    """The ceiling invariant: a corruption that cannot update ``true_prob`` destroys the Bayes ceiling."""
    ok = NoiseSpec(kind="uniform_flip", rate=0.1, true_prob_update="uniform_flip")
    assert ok.kind == "uniform_flip"
    with pytest.raises(ValidationError):
        NoiseSpec(kind="uniform_flip", rate=0.1)
    with pytest.raises(ValidationError):
        NoiseSpec(kind="feature_dependent_flip", rate=0.1, true_prob_update="feature_dependent_flip")


def test_target_spec_keeps_kind_and_arity_consistent():
    """A binary target with three classes is a contradiction, not a configuration."""
    assert TargetSpec(name="y", calibrate_to=CeilingTarget(value=0.75)).n_classes == 2
    with pytest.raises(ValidationError):
        TargetSpec(name="y", kind="binary", n_classes=3)
    with pytest.raises(ValidationError):
        TargetSpec(name="y", kind="multiclass", n_classes=5000)


def test_edge_spec_rejects_self_loops():
    """An SCM edge connects two distinct nodes."""
    with pytest.raises(ValidationError):
        EdgeSpec(source="x1", target="x1")


def _spec(**overrides):
    """Build a small valid DatasetSpec, overriding named fields.

    Args:
        **overrides: Fields to replace in the default two-feature, one-target spec.

    Returns:
        The constructed :class:`DatasetSpec`.
    """
    kwargs = dict(
        name="tiny",
        n_samples=100,
        features=(FeatureSpec(name="x1", cost=2.0), FeatureSpec(name="x2", cost=0.5)),
        targets=(TargetSpec(name="y"),),
        edges=(EdgeSpec(source="x1", target="y"),),
    )
    kwargs.update(overrides)
    return DatasetSpec(**kwargs)


def test_dataset_spec_rejects_duplicate_names():
    """Two nodes with one name make every graph query ambiguous."""
    with pytest.raises(ValidationError):
        _spec(features=(FeatureSpec(name="x1"), FeatureSpec(name="x1")))


def test_dataset_spec_rejects_dangling_edges():
    """An edge to an undeclared node is caught before a single row is generated."""
    with pytest.raises(ValidationError):
        _spec(edges=(EdgeSpec(source="x9", target="y"),))


def test_dataset_spec_preserves_column_order_and_totals_cost():
    """Column order is declaration order (never shuffled), and total cost is the all-features baseline."""
    spec = _spec()
    assert spec.feature_names() == ("x1", "x2")
    assert spec.total_cost() == pytest.approx(2.5)
