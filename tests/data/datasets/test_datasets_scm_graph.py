"""Graph queries, role derivation and target-set derivation (``mlframe.data.datasets._scm``).

The point of these tests is that ground truth cannot drift from the generator: roles and answer keys are
computed from the edge list the generator itself consumes, so changing an arrow changes the key in the same
commit or not at all.
"""

import pytest

from mlframe.data.datasets._scm import (
    CausalGraph,
    build_ground_truth,
    build_target_sets,
    derive_roles,
    edges_from_specs,
    graph_from_spec,
)
from mlframe.data.datasets.ground_truth import Edge, FeatureRole, RedundancyGroup
from mlframe.data.datasets.spec import DatasetSpec, EdgeSpec, FeatureSpec, GateSpec, LatentSpec, TargetSpec


def _graph(edges, observed, latents=()):
    """Build a CausalGraph on target ``y`` from ``(source, target[, kind])`` tuples.

    Args:
        edges: Iterable of 2- or 3-tuples describing the arrows.
        observed: Observed column names, in reporting order.
        latents: Unobserved node names.

    Returns:
        The constructed :class:`CausalGraph`.
    """
    built = tuple(Edge(source=item[0], target=item[1], kind=item[2] if len(item) > 2 else "direct") for item in edges)
    return CausalGraph(edges=built, target="y", observed=observed, latents=latents)


def test_graph_rejects_cycles_and_observed_latent_overlap():
    """An SCM must be a DAG, and a node cannot be observed and latent at once."""
    with pytest.raises(ValueError, match="cycle"):
        _graph([("a", "b"), ("b", "c"), ("c", "a"), ("a", "y")], observed=("a", "b", "c"))
    with pytest.raises(ValueError, match="observed and latent"):
        _graph([("a", "y")], observed=("a",), latents=("a",))


def test_ancestors_and_descendants_are_transitive():
    """Reachability is computed over paths, not over direct edges."""
    graph = _graph([("a", "b"), ("b", "c"), ("c", "y")], observed=("a", "b", "c"))
    assert graph.ancestors("y") == {"a", "b", "c"}
    assert graph.descendants("a") == {"b", "c", "y"}
    assert graph.parents("c") == ("b",)
    assert graph.children("a") == ("b",)


def test_markov_blanket_is_parents_children_and_spouses():
    """The canonical blanket, on the spouse/collider shape ``Y -> C <- S``.

    ``S`` is invisible to every marginal method until ``C`` is conditioned on, yet it is a genuine blanket
    member - which is exactly why the answer key has to be derived rather than eyeballed.
    """
    graph = _graph([("x1", "y"), ("y", "c"), ("s", "c"), ("noise", "other")], observed=("x1", "c", "s", "noise", "other"))
    assert graph.markov_blanket() == ("x1", "c", "s")
    assert graph.causal_parents() == ("x1",)


def test_roles_on_the_spouse_collider_shape():
    """Each column gets the role its position in the graph implies, and a disconnected one is a probe."""
    graph = _graph([("x1", "y"), ("y", "c"), ("s", "c")], observed=("x1", "c", "s", "p1"))
    assert derive_roles(graph) == {
        "x1": FeatureRole.CAUSAL_PARENT,
        "c": FeatureRole.CHILD,
        "s": FeatureRole.SPOUSE,
        "p1": FeatureRole.PROBE,
    }


def test_mediator_outranks_causal_parent():
    """A column transmitting another feature's effect is a mediator, not merely a parent.

    The distinction is load-bearing: a mediator is screened off once its own upstream cause is conditioned
    on, so "adjust or do not adjust" has opposite answers for the two roles.
    """
    graph = _graph([("x1", "m"), ("m", "y")], observed=("x1", "m"))
    roles = derive_roles(graph)
    assert roles["m"] is FeatureRole.MEDIATOR
    assert roles["x1"] is FeatureRole.CAUSAL_ANCESTOR


def test_m_bias_collider_is_detected():
    """``U1 -> X``, ``U1 -> Z``, ``U2 -> Z``, ``U2 -> Y``: ``Z`` must NOT be selected.

    ``Z`` is marginally independent of both ``X`` and ``Y``; conditioning on it CREATES the association.
    """
    graph = _graph(
        [("u1", "x"), ("u1", "z"), ("u2", "z"), ("u2", "y"), ("x", "y")],
        observed=("x", "z"),
        latents=("u1", "u2"),
    )
    assert derive_roles(graph)["z"] is FeatureRole.M_COLLIDER
    assert "z" not in graph.markov_blanket()


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("redundant_exact", FeatureRole.REDUNDANT_EXACT),
        ("redundant_noisy", FeatureRole.REDUNDANT_NOISY),
        ("instrument", FeatureRole.INSTRUMENT),
        ("proxy", FeatureRole.PROXY),
        ("shift", FeatureRole.SHIFT_NUISANCE),
        ("conditional_only", FeatureRole.CONDITIONAL_ONLY),
    ],
)
def test_edge_kind_declares_the_roles_shape_alone_cannot_determine(kind, expected):
    """An instrument and a plain ancestor have the same shape; the edge kind resolves them."""
    graph = _graph([("x1", "y"), ("x1", "r"), ("r", "y")], observed=("x1", "r"))
    graph_with_kind = _graph([("x1", "y"), ("x1", "r", kind), ("r", "y")], observed=("x1", "r"))
    assert derive_roles(graph)["r"] is not expected
    assert derive_roles(graph_with_kind)["r"] is expected


def test_regional_role_needs_a_gate():
    """A feature informative on part of the space is regional, not a probe and not a global weak effect."""
    graph = _graph([("x1", "y")], observed=("x1", "g"))
    assert derive_roles(graph)["g"] is FeatureRole.PROBE
    assert derive_roles(graph, regions={"g": GateSpec(column="g", low=0.0)})["g"] is FeatureRole.REGIONAL


def test_latent_blanket_member_is_reported_as_a_caveat_not_dropped_silently():
    """An unobservable blanket member is a real property of the scenario, so it is recorded."""
    graph = _graph([("z", "y"), ("x1", "y")], observed=("x1",), latents=("z",))
    blanket = build_target_sets(graph)["markov_blanket"]
    assert blanket.members == ("x1",)
    assert blanket.caveats and "z" in blanket.caveats[0]


def test_three_target_sets_are_all_built_and_carry_their_uniqueness():
    """All three keys are reported; only the blanket and the parents are unique."""
    graph = _graph([("x1", "y"), ("y", "c"), ("s", "c")], observed=("x1", "c", "s"))
    target_sets = build_target_sets(graph)
    assert set(target_sets) == {"markov_blanket", "minimal_sufficient", "causal_parents"}
    assert target_sets["markov_blanket"].unique
    assert target_sets["causal_parents"].unique
    assert not target_sets["minimal_sufficient"].unique
    assert target_sets["causal_parents"].members == ("x1",)


def test_minimal_sufficient_collapses_exact_copies_but_not_private_deltas():
    """Exact copies form one equivalence class; replicates with private deltas stay separate.

    Collapsing the latter would declare a correct arm wrong for keeping information that averaging destroys.
    """
    graph = _graph([("r1", "y"), ("r2", "y"), ("x1", "y")], observed=("r1", "r2", "x1"))
    exact = build_target_sets(graph, [RedundancyGroup(members=("r1", "r2"), rank=1, exact=True)])["minimal_sufficient"]
    assert exact.classes == (("r1", "r2"), ("x1",))
    assert exact.covers(("r2", "x1")) and not exact.covers(("r1", "r2"))

    private = build_target_sets(graph, [RedundancyGroup(members=("r1", "r2"), rank=2, exact=False)])["minimal_sufficient"]
    assert private.classes == (("r1",), ("r2",), ("x1",))
    assert not private.covers(("r1", "x1"))


def _spec():
    """Build a spec whose graph exercises parent, child, spouse and probe roles.

    Returns:
        The constructed :class:`DatasetSpec`.
    """
    return DatasetSpec(
        name="spouse_collider",
        n_samples=256,
        root_seed=7,
        features=(
            FeatureSpec(name="x1", cost=3.0),
            FeatureSpec(name="c"),
            FeatureSpec(name="s"),
            FeatureSpec(name="p1"),
        ),
        latents=(LatentSpec(name="z"),),
        targets=(TargetSpec(name="y"),),
        edges=(
            EdgeSpec(source="x1", target="y"),
            EdgeSpec(source="y", target="c"),
            EdgeSpec(source="s", target="c"),
            EdgeSpec(source="z", target="p1", kind="latent"),
        ),
    )


def test_edges_from_specs_round_trips_kind_and_weight():
    """Spec edges convert to truth edges without losing their annotations."""
    converted = edges_from_specs((EdgeSpec(source="a", target="b", kind="proxy", weight=0.5),))
    assert converted == (Edge(source="a", target="b", kind="proxy", weight=0.5),)


def test_graph_from_spec_defaults_to_the_first_target_and_rejects_an_unknown_one():
    """A single-target scenario needs no choice; an unknown target name is an error, not a fallback."""
    assert graph_from_spec(_spec()).target == "y"
    with pytest.raises(ValueError, match="not declared"):
        graph_from_spec(_spec(), target_name="nope")
    with pytest.raises(ValueError, match="no target"):
        graph_from_spec(DatasetSpec(name="empty", n_samples=1))


def test_build_ground_truth_derives_structural_truth_from_the_spec():
    """Roles, target sets, edges and per-feature cost all come from the one declaration."""
    truth = build_ground_truth(_spec(), regions={"p1": GateSpec(column="p1", low=0.0)})
    assert truth.target_name == "y"
    assert truth.roles() == {
        "x1": FeatureRole.CAUSAL_PARENT,
        "c": FeatureRole.CHILD,
        "s": FeatureRole.SPOUSE,
        "p1": FeatureRole.REGIONAL,
    }
    assert truth.primary_target_set().members == ("x1", "c", "s")
    assert truth.features["x1"].cost == 3.0
    assert truth.features["p1"].region is not None
    assert len(truth.graph) == 4


def test_build_ground_truth_records_the_source_of_a_derived_column():
    """A redundant copy or a proxy remembers what it was derived from."""
    spec = DatasetSpec(
        name="redundant",
        n_samples=64,
        features=(FeatureSpec(name="x1"), FeatureSpec(name="r1")),
        targets=(TargetSpec(name="y"),),
        edges=(EdgeSpec(source="x1", target="y"), EdgeSpec(source="x1", target="r1", kind="redundant_exact")),
    )
    truth = build_ground_truth(spec, redundancy_groups=[RedundancyGroup(members=("x1", "r1"), rank=1, exact=True)])
    assert truth.features["r1"].role is FeatureRole.REDUNDANT_EXACT
    assert truth.features["r1"].source == "x1"
    assert truth.redundancy_groups[0].rank == 1


def test_changing_one_edge_changes_the_answer_key():
    """The anti-drift property: the key is a function of the graph, not a parallel hand-written list."""
    spec = _spec()
    flipped = spec.model_copy(update={"edges": (EdgeSpec(source="x1", target="y"), EdgeSpec(source="c", target="y"), EdgeSpec(source="s", target="c"))})
    assert build_ground_truth(spec).primary_target_set().members == ("x1", "c", "s")
    assert build_ground_truth(flipped).primary_target_set().members == ("x1", "c")
