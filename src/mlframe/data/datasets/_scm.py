"""The structural causal model: the graph, the roles derived from it, and the target sets derived from those.

This module exists so that ground truth CANNOT drift from the generator. Roles are not hand-listed next to
a generator that may or may not still implement them; they are computed from the edge list that the
generator itself consumes, and the three target sets are computed from the same edges. Change an arrow and
the answer key changes with it, in the same commit, or not at all.

Role precedence
---------------
Several roles can describe one node at once (a mediator is usually also a parent of the target), so the
derivation applies a fixed, documented precedence. The order below is the order
:func:`derive_roles` tests, and it is chosen so the most specific and most consequential statement wins:

1. ``REDUNDANT_EXACT`` / ``REDUNDANT_NOISY`` - declared by edge kind; whether a copy is exact decides
   whether collapsing the group is free or destroys information.
2. ``SHIFT_NUISANCE`` / ``CONDITIONAL_ONLY`` / ``INSTRUMENT`` / ``PROXY`` - declared by edge kind, because
   these are the roles the arrow direction genuinely cannot determine on its own.
3. ``M_COLLIDER`` - a collider off the causal path whose conditioning OPENS a spurious association;
   selecting it is actively harmful, so this outranks the merely descriptive roles below it.
4. ``CHILD`` - the target is a parent of the node.
5. ``SPOUSE`` - shares a child with the target.
6. ``MEDIATOR`` - lies on a directed path from another feature to the target.
7. ``CAUSAL_PARENT`` then ``CAUSAL_ANCESTOR``.
8. ``REGIONAL`` when a gate is attached, else ``PROBE``.

Latent nodes are part of the graph for path reasoning but never appear in a target set: an answer key
containing an unobservable column is unanswerable. When a latent lands inside the Markov blanket that is a
real property of the scenario, so it is recorded as a caveat on the target set rather than silently dropped.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

from mlframe.data.datasets.ground_truth import (
    Edge,
    FeatureRole,
    FeatureTruth,
    GroundTruth,
    RedundancyGroup,
    TargetSet,
)
from mlframe.data.datasets.spec import DatasetSpec, EdgeSpec, GateSpec

#: Edge kinds that directly name a role the graph shape alone cannot distinguish.
_KIND_TO_ROLE: Dict[str, FeatureRole] = {
    "redundant_exact": FeatureRole.REDUNDANT_EXACT,
    "redundant_noisy": FeatureRole.REDUNDANT_NOISY,
    "shift": FeatureRole.SHIFT_NUISANCE,
    "conditional_only": FeatureRole.CONDITIONAL_ONLY,
    "instrument": FeatureRole.INSTRUMENT,
    "proxy": FeatureRole.PROXY,
}


def edges_from_specs(edge_specs: Sequence[EdgeSpec]) -> Tuple[Edge, ...]:
    """Convert validated spec edges into the truth-side edge records.

    Args:
        edge_specs: Edges as declared in a :class:`~mlframe.data.datasets.spec.DatasetSpec`.

    Returns:
        The same edges as frozen :class:`~mlframe.data.datasets.ground_truth.Edge` dataclasses, in order.
    """
    return tuple(Edge(source=spec.source, target=spec.target, kind=spec.kind, weight=spec.weight) for spec in edge_specs)


class CausalGraph:
    """A directed graph over features, latents and one target, with the queries the truth layer needs.

    Not a frozen dataclass despite being immutable in practice: it precomputes adjacency maps in
    ``__init__``, and a frozen dataclass would need ``object.__setattr__`` for each of them, which buys
    nothing here (the class is internal and never hashed).
    """

    def __init__(
        self,
        edges: Sequence[Edge],
        target: str,
        observed: Sequence[str],
        latents: Sequence[str] = (),
    ) -> None:
        """Build adjacency maps and validate that the graph is acyclic.

        Args:
            edges: The arrows of the model.
            target: Name of the target node.
            observed: Names of the observed feature columns, in the order the frame presents them; roles
                and target sets are reported in this order so an answer key is stable.
            latents: Names of unobserved nodes.

        Raises:
            ValueError: If the target is not a node of the graph, an observed name is also declared latent,
                or the edges contain a directed cycle (an SCM must be a DAG for any of these queries to be
                well defined).
        """
        self.edges: Tuple[Edge, ...] = tuple(edges)
        self.target = target
        self.observed: Tuple[str, ...] = tuple(observed)
        self.latents: Tuple[str, ...] = tuple(latents)
        overlap = set(self.observed) & set(self.latents)
        if overlap:
            raise ValueError(f"nodes declared both observed and latent: {sorted(overlap)}")
        self._parents: Dict[str, List[str]] = {}
        self._children: Dict[str, List[str]] = {}
        for node in (*self.observed, *self.latents, self.target):
            self._parents.setdefault(node, [])
            self._children.setdefault(node, [])
        for edge in self.edges:
            self._parents.setdefault(edge.target, []).append(edge.source)
            self._children.setdefault(edge.source, []).append(edge.target)
            self._parents.setdefault(edge.source, [])
            self._children.setdefault(edge.target, [])
        if self.target not in self._parents:
            raise ValueError(f"target {target!r} is not a node of the graph")
        self._check_acyclic()

    def _check_acyclic(self) -> None:
        """Verify the edge list contains no directed cycle.

        Uses an iterative colour-marking depth-first search rather than recursion, so a pathological
        scenario with a long chain cannot blow the interpreter stack.

        Raises:
            ValueError: If a back edge is found, naming the node the cycle closes on.
        """
        white, grey, black = 0, 1, 2
        colour: Dict[str, int] = {node: white for node in self._parents}
        for root in list(colour):
            if colour[root] != white:
                continue
            stack: List[Tuple[str, bool]] = [(root, False)]
            while stack:
                node, leaving = stack.pop()
                if leaving:
                    colour[node] = black
                    continue
                if colour[node] == grey:
                    continue
                colour[node] = grey
                stack.append((node, True))
                for child in self._children.get(node, []):
                    if colour.get(child, white) == grey:
                        raise ValueError(f"cycle detected in SCM edges through node {child!r}; an SCM must be a DAG")
                    if colour.get(child, white) == white:
                        stack.append((child, False))

    def parents(self, node: str) -> Tuple[str, ...]:
        """Return the direct parents of ``node``.

        Args:
            node: Node to query.

        Returns:
            Parent names, de-duplicated, in first-seen edge order.
        """
        return _unique(self._parents.get(node, []))

    def children(self, node: str) -> Tuple[str, ...]:
        """Return the direct children of ``node``.

        Args:
            node: Node to query.

        Returns:
            Child names, de-duplicated, in first-seen edge order.
        """
        return _unique(self._children.get(node, []))

    def ancestors(self, node: str) -> Set[str]:
        """Return every node with a directed path INTO ``node``.

        Args:
            node: Node to query.

        Returns:
            Ancestor names as a set, excluding ``node`` itself.
        """
        return _reachable(self._parents, node)

    def descendants(self, node: str) -> Set[str]:
        """Return every node reachable FROM ``node`` along directed edges.

        Args:
            node: Node to query.

        Returns:
            Descendant names as a set, excluding ``node`` itself.
        """
        return _reachable(self._children, node)

    def markov_blanket(self) -> Tuple[str, ...]:
        """Return the Markov blanket of the target, restricted to observed columns.

        The blanket is parents, children and the other parents of those children. It is the optimal
        predictive set and it is unique under faithfulness and positivity, which is what makes it a valid
        scoring key. Latent blanket members are excluded here and reported as a caveat by
        :func:`build_target_sets` - an answer key naming an unobservable column cannot be answered.

        Returns:
            Blanket members in the graph's observed-column order.
        """
        blanket: Set[str] = set(self.parents(self.target)) | set(self.children(self.target))
        for child in self.children(self.target):
            blanket |= set(self.parents(child))
        blanket.discard(self.target)
        return tuple(name for name in self.observed if name in blanket)

    def causal_parents(self) -> Tuple[str, ...]:
        """Return the observed direct causes of the target.

        Returns:
            Parent columns in observed order. Unique as a set, but NOT identifiable from observational data
            alone, which is why it is never a headline key.
        """
        parents = set(self.parents(self.target))
        return tuple(name for name in self.observed if name in parents)

    def latent_blanket_members(self) -> Tuple[str, ...]:
        """Return blanket members that are latent, i.e. present in the truth but absent from the frame.

        Returns:
            Latent blanket member names, in declared latent order.
        """
        blanket: Set[str] = set(self.parents(self.target)) | set(self.children(self.target))
        for child in self.children(self.target):
            blanket |= set(self.parents(child))
        blanket.discard(self.target)
        return tuple(name for name in self.latents if name in blanket)


def _unique(names: Sequence[str]) -> Tuple[str, ...]:
    """Return ``names`` de-duplicated, preserving first-seen order.

    Order preservation matters: role and target-set output is compared across runs, and a set-derived
    order would be stable within a process but is not something to rely on across refactors.

    Args:
        names: Possibly repeating names.

    Returns:
        De-duplicated names.
    """
    seen: Set[str] = set()
    out: List[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return tuple(out)


def _reachable(adjacency: Dict[str, List[str]], start: str) -> Set[str]:
    """Return every node reachable from ``start`` through ``adjacency``, excluding ``start``.

    Args:
        adjacency: Either the parent map (giving ancestors) or the child map (giving descendants).
        start: Node to start from.

    Returns:
        Reachable node names.
    """
    seen: Set[str] = set()
    stack: List[str] = list(adjacency.get(start, []))
    while stack:
        node = stack.pop()
        if node in seen or node == start:
            continue
        seen.add(node)
        stack.extend(adjacency.get(node, []))
    return seen


def _declared_kind_role(graph: CausalGraph, node: str) -> Optional[FeatureRole]:
    """Return the role an incoming edge kind declares for ``node``, if any.

    Some roles are not determined by graph shape: an instrument and a plain causal ancestor have the same
    shape, and so do a proxy and any other child of a latent. Those are declared on the edge and read back
    here, so the declaration still lives in the one structure the generator consumes.

    Args:
        graph: The model.
        node: Column to classify.

    Returns:
        The declared role, or ``None`` when every incoming edge is a plain ``direct`` / ``latent`` arrow.
    """
    for edge in graph.edges:
        if edge.target == node and edge.kind in _KIND_TO_ROLE:
            return _KIND_TO_ROLE[edge.kind]
    return None


def _is_m_collider(graph: CausalGraph, node: str, ancestors_of_target: Set[str], descendants_of_target: Set[str]) -> bool:
    """Report whether ``node`` is an M-bias collider.

    The M-bias shape is ``U1 -> X``, ``U1 -> Z``, ``U2 -> Z``, ``U2 -> Y``: ``Z`` is marginally independent
    of both ``X`` and ``Y``, but conditioning on it CREATES an association between them. Structurally that
    is a node with two or more parents which is neither an ancestor nor a descendant of the target, and at
    least one of whose parents leads to the target.

    Args:
        graph: The model.
        node: Column to classify.
        ancestors_of_target: Precomputed ancestor set of the target.
        descendants_of_target: Precomputed descendant set of the target.

    Returns:
        ``True`` when the node matches the shape.
    """
    if node in ancestors_of_target or node in descendants_of_target:
        return False
    parents = graph.parents(node)
    if len(parents) < 2:
        return False
    return any(parent in ancestors_of_target or graph.target in graph.descendants(parent) for parent in parents)


def derive_roles(graph: CausalGraph, regions: Optional[Dict[str, GateSpec]] = None) -> Dict[str, FeatureRole]:
    """Assign one role to every observed column, from the graph alone.

    Args:
        graph: The structural model.
        regions: Optional per-column gates; a column with a gate and no stronger classification is
            ``REGIONAL`` rather than ``PROBE``, because "informative on 30% of the space" and "not
            informative" are different findings.

    Returns:
        Column name to role, in the graph's observed order. Precedence is the module docstring's list.
    """
    gates = regions or {}
    ancestors_of_target = graph.ancestors(graph.target)
    descendants_of_target = graph.descendants(graph.target)
    parents_of_target = set(graph.parents(graph.target))
    children_of_target = set(graph.children(graph.target))
    spouses: Set[str] = set()
    for child in children_of_target:
        spouses |= set(graph.parents(child))
    spouses -= {graph.target}

    roles: Dict[str, FeatureRole] = {}
    for name in graph.observed:
        declared = _declared_kind_role(graph, name)
        if declared is not None:
            roles[name] = declared
            continue
        if _is_m_collider(graph, name, ancestors_of_target, descendants_of_target):
            roles[name] = FeatureRole.M_COLLIDER
            continue
        if name in children_of_target:
            roles[name] = FeatureRole.CHILD
            continue
        if name in spouses:
            roles[name] = FeatureRole.SPOUSE
            continue
        if name in ancestors_of_target and _has_feature_ancestor(graph, name):
            roles[name] = FeatureRole.MEDIATOR
            continue
        if name in parents_of_target:
            roles[name] = FeatureRole.CAUSAL_PARENT
            continue
        if name in ancestors_of_target:
            roles[name] = FeatureRole.CAUSAL_ANCESTOR
            continue
        roles[name] = FeatureRole.REGIONAL if name in gates else FeatureRole.PROBE
    return roles


def _has_feature_ancestor(graph: CausalGraph, node: str) -> bool:
    """Report whether ``node`` has an OBSERVED ancestor, i.e. sits downstream of another feature.

    This is what separates a mediator from a plain root cause: both are ancestors of the target, but only
    the mediator transmits another feature's effect, and only the mediator is screened off when that
    feature is conditioned on.

    Args:
        graph: The model.
        node: Column to test.

    Returns:
        ``True`` when at least one ancestor is an observed feature.
    """
    return bool(graph.ancestors(node) & set(graph.observed))


def build_target_sets(
    graph: CausalGraph,
    redundancy_groups: Sequence[RedundancyGroup] = (),
) -> Dict[str, TargetSet]:
    """Derive the three named answer keys from the graph.

    ``minimal_sufficient`` is built by collapsing each redundancy group inside the blanket into ONE
    equivalence class: an arm that takes any single representative of an exact-copy group has answered
    correctly, and scoring it by set equality against one arbitrarily chosen representative would mark a
    correct answer wrong. Groups flagged ``exact=False`` (replicates carrying private deltas) are NOT
    collapsed, because there one representative is genuinely not sufficient - averaging them destroys the
    private component that moves the target.

    Args:
        graph: The structural model.
        redundancy_groups: Known redundancy structure, normally derived alongside the generator's own
            construction of the redundant columns.

    Returns:
        Mapping of set name to :class:`TargetSet` for ``markov_blanket``, ``minimal_sufficient`` and
        ``causal_parents``.
    """
    blanket = graph.markov_blanket()
    latent_members = graph.latent_blanket_members()
    blanket_caveats: Tuple[str, ...] = ()
    if latent_members:
        blanket_caveats = (
            f"blanket contains unobserved node(s) {list(latent_members)}; no arm can name them, so the "
            "observed key is achievable but the achievable ceiling is below the complete-data ceiling",
        )

    minimal_classes = _collapse_redundant(blanket, redundancy_groups)
    minimal_members = tuple(equivalence_class[0] for equivalence_class in minimal_classes)
    parents = graph.causal_parents()
    return {
        "markov_blanket": TargetSet(
            name="markov_blanket",
            members=blanket,
            classes=tuple((name,) for name in blanket),
            definition="parents, children and spouses of the target",
            unique=True,
            caveats=blanket_caveats,
        ),
        "minimal_sufficient": TargetSet(
            name="minimal_sufficient",
            members=minimal_members,
            classes=minimal_classes,
            definition="smallest subset attaining the ceiling; stored as an equivalence partition",
            unique=False,
            caveats=("not unique: score by class coverage, never by set equality",),
        ),
        "causal_parents": TargetSet(
            name="causal_parents",
            members=parents,
            classes=tuple((name,) for name in parents),
            definition="columns with a non-zero direct causal effect on the target",
            unique=True,
            caveats=("unique but not identifiable from observational data; never a headline key",),
        ),
    }


def _collapse_redundant(
    blanket: Sequence[str],
    redundancy_groups: Sequence[RedundancyGroup],
) -> Tuple[Tuple[str, ...], ...]:
    """Partition ``blanket`` into equivalence classes, merging exact-redundancy groups.

    Args:
        blanket: Blanket members in reporting order.
        redundancy_groups: Known groups; only those with ``exact=True`` are collapsed.

    Returns:
        The partition, each class ordered by the blanket's own order, and the classes themselves ordered by
        the position of their first member.
    """
    remaining = list(blanket)
    classes: List[Tuple[str, ...]] = []
    consumed: Set[str] = set()
    for name in remaining:
        if name in consumed:
            continue
        group_members = [name]
        for group in redundancy_groups:
            if group.exact and name in group.members:
                group_members = [member for member in remaining if member in set(group.members)]
                break
        consumed.update(group_members)
        classes.append(tuple(group_members))
    return tuple(classes)


def graph_from_spec(spec: DatasetSpec, target_name: Optional[str] = None) -> CausalGraph:
    """Build the causal graph declared by a dataset spec.

    Args:
        spec: The dataset specification.
        target_name: Which of the spec's targets to centre the graph on. Defaults to the first declared
            target, which is the only choice when a scenario declares exactly one.

    Returns:
        The corresponding :class:`CausalGraph`.

    Raises:
        ValueError: If the spec declares no target, or ``target_name`` is not one of its targets.
    """
    if not spec.targets:
        raise ValueError(f"spec {spec.name!r} declares no target, so no causal graph can be centred")
    declared = [target.name for target in spec.targets]
    chosen = target_name if target_name is not None else declared[0]
    if chosen not in declared:
        raise ValueError(f"target {chosen!r} is not declared by spec {spec.name!r}; declared: {declared}")
    return CausalGraph(
        edges=edges_from_specs(spec.edges),
        target=chosen,
        observed=spec.feature_names(),
        latents=tuple(latent.name for latent in spec.latents),
    )


def build_ground_truth(
    spec: DatasetSpec,
    target_name: Optional[str] = None,
    redundancy_groups: Sequence[RedundancyGroup] = (),
    regions: Optional[Dict[str, GateSpec]] = None,
) -> GroundTruth:
    """Assemble the structural half of a dataset's ground truth from its spec.

    Only structural truth is filled in: roles, edges, redundancy groups and the three target sets. The
    statistical half (Bayes ceiling, reference MI) stays behind
    :meth:`~mlframe.data.datasets.ground_truth.GroundTruth.ceiling` and
    :meth:`~mlframe.data.datasets.ground_truth.GroundTruth.mi_reference`, and ``true_prob`` / ``true_mean``
    are attached by the generator when it realises the data.

    Args:
        spec: The dataset specification.
        target_name: Which target to build truth for; defaults to the first declared.
        redundancy_groups: Known redundancy structure to record and to collapse in the minimal-sufficient
            partition.
        regions: Optional per-column gates, used both for the ``REGIONAL`` role and for
            ``FeatureTruth.region``.

    Returns:
        A :class:`~mlframe.data.datasets.ground_truth.GroundTruth` carrying the structural truth.
    """
    graph = graph_from_spec(spec, target_name=target_name)
    gates = regions or {}
    roles = derive_roles(graph, regions=gates)
    source_of = {edge.target: edge.source for edge in graph.edges if edge.kind in ("redundant_exact", "redundant_noisy", "proxy")}
    costs = {feature.name: feature.cost for feature in spec.features}
    features = {
        name: FeatureTruth(
            role=roles[name],
            region=gates.get(name),
            source=source_of.get(name),
            cost=costs.get(name, 1.0),
        )
        for name in graph.observed
    }
    return GroundTruth(
        features=features,
        redundancy_groups=tuple(redundancy_groups),
        target_sets=build_target_sets(graph, redundancy_groups),
        graph=graph.edges,
        target_name=graph.target,
    )
