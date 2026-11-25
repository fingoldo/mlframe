"""Cluster 04 regression: declared kind roles are read from one precomputed map, with the old first-edge-wins rule (audit 2026-11-20, finding 04-F28).

Before, every observed column rescanned the whole edge list to find its declared role. The map is now built once per derivation. These
controls pin the marking convention (copy and proxy kinds mark their target, an instrument marks its source), that a plain ``direct`` edge
declares nothing, and that when two edges declare the same node the first one in graph order wins.
"""

from mlframe.data.datasets._scm import CausalGraph, FeatureRole, _declared_kind_roles, derive_roles
from mlframe.data.datasets.ground_truth import Edge


def _graph(edges, observed):
    """Build a graph with target ``y`` over the given edges and observed columns."""
    return CausalGraph(list(edges), target="y", observed=tuple(observed))


def test_declared_map_follows_marking_convention_and_skips_direct_edges():
    """Copy and proxy kinds mark their target, instrument marks its source, and a direct edge declares nothing (04-F28)."""
    graph = _graph(
        [
            Edge(source="y", target="a", kind="direct"),
            Edge(source="y", target="b", kind="redundant_exact"),
            Edge(source="z", target="c", kind="proxy"),
            Edge(source="d", target="y", kind="instrument"),
        ],
        observed=("a", "b", "c", "d"),
    )
    roles = _declared_kind_roles(graph)
    assert roles == {
        "b": FeatureRole.REDUNDANT_EXACT,
        "c": FeatureRole.PROXY,
        "d": FeatureRole.INSTRUMENT,
    }


def test_first_declaring_edge_wins_for_a_node_declared_twice():
    """Two declaring edges on one node: the earlier edge in graph order decides the role (04-F28)."""
    graph = _graph(
        [
            Edge(source="y", target="b", kind="proxy"),
            Edge(source="y", target="b", kind="redundant_exact"),
        ],
        observed=("b",),
    )
    assert _declared_kind_roles(graph) == {"b": FeatureRole.PROXY}


def test_derive_roles_uses_declared_role_for_marked_columns():
    """derive_roles assigns the declared role to the marked columns and still classifies the rest (04-F28)."""
    graph = _graph(
        [
            Edge(source="y", target="a", kind="direct"),
            Edge(source="y", target="b", kind="redundant_exact"),
            Edge(source="z", target="c", kind="proxy"),
        ],
        observed=("a", "b", "c"),
    )
    roles = derive_roles(graph)
    assert roles["b"] is FeatureRole.REDUNDANT_EXACT
    assert roles["c"] is FeatureRole.PROXY
    assert "a" in roles
