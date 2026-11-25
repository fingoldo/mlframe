"""Cluster 04 regression: derived-column ordering is unchanged by the set-based rewrite (audit 2026-11-20, finding 04-F27).

The ordering loop now tests membership against a per-round set and drops ready names through a set. These controls pin
that a chain orders parents first.
"""

from mlframe.data.datasets._derived import derived_order
from mlframe.data.datasets._scm import CausalGraph
from mlframe.data.datasets.ground_truth import Edge


def test_chain_orders_parents_before_children():
    """a -> b -> c with all three derived after the target y: b comes before c, and both are in the early or late list (04-F27)."""
    graph = CausalGraph(
        [Edge(source="y", target="a", kind="causal"), Edge(source="a", target="b", kind="causal"), Edge(source="b", target="c", kind="causal")],
        target="y",
        observed=("a", "b", "c"),
    )
    early, late = derived_order(graph, ("a", "b", "c"), "y")
    ordered = list(early) + list(late)
    assert ordered.index("a") < ordered.index("b") < ordered.index("c")
