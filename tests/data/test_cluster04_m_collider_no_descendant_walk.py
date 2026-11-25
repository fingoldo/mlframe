"""Cluster 04 regression: the M-bias collider test needs no per-parent descendant walk (audit 2026-11-20, finding 04-F28).

A parent reaches the target exactly when it is an ancestor of the target, so membership in the precomputed ancestor set is the same test.
"""

from mlframe.data.datasets._scm import _is_m_collider


class _Graph:
    """Minimal graph stand-in: parents, reachable descendants, and the target name."""

    def __init__(self, parents, children, target):
        """Store the parent and child maps and the target."""
        self._parents = parents
        self._children = children
        self.target = target

    def parents(self, node):
        """Return the parents of a node."""
        return tuple(self._parents.get(node, ()))

    def descendants(self, node):
        """Return every node reachable from ``node`` along child edges."""
        out, stack = set(), list(self._children.get(node, ()))
        while stack:
            n = stack.pop()
            if n not in out:
                out.add(n)
                stack.extend(self._children.get(n, ()))
        return out


def _m_bias_graph():
    """U1 -> X, U1 -> Z, U2 -> Z, U2 -> Y: Z is an M-bias collider for target Y."""
    parents = {"X": ["U1"], "Z": ["U1", "U2"], "Y": ["U2"]}
    children = {"U1": ["X", "Z"], "U2": ["Z", "Y"]}
    return _Graph(parents, children, "Y")


def _ancestors(graph, node):
    """Return every node from which ``node`` is reachable (the ancestor set the caller passes)."""
    return {n for n in ["U1", "U2", "X", "Z", "Y"] if node in graph.descendants(n)}


def test_m_bias_collider_is_detected_without_descendant_walk():
    """Z has two parents and one of them (U2) leads to the target Y, so it is a collider (04-F28)."""
    g = _m_bias_graph()
    anc = _ancestors(g, "Y")
    desc = g.descendants("Y")
    assert _is_m_collider(g, "Z", anc, desc) is True


def test_unrelated_two_parent_node_is_not_a_collider():
    """Control: a node whose parents do not lead to the target is not flagged (04-F28)."""
    parents = {"W": ["A", "B"], "Y": ["C"]}
    children = {"A": ["W"], "B": ["W"], "C": ["Y"]}
    g = _Graph(parents, children, "Y")
    anc = _ancestors(g, "Y")
    desc = g.descendants("Y")
    assert _is_m_collider(g, "W", anc, desc) is False
