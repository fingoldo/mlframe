"""Audit 2026-11-20 finding 04-F26: a column whose parent is another declared target is refused, not silently generated.

Before the fix the parent was dropped from the realisation while the truth graph still recorded the edge. The check is
that _feature_parents raises naming the dropped parent, and that a realisable parent set is returned unchanged.
"""

import pytest

from mlframe.data.datasets._derived import _feature_parents
from mlframe.data.datasets._scm import CausalGraph
from mlframe.data.datasets.ground_truth import Edge


def test_parent_that_is_another_target_is_refused():
    """y2 -> col with y2 neither observed nor the selected target raises, and the message names y2."""
    graph = CausalGraph([Edge(source="a", target="col", kind="causal"), Edge(source="y2", target="col", kind="causal")], target="y", observed=("a", "col"))
    with pytest.raises(ValueError, match="y2"):
        _feature_parents(graph, "col", {"a", "col"}, "y")


def test_realisable_parents_are_returned_unchanged():
    """Observed parents and the selected target are kept, so a valid graph still realises."""
    graph = CausalGraph([Edge(source="a", target="col", kind="causal"), Edge(source="y", target="col", kind="causal")], target="y", observed=("a", "col"))
    assert set(_feature_parents(graph, "col", {"a", "col"}, "y")) == {"a", "y"}
