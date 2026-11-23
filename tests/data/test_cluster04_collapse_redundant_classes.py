"""Cluster 04 regression: exact-redundancy groups collapse into one class with the set hoisted out of the loop (audit 2026-11-20, finding 04-F28).

The partition must be unchanged by the refactor: exact groups merge their blanket members into one class, a non-exact group does not,
and an unrelated column stays a singleton.
"""

from mlframe.data.datasets._scm import _collapse_redundant
from mlframe.data.datasets.ground_truth import RedundancyGroup


def test_exact_group_collapses_its_blanket_members():
    """Two exact copies in the blanket form one class; the unrelated column is its own class (04-F28)."""
    groups = [RedundancyGroup(members=("a", "b"), rank=1, exact=True)]
    classes = _collapse_redundant(["a", "b", "c"], groups)
    assert classes == (("a", "b"), ("c",))


def test_non_exact_group_does_not_merge():
    """Replicates with private deltas are not collapsed (04-F28)."""
    groups = [RedundancyGroup(members=("a", "b"), rank=2, exact=False)]
    classes = _collapse_redundant(["a", "b"], groups)
    assert classes == (("a",), ("b",))
