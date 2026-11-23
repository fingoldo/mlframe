"""Cluster 04 regression: duplicate scenario names are refused, not silently overwritten (audit 2026-11-20, finding 04-F32)."""

import pytest

from mlframe.data.datasets import scenarios as sc


class _Fake:
    """Minimal stand-in carrying only the ``name`` attribute the index reads."""

    def __init__(self, name):
        """Store the scenario name."""
        self.name = name


def test_duplicate_name_is_refused():
    """Two scenarios with the same name raise instead of the second replacing the first (04-F32)."""
    with pytest.raises(ValueError, match="duplicate scenario name 'x'"):
        sc._index_by_name((_Fake("x"), _Fake("x")))


def test_unique_names_are_all_indexed():
    """Control: distinct names are all kept (04-F32)."""
    a, b = _Fake("a"), _Fake("b")
    assert sc._index_by_name((a, b)) == {"a": a, "b": b}


def test_shipped_registry_has_unique_names():
    """The built-in registry builds without a duplicate (04-F32)."""
    assert len(sc.names()) == len(set(sc.names()))
