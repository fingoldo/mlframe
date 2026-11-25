"""Cluster 37 regression: the chain factories take no unused short_name, and the registry still builds its chain transforms (audit 2026-11-20, finding 37-F35).

short_name was accepted by both factories and never read. It is removed from the signatures and from every registry call site; the registry
keeps registering its chain transforms.
"""

import inspect

import pytest

from mlframe.training.composite.transforms import nonlinear


@pytest.mark.parametrize("factory", [nonlinear._make_chain_transform, nonlinear._make_multi_chain_transform])
def test_factories_do_not_accept_short_name(factory):
    """The unused short_name parameter is gone from both chain factories (37-F35)."""
    assert "short_name" not in inspect.signature(factory).parameters


def test_registry_still_builds_the_chain_transforms():
    """The registry imports cleanly and its chain transforms are present (37-F35)."""
    from mlframe.training.composite.transforms import registry

    assert hasattr(registry, "_make_chain_transform")
    assert hasattr(registry, "_make_multi_chain_transform")
