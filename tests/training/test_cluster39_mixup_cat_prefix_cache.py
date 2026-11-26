"""Cluster 39 regression: the mixup categorical-prefix lookup is computed once per network (audit 2026-11-20, finding 39-F71).

``_mixup_cat_prefix`` returns the number of leading categorical columns and memoises it against the network object. The module walk runs only
on the first call, and a replaced network is recomputed.
"""

import pytest

pytest.importorskip("lightning")  # the neural package subclasses Lightning classes: shards without the neural extras skip this module

import torch.nn as nn

from mlframe.training.neural._categorical_embeddings import CategoricalEmbedding
from mlframe.training.neural._flat_torch_module import MLPTorchModel


class _Holder:
    """Minimal stand-in carrying only the ``network`` attribute the helper reads."""

    def __init__(self, network):
        """Store the network the helper should inspect."""
        self.network = network


def _call(holder):
    """Invoke the unbound helper on a stand-in object."""
    return MLPTorchModel._mixup_cat_prefix(holder)


def test_returns_number_of_categorical_columns():
    """The prefix equals the embedding's cardinality count (39-F71)."""
    net = nn.Sequential(CategoricalEmbedding(cardinalities=[3, 5]), nn.Linear(2, 1))
    assert _call(_Holder(net)) == 2


def test_result_is_cached_per_network(monkeypatch):
    """A second call on the same network does not walk the module tree again (39-F71)."""
    net = nn.Sequential(CategoricalEmbedding(cardinalities=[4]), nn.Linear(1, 1))
    holder = _Holder(net)
    assert _call(holder) == 1

    def _boom(*_a, **_k):
        """Fail the test if the module tree is walked a second time."""
        raise AssertionError("module walk ran again")

    monkeypatch.setattr(net, "modules", _boom)
    assert _call(holder) == 1


def test_replaced_network_is_recomputed():
    """Control: swapping the network object invalidates the cache (39-F71)."""
    holder = _Holder(nn.Sequential(CategoricalEmbedding(cardinalities=[2]), nn.Linear(1, 1)))
    assert _call(holder) == 1
    holder.network = nn.Sequential(nn.Linear(1, 1))
    assert _call(holder) == 0
