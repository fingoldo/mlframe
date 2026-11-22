"""Audit 2026-11-20 finding 35-F39: both stacking passes derive their group ids and ranking through one helper each.

The two passes used to repeat the ranking and the group derivation. These tests pin the shared contracts.
"""

from types import SimpleNamespace

import numpy as np

from mlframe.training.composite.discovery import _stacked


def test_train_group_ids_slices_the_train_rows():
    """With group ids covering the frame, the train rows' ids come back in train order."""
    groups = np.arange(10) % 3
    self_ = SimpleNamespace(_group_ids_for_rerank=groups)
    out = _stacked._train_group_ids(self_, np.array([2, 5, 7]))
    assert out.tolist() == [groups[2], groups[5], groups[7]]


def test_train_group_ids_is_none_without_covering_ids():
    """No ids, or ids too short for the train rows, give None."""
    assert _stacked._train_group_ids(SimpleNamespace(), np.array([0, 1])) is None
    short = SimpleNamespace(_group_ids_for_rerank=np.arange(3))
    assert _stacked._train_group_ids(short, np.array([0, 5])) is None
