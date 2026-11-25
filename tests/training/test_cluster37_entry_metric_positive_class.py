"""Cluster 37 regression: a binary class-indexed metric reads the positive class (audit 2026-11-20, finding 37-F49).

``_entry_metric`` walked class keys in ascending order, so a binary layout returned class 0's value, while its comment says binary reports the
positive class (1). For a {0, 1} layout the positive class is now read first. A multiclass layout keeps the ascending order.
"""

from types import SimpleNamespace

from mlframe.training.core._misc_helpers import _entry_metric


def _entry(metrics):
    """A minimal entry exposing ``metrics``."""
    return SimpleNamespace(metrics=metrics)


def test_binary_reads_positive_class():
    """With classes {0, 1} the value of class 1 is returned (37-F49)."""
    entry = _entry({"test": {0: {"auc": 0.2}, 1: {"auc": 0.9}}})
    assert _entry_metric(entry, "test", "auc") == 0.9


def test_multiclass_keeps_first_class_order():
    """Control: a multiclass layout still takes the first class that carries the metric (37-F49)."""
    entry = _entry({"test": {0: {"auc": 0.3}, 1: {"auc": 0.6}, 2: {"auc": 0.9}}})
    assert _entry_metric(entry, "test", "auc") == 0.3
