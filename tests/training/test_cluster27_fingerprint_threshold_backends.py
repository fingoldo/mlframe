"""Cluster 27 regressions: slot-aware fingerprints, the positive class in honest thresholds, and requested RFECV backends (audit 2026-11-20, findings 27-F3, 27-F4, 27-F6)."""

import numpy as np

from mlframe.training._honest_decision_threshold import _counts_at
from mlframe.training._model_cache_fingerprint import _stable
from mlframe.training._trainer_configure import _backend_requested


class _Weighted:
    """A metric-like object whose weights live in __slots__ and which has no get_params."""

    __slots__ = ("weights",)

    def __init__(self, weights):
        """Store the weights."""
        self.weights = weights


def test_fingerprint_changes_when_a_slot_weight_changes():
    """Two metrics differing only in one slot weight used to share a fingerprint, since only the class name was digested (27-F3)."""
    assert _stable(_Weighted([1.0, 2.0])) != _stable(_Weighted([1.0, 3.0]))


def test_fingerprint_is_stable_for_equal_slot_state():
    """Equal state gives equal fingerprints."""
    assert _stable(_Weighted([1.0, 2.0])) == _stable(_Weighted([1.0, 2.0]))


def test_counts_treat_the_larger_label_as_positive():
    """Labels {1, 2}: class 2 is positive. The former ``y > 0.5`` made every row positive (27-F4)."""
    y = np.array([1.0, 2.0, 2.0, 1.0])
    p = np.array([0.9, 0.9, 0.1, 0.1])
    counts = _counts_at(y, p, 0.5)
    assert counts == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}


def test_backend_requested_when_unset_or_named():
    """An unset model list allows every backend; a named list allows only its members (27-F6)."""
    assert _backend_requested("cb_rfecv", None)
    assert _backend_requested("cb_rfecv", ["cb_rfecv", "lgb_rfecv"])
    assert not _backend_requested("lgb_rfecv", ["cb_rfecv"])
