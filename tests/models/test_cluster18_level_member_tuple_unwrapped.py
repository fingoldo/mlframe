"""Cluster 18 regression: a level-1 member tuple is read through its entry namespace (audit 2026-11-20, finding 18-F4)."""

from types import SimpleNamespace

import numpy as np

from mlframe.models.ensembling.process_method import _oof_or_train, _select_member_probs


def _level_member():
    """``train_and_evaluate_model``'s return shape: (entry, train_df, val_df, test_df)."""
    entry = SimpleNamespace(val_probs=np.array([0.2, 0.8]), oof_probs=np.array([0.1, 0.9]))
    return (entry, None, None, None)


def test_select_member_probs_reads_the_entry_inside_the_tuple():
    """The split probabilities come from the entry, not from the tuple (18-F4)."""
    out = _select_member_probs(_level_member(), "val", use_calibrated=False)
    assert out is not None and np.allclose(out, [0.2, 0.8])


def test_oof_or_train_reads_the_entry_inside_the_tuple():
    """OOF predictions of a tuple member are found (18-F4)."""
    fallback = []
    out = _oof_or_train(_level_member(), "oof_probs", "train_probs", 0, fallback)
    assert out is not None and np.allclose(out, [0.1, 0.9])
    assert fallback == []
