"""Cluster 18 regression: the regression train-split path is chosen by any member, not by member 0 (audit 2026-11-20, finding 18-F10).

When member 0 has neither OOF nor train predictions but another member does, the train-split predictions are kept (with None for the empty
member), as the classification path already does. Before the fix the whole list was dropped.
"""

from types import SimpleNamespace

import numpy as np

from mlframe.models.ensembling.process_method import _collect_train_predictions


def test_member_zero_empty_does_not_drop_the_train_ensemble():
    """A member with no predictions first, a member with OOF second: the second member's predictions are kept (18-F10)."""
    empty = SimpleNamespace(oof_preds=None, train_preds=None)
    has_oof = SimpleNamespace(oof_preds=np.array([1.0, 2.0, 3.0]), train_preds=None)
    out = _collect_train_predictions(True, [empty, has_oof], [], False)
    assert len(out) == 2
    assert out[0] is None
    assert np.allclose(np.asarray(out[1]).reshape(-1), [1.0, 2.0, 3.0])


def test_no_member_with_predictions_gives_an_empty_list():
    """Control: when no member has predictions the result is still an empty list (18-F10)."""
    empty = SimpleNamespace(oof_preds=None, train_preds=None)
    assert _collect_train_predictions(True, [empty, empty], [], False) == []
