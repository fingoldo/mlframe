"""Audit 2026-11-20 finding 28-F34: the OOF target is found on the inner model the same way the OOF probabilities are.

An entry whose probabilities and targets both live on its inner model used to skip calibration with "oof_target absent" while the
rows paired. The shared helper now finds the target on the entry first and falls back to the inner model.
"""

from types import SimpleNamespace

import numpy as np

from mlframe.training.honest_diagnostics import _entry_oof_target


def test_target_on_the_inner_model_is_found():
    """A target held only on model.oof_target is returned, as the probabilities are."""
    target = np.array([0, 1, 1, 0])
    entry = SimpleNamespace(model=SimpleNamespace(oof_probs=np.array([0.1, 0.9, 0.8, 0.2]), oof_target=target))
    assert np.array_equal(_entry_oof_target(entry), target)


def test_target_on_the_entry_wins_over_the_inner_model():
    """A target on the entry itself is returned as before."""
    entry = SimpleNamespace(oof_target=np.array([1, 0]), model=SimpleNamespace(oof_target=np.array([0, 0])))
    assert np.array_equal(_entry_oof_target(entry), np.array([1, 0]))
