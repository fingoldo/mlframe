"""Audit 2026-11-20 finding 37-F4: the member score can come from the OOF surface even when a validation metric exists.

With use_val=False a member with a validation roc_auc is scored from its OOF surface, so a mixed call compares one scale.
"""

from types import SimpleNamespace

import numpy as np

from mlframe.training.core._diversity_recommendations import _member_individual_score


def test_use_val_false_ignores_the_validation_metric():
    """A member with val roc_auc 0.9 gets its OOF-derived score when use_val is False."""
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4])
    member = SimpleNamespace(metrics={"val": {"roc_auc": 0.9}})
    with_val = _member_individual_score(member, True, oof_fallback=(y, p), use_val=True)
    oof_only = _member_individual_score(member, True, oof_fallback=(y, p), use_val=False)
    assert with_val == 0.9
    assert oof_only != with_val
    assert oof_only < 0.0  # negated log-loss, the OOF scale
