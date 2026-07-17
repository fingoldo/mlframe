"""Regression: score_ensemble must reject a mixed classifier+regressor
member list, not silently miscategorise it.

Pre-fix: dispatch keyed off ``level_models_and_predictions[0]`` only.
A classifier at index 0 + regressor at index 1 quietly ran the
classification path on the regressor's predictions.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from mlframe.models.ensembling import score_ensemble


def _clf_member(n: int = 10, k: int = 3):
    """Helper: Clf member."""
    rng = np.random.default_rng(0)
    return types.SimpleNamespace(
        val_probs=rng.random((n, k)),
        test_probs=rng.random((n, k)),
        train_probs=rng.random((n, k)),
        val_preds=None,
        test_preds=None,
        train_preds=None,
    )


def _reg_member(n: int = 10):
    """Helper: Reg member."""
    rng = np.random.default_rng(1)
    return types.SimpleNamespace(
        val_probs=None,
        test_probs=None,
        train_probs=None,
        val_preds=rng.normal(size=n),
        test_preds=rng.normal(size=n),
        train_preds=rng.normal(size=n),
    )


def test_score_ensemble_rejects_mixed_clf_and_reg_members():
    """Score ensemble rejects mixed clf and reg members."""
    members = [_clf_member(), _reg_member(), _clf_member()]
    with pytest.raises(ValueError) as exc_info:
        score_ensemble(models_and_predictions=members, ensemble_name="t")
    msg = str(exc_info.value)
    assert "uniform member types" in msg
    # Message must surface which indices are which.
    assert "0" in msg and "1" in msg
