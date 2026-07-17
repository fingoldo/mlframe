"""brier_and_precision_score must surface a real input-contract violation instead of
silently returning 0.0 (the worst score), which would corrupt model selection."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics._core_auc_brier import brier_and_precision_score


def test_multiclass_input_raises_not_silent_zero():
    # Multiclass y_true fed to a binary precision_score is a real contract violation.
    # Pre-fix: a bare ``except Exception: return 0.0`` swallowed it -> the scorer reported
    # the worst value, silently changing which model wins. Post-fix it propagates.
    # One multiclass label among 0/1 with matching proba keeps brier <= 0.25 so the brier gate
    # passes and execution reaches precision_score, which raises on the multiclass y_true.
    """Multiclass input raises not silent zero."""
    y_true = np.array([0, 1, 1, 1, 2])
    y_proba = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        brier_and_precision_score(y_true, y_proba)


def test_valid_binary_input_scores_normally():
    """Valid binary input scores normally."""
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.7, 0.05])
    out = brier_and_precision_score(y_true, y_proba)
    assert np.isfinite(out)
