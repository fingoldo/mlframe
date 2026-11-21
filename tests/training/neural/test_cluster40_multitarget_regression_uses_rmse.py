"""Regression tests for the audit findings in this module."""
import pytest
"""Audit 2026-11-20 finding 40-F3: multi-target regression monitors RMSE, not the classification metric ICE."""

pytest.importorskip("lightning")
from mlframe.training.neural.base._base_fit_steps import _metric_branch_for


def test_multi_target_regression_does_not_take_the_ice_branch():
    """num_classes=K with the multi-target flag set is regression: no ICE metric."""
    name, is_ice = _metric_branch_for(num_classes=3, is_multi_target_regression=True)
    assert is_ice is False
    assert name != "ICE"


def test_true_classification_still_takes_the_ice_branch():
    """num_classes > 1 without the regression flag is classification: ICE as before."""
    name, is_ice = _metric_branch_for(num_classes=3, is_multi_target_regression=False)
    assert is_ice is True and name == "ICE"
