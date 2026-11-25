"""Cluster 38 regression: a CV policy without a target type is refused, not wired as regression (audit 2026-11-20, finding 38-F24)."""

import pytest

from mlframe.training.core._setup_helpers_pre_pipelines import _wire_cv_policies


def test_cv_policy_without_target_type_is_refused():
    """A supplied CV policy with target_type None raises instead of silently choosing regression (38-F24)."""
    with pytest.raises(ValueError, match="target_type"):
        _wire_cv_policies([], cv_policy=object(), target_type=None)


def test_no_policy_is_a_no_op_without_target_type():
    """Control: with no CV policy the list is returned unchanged, target_type not needed (38-F24)."""
    pipelines = [object()]
    assert _wire_cv_policies(pipelines, cv_policy=None, target_type=None) is pipelines
