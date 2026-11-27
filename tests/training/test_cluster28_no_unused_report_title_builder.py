"""Cluster 28 regression: the report-title builder whose result nothing used is gone (audit 2026-11-20, 28-F48)."""

from mlframe.training import _trainer_train_and_evaluate as main_mod
from mlframe.training import _trainer_train_and_evaluate_parts as parts


def test_unused_title_builder_is_removed_from_both_modules():
    """Neither the parts module nor the orchestrator defines or imports the discarded title builder (28-F48)."""
    assert not hasattr(parts, "_train_and_evaluate_train_df_none")
    assert not hasattr(main_mod, "_train_and_evaluate_train_df_none")
