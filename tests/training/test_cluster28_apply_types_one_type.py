"""Cluster 28 regression: the two same-named target-type fields use one type (audit 2026-11-20, finding 28-F69).

Both ``apply_to_target_types`` fields are frozensets, and list or tuple input is coerced to a frozenset by the pydantic config.
"""

from mlframe.training.configs import BaselineDiagnosticsConfig, DummyBaselinesConfig


def test_both_fields_are_frozensets():
    """The default of each same-named field is a frozenset (28-F69)."""
    assert isinstance(BaselineDiagnosticsConfig().apply_to_target_types, frozenset)
    assert isinstance(DummyBaselinesConfig().apply_to_target_types, frozenset)


def test_list_input_is_coerced_to_frozenset():
    """A list passed by a caller becomes a frozenset in both configs, so set operations work on either (28-F69)."""
    assert BaselineDiagnosticsConfig(apply_to_target_types=["regression"]).apply_to_target_types == frozenset({"regression"})
    assert DummyBaselinesConfig(apply_to_target_types=["regression"]).apply_to_target_types == frozenset({"regression"})
