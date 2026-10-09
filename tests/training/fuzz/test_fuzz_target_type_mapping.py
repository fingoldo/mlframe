"""Regression tests for the fuzz suites' target-type mapping: a quantile-regression combo must not be treated as classification."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlframe.training.configs import TargetTypes

from ._target_type import target_type_for_combo


@pytest.mark.parametrize(
    "name, expected",
    [
        ("regression", TargetTypes.REGRESSION),
        ("quantile_regression", TargetTypes.QUANTILE_REGRESSION),
        ("binary_classification", TargetTypes.BINARY_CLASSIFICATION),
        ("multiclass_classification", TargetTypes.MULTICLASS_CLASSIFICATION),
        ("multilabel_classification", TargetTypes.MULTILABEL_CLASSIFICATION),
        ("learning_to_rank", TargetTypes.LEARNING_TO_RANK),
    ],
)
def test_each_combo_target_type_maps_to_its_own_member(name, expected) -> None:
    """Every axis value maps to the matching member, in particular quantile regression, which a boolean flag turned into classification."""
    assert target_type_for_combo(SimpleNamespace(target_type=name), "target") is expected


def test_a_multi_target_combo_keeps_its_type_only_when_the_frame_has_a_2d_target() -> None:
    """The frame builder emits ``target`` for a native 2-D combo and the 1-D ``target_reg`` for a downgraded one, which is plain regression."""
    combo = SimpleNamespace(target_type="multi_target_regression")
    assert target_type_for_combo(combo, "target") is TargetTypes.MULTI_TARGET_REGRESSION
    assert target_type_for_combo(combo, "target_reg") is TargetTypes.REGRESSION


def test_a_quantile_regression_combo_is_no_longer_resolved_as_binary_classification() -> None:
    """The boolean flag alone resolved a quantile-regression combo to BINARY_CLASSIFICATION; the explicit type resolves it to QUANTILE_REGRESSION."""
    from tests.training.shared import SimpleFeaturesAndTargetsExtractor

    combo = SimpleNamespace(target_type="quantile_regression")
    flag_only = SimpleFeaturesAndTargetsExtractor(target_column="target_reg", regression=(combo.target_type == "regression"))
    assert flag_only._resolve_target_type() is TargetTypes.BINARY_CLASSIFICATION
    explicit = SimpleFeaturesAndTargetsExtractor(
        target_column="target_reg",
        regression=(combo.target_type == "regression"),
        target_type=target_type_for_combo(combo, "target_reg"),
    )
    assert explicit._resolve_target_type() is TargetTypes.QUANTILE_REGRESSION
