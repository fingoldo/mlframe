"""Audit 2026-11-20 finding 04-F19: FeatureTruth declares only the fields the generator populates.

The three MI/R^2 fields were never set by any code path, so a consumer reading them got None without warning. They were
removed; this test pins the field set so they cannot quietly come back as silent placeholders.
"""

import dataclasses

from mlframe.data.datasets.ground_truth import FeatureTruth


def test_feature_truth_declares_no_unpopulated_mi_or_r2_fields():
    """The removed placeholder fields are gone from the dataclass."""
    names = {field.name for field in dataclasses.fields(FeatureTruth)}
    assert "marginal_mi" not in names
    assert "conditional_mi_given_rest" not in names
    assert "r2_to_source" not in names


def test_feature_truth_keeps_the_fields_the_generator_fills():
    """role, region, source and cost, which build_ground_truth passes, remain declared."""
    names = {field.name for field in dataclasses.fields(FeatureTruth)}
    assert {"role", "region", "source", "cost"} <= names
