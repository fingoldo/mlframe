"""Cluster 39 regression: CustomParams rejects undeclared keys (audit 2026-11-20, finding 39-F47).

The model used ``extra="allow"``, so a misspelled field such as ``outptu_kind`` was accepted and silently ignored. It now forbids extra keys like
the other params models, and a valid construction with the declared fields still works.
"""

import pydantic
import pytest
from sklearn.preprocessing import StandardScaler

from mlframe.training.feature_handling.handlers import CustomParams


def test_misspelled_field_is_rejected():
    """An undeclared key raises a validation error instead of being ignored (39-F47)."""
    with pytest.raises(pydantic.ValidationError):
        CustomParams(transformer=StandardScaler(), outptu_kind="sparse")


def test_declared_fields_still_construct():
    """Control: the declared fields, including an opaque transformer object, still construct (39-F47)."""
    params = CustomParams(transformer=StandardScaler(), output_kind="sparse")
    assert params.output_kind == "sparse"
    assert isinstance(params.transformer, StandardScaler)
