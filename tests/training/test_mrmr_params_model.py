"""MRMRParams: strict, frozen parameter model whose defaults equal the MRMR constructor's and whose values build a working MRMR."""

from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from mlframe.feature_selection.filters import MRMR
from mlframe.training.fs_params.mrmr import MRMRParams

_SKIPPED_DEFAULTS = set(MRMRParams.__signature_skip_default__)


def test_defaults_match_the_mrmr_constructor_defaults():
    """Every field that has a constructor default carries the same default, except the ones the generator deliberately leaves out."""
    sig = inspect.signature(MRMR.__init__)
    params = MRMRParams()
    mismatched = []
    for name, parameter in sig.parameters.items():
        if name in {"self", "args", "kwargs"} or name in _SKIPPED_DEFAULTS or parameter.default is inspect.Parameter.empty:
            continue
        if getattr(params, name) != parameter.default:
            mismatched.append((name, getattr(params, name), parameter.default))
    assert mismatched == []


def test_field_set_equals_constructor_parameter_set():
    """No constructor parameter is missing from the model and the model has no extra field."""
    sig = inspect.signature(MRMR.__init__)
    expected = {name for name in sig.parameters if name not in {"self", "args", "kwargs"}}
    assert set(MRMRParams.model_fields) == expected


def test_unknown_name_wrong_type_and_out_of_enum_value_are_rejected():
    """extra='forbid', int-typed fields refuse strings and Literal fields refuse values outside the enum."""
    with pytest.raises(ValidationError):
        MRMRParams(not_a_param=1)
    with pytest.raises(ValidationError):
        MRMRParams(quantization_nbins="many")
    with pytest.raises(ValidationError):
        MRMRParams(quantization_method="nope")


def test_instance_is_frozen_and_overrides_are_kept():
    """Assignment after construction fails; a constructor override is stored while other fields keep their defaults."""
    params = MRMRParams(quantization_nbins=7)
    assert params.quantization_nbins == 7
    assert params.quantization_method == "quantile"
    with pytest.raises(ValidationError):
        params.quantization_nbins = 3


def test_dumped_values_construct_an_mrmr_with_the_same_settings():
    """The model's values are valid constructor kwargs: the resulting MRMR reports the overridden and the default settings."""
    params = MRMRParams(quantization_nbins=6, quantization_method="uniform")
    kwargs = {k: v for k, v in params.model_dump().items() if k not in _SKIPPED_DEFAULTS}
    selector = MRMR(**kwargs)
    assert selector.quantization_nbins == 6
    assert selector.quantization_method == "uniform"
