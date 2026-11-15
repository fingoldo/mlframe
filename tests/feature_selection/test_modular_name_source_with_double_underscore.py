"""A modular column built on a source whose own name contains ``__`` must parse back to that source, so its recipe is not silently dropped."""

import pytest

from mlframe.feature_selection.filters._periodic_fe import _parse_modular_name, engineered_name_modular


@pytest.mark.parametrize("src", ["b", "rankgauss__b", "a__b__c"])
@pytest.mark.parametrize("op", ["mod", "sin", "cos", "parity"])
def test_parse_modular_name_keeps_a_source_containing_double_underscore(src, op):
    """Round trip: the parsed source is the whole source name, not the text before its first ``__``."""
    name = engineered_name_modular(src, 365.0, op)
    parsed = _parse_modular_name(name)
    assert parsed is not None, name
    assert parsed[0] == op
    assert parsed[1] == src
    assert parsed[2] == 365.0
