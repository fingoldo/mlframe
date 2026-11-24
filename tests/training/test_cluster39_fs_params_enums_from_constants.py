"""Cluster 39 regression: the fs_params MRMR enums for mi_normalization and group_mi_aggregate come from the MRMR constants (audit 2026-11-20, finding 39-F66).

The spec used to hardcode the two tuples, so a new accepted value would leave it stale without failing any test. The tuples now live in the MRMR
constants module, which both the fit validator and the spec read.
"""

from pathlib import Path

from mlframe.feature_selection.filters.mrmr import _mrmr_param_constants as pc
from mlframe.training.fs_params import _spec

_SPEC_SRC = Path(_spec.__file__).resolve().read_text(encoding="utf-8")


def test_spec_enums_equal_the_constants():
    """The spec's MRMR enums read the shared constants (39-F66)."""
    enums = _spec._mrmr_enums()
    assert enums["mi_normalization"] == pc._VALID_MI_NORMALIZATIONS == ("none", "su")
    assert enums["group_mi_aggregate"] == pc._VALID_GROUP_MI_AGGREGATES == ("size", "equal")


def test_spec_source_has_no_hardcoded_tuples():
    """The two keys are wired to the constants, not to literal tuples in the spec (39-F66)."""
    assert '"mi_normalization": ("none", "su")' not in _SPEC_SRC
    assert '"group_mi_aggregate": ("size", "equal")' not in _SPEC_SRC
    assert "c._VALID_MI_NORMALIZATIONS" in _SPEC_SRC
    assert "c._VALID_GROUP_MI_AGGREGATES" in _SPEC_SRC


def test_fit_validator_uses_the_constant():
    """The mi_normalization check in the fit stages reads the shared constant (39-F66)."""
    fit_src = Path(__import__("mlframe.feature_selection.filters.mrmr._mrmr_class_fit_stages", fromlist=["x"]).__file__).read_text(encoding="utf-8")
    assert "_VALID_MI_NORMALIZATIONS" in fit_src
