"""Every transform in the live registry must have a dedicated provenance formula and description."""

from __future__ import annotations

from mlframe.training.composite.provenance_formulas import (
    _TRANSFORM_DESCRIPTIONS,
    _TRANSFORM_FORMULA_BUILDERS,
    _format_transform_formulas,
    _registered_transform_names,
)


def test_registry_names_are_discovered():
    """The helper sees the live registry (non-vacuity for the coverage checks below)."""
    names = _registered_transform_names()
    assert len(names) >= 10 and "linear_residual" in names


def test_every_registered_transform_has_a_formula_builder():
    """A registered transform without a builder would render the opaque generic stub in provenance reports."""
    missing = sorted(_registered_transform_names() - set(_TRANSFORM_FORMULA_BUILDERS))
    assert not missing, f"registered transforms without a provenance formula builder: {missing}"


def test_every_registered_transform_has_a_description():
    """Descriptions feed the audit report header; an empty one falls back to the bare transform name."""
    missing = sorted(_registered_transform_names() - set(_TRANSFORM_DESCRIPTIONS))
    assert not missing, f"registered transforms without a description: {missing}"


def test_unknown_transform_falls_back_to_the_generic_stub():
    """Negative control: a name outside the registry gets the generic stub, so the dedicated builders are distinguishable."""
    fwd, inv, desc = _format_transform_formulas("not_a_transform", "x", "y", {})
    assert "forward(" in fwd and "inverse(" in inv and desc == "transform 'not_a_transform'"
    fwd_known, _, _ = _format_transform_formulas("linear_residual", "x", "y", {"alpha": 1.0, "beta": 0.0})
    assert "forward(" not in fwd_known
