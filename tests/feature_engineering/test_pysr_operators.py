"""Unit tests for `feature_engineering.pysr_operators.get_preset_kwargs`.

The preset bundles plug straight into `PySRRegressor(**preset_kwargs)`, so a
silent regression in any field (missing key, empty list, wrong type) breaks
the production GA without surfacing a clean error. These tests pin every
documented contract.
"""
from __future__ import annotations

import pytest
import sympy as sp  # extra_sympy_mappings is built with sympy; module-level import is fine.

from mlframe.feature_engineering.pysr_operators import (
    OPERATOR_JULIA_SIGNATURES,
    VALID_PRESETS,
    get_preset_kwargs,
)

try:
    from tests.conftest import fast_subset
except ImportError:  # pragma: no cover
    def fast_subset(values, **_):
        return list(values)


_REQUIRED_KEYS = {
    "binary_operators",
    "unary_operators",
    "complexity_of_operators",
    "nested_constraints",
    "extra_sympy_mappings",
}

# In --fast mode the contract suite collapses to a single representative preset
# (``standard`` if present, else the first registered). Each individual contract
# is still exercised end-to-end -- only the preset axis is collapsed.
_PRESETS_FAST = fast_subset(VALID_PRESETS, representative="standard")


@pytest.mark.parametrize("preset", _PRESETS_FAST)
def test_preset_returns_documented_keys(preset):
    out = get_preset_kwargs(preset)
    missing = _REQUIRED_KEYS - set(out.keys())
    assert not missing, f"preset {preset!r} missing keys: {missing}"


@pytest.mark.parametrize("preset", _PRESETS_FAST)
def test_preset_binary_and_unary_lists_non_empty(preset):
    out = get_preset_kwargs(preset)
    assert isinstance(out["binary_operators"], list)
    assert isinstance(out["unary_operators"], list)
    assert len(out["binary_operators"]) >= 2
    assert len(out["unary_operators"]) >= 2


@pytest.mark.parametrize("preset", _PRESETS_FAST)
def test_preset_unary_operators_are_strings(preset):
    # PySR contract: each unary entry is either the operator name (for
    # built-ins like "tanh", "square", "sin") or the full Julia signature
    # for custom operators. Both must be `str`.
    out = get_preset_kwargs(preset)
    for op in out["unary_operators"]:
        assert isinstance(op, str), f"unary op {op!r} not a string in preset {preset!r}"


@pytest.mark.parametrize("preset", _PRESETS_FAST)
def test_preset_binary_operators_are_strings(preset):
    out = get_preset_kwargs(preset)
    for op in out["binary_operators"]:
        assert isinstance(op, str), f"binary op {op!r} not a string in preset {preset!r}"


@pytest.mark.parametrize("preset", _PRESETS_FAST)
def test_preset_complexity_dict_has_positive_int_weights(preset):
    out = get_preset_kwargs(preset)
    weights = out["complexity_of_operators"]
    assert isinstance(weights, dict)
    assert len(weights) >= 1
    for k, v in weights.items():
        assert isinstance(k, str)
        assert isinstance(v, int)
        assert v >= 1


@pytest.mark.parametrize("preset", _PRESETS_FAST)
def test_preset_nested_constraints_is_nested_dict(preset):
    out = get_preset_kwargs(preset)
    nc = out["nested_constraints"]
    assert isinstance(nc, dict)
    # Each value is itself a {child_op: max_depth} dict; 0 = forbidden.
    for parent, children in nc.items():
        assert isinstance(parent, str)
        assert isinstance(children, dict)
        for child, depth in children.items():
            assert isinstance(child, str)
            assert isinstance(depth, int)
            assert depth >= 0


@pytest.mark.parametrize("preset", _PRESETS_FAST)
def test_preset_extra_sympy_mappings_callables(preset):
    out = get_preset_kwargs(preset)
    mappings = out["extra_sympy_mappings"]
    assert isinstance(mappings, dict)
    # All seven custom operator names must be mapped (predict-time symbolic
    # eval needs them all regardless of preset).
    for name in OPERATOR_JULIA_SIGNATURES:
        assert name in mappings, f"sympy mapping missing for {name!r}"
        assert callable(mappings[name])


def test_unknown_preset_raises_value_error():
    with pytest.raises(ValueError, match="Unknown pysr_operator_preset"):
        get_preset_kwargs("not_a_preset")


def test_unknown_preset_empty_string_raises():
    with pytest.raises(ValueError, match="Unknown pysr_operator_preset"):
        get_preset_kwargs("")


def test_minimal_preset_is_subset_of_standard_binary():
    # minimal -> standard relationship: minimal binary ops {+, *} must be a
    # strict subset of standard's binary ops. The presets are deliberately
    # layered; a future edit that drops "+" from standard would silently
    # break the layering.
    minimal = get_preset_kwargs("minimal")
    standard = get_preset_kwargs("standard")
    assert set(minimal["binary_operators"]).issubset(set(standard["binary_operators"]))


def test_physics_preset_includes_trig_operators():
    out = get_preset_kwargs("physics")
    unary_names = out["unary_operators"]
    # Trig ops live in physics by docstring contract.
    for trig in ("sin", "cos", "tan"):
        assert trig in unary_names, f"physics preset must include {trig!r}"


def test_standard_preset_includes_safe_log_signature():
    # safe_log MUST be passed as the full Julia signature (not just "safe_log")
    # so PySR registers it correctly. Verify the signature string actually
    # ends up in the unary list.
    out = get_preset_kwargs("standard")
    assert OPERATOR_JULIA_SIGNATURES["safe_log"] in out["unary_operators"]


def test_safe_log_sympy_mapping_returns_nan_on_nonpositive_input():
    # The 2026-05 PySR Piecewise fix replaced the legacy sp.log(|x|+eps)
    # form with strict NaN-on-x<=0. Pin the new semantic.
    out = get_preset_kwargs("standard")
    safe_log = out["extra_sympy_mappings"]["safe_log"]
    expr = safe_log(sp.Symbol("x"))
    # Substitute x=-1 and evaluate -> nan; x=2 -> log(2).
    nan_result = expr.subs(sp.Symbol("x"), -1.0)
    pos_result = expr.subs(sp.Symbol("x"), 2.0)
    assert nan_result == sp.nan
    assert float(pos_result) == pytest.approx(float(sp.log(2)))


def test_safe_sqrt_sympy_mapping_handles_negative_inputs():
    out = get_preset_kwargs("standard")
    safe_sqrt = out["extra_sympy_mappings"]["safe_sqrt"]
    expr = safe_sqrt(sp.Symbol("x"))
    # Negative input -> sqrt(-x), so x=-4 -> sqrt(4)=2 (matches Julia branch).
    neg_result = expr.subs(sp.Symbol("x"), -4.0)
    pos_result = expr.subs(sp.Symbol("x"), 9.0)
    assert float(neg_result) == pytest.approx(2.0)
    assert float(pos_result) == pytest.approx(3.0)
