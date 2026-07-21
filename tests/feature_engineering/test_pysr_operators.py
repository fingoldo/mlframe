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
        """Fast subset."""
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
    """Preset returns documented keys."""
    out = get_preset_kwargs(preset)
    missing = _REQUIRED_KEYS - set(out.keys())
    assert not missing, f"preset {preset!r} missing keys: {missing}"


@pytest.mark.parametrize("preset", _PRESETS_FAST)
def test_preset_binary_and_unary_lists_non_empty(preset):
    """Preset binary and unary lists non empty."""
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
    """Preset unary operators are strings."""
    out = get_preset_kwargs(preset)
    for op in out["unary_operators"]:
        assert isinstance(op, str), f"unary op {op!r} not a string in preset {preset!r}"


@pytest.mark.parametrize("preset", _PRESETS_FAST)
def test_preset_binary_operators_are_strings(preset):
    """Preset binary operators are strings."""
    out = get_preset_kwargs(preset)
    for op in out["binary_operators"]:
        assert isinstance(op, str), f"binary op {op!r} not a string in preset {preset!r}"


@pytest.mark.parametrize("preset", _PRESETS_FAST)
def test_preset_complexity_dict_has_positive_int_weights(preset):
    """Preset complexity dict has positive int weights."""
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
    """Preset nested constraints is nested dict."""
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
    """Preset extra sympy mappings callables."""
    out = get_preset_kwargs(preset)
    mappings = out["extra_sympy_mappings"]
    assert isinstance(mappings, dict)
    # All seven custom operator names must be mapped (predict-time symbolic
    # eval needs them all regardless of preset).
    for name in OPERATOR_JULIA_SIGNATURES:
        assert name in mappings, f"sympy mapping missing for {name!r}"
        assert callable(mappings[name])


def test_unknown_preset_raises_value_error():
    """Unknown preset raises value error."""
    with pytest.raises(ValueError, match="Unknown pysr_operator_preset"):
        get_preset_kwargs("not_a_preset")


def test_unknown_preset_empty_string_raises():
    """Unknown preset empty string raises."""
    with pytest.raises(ValueError, match="Unknown pysr_operator_preset"):
        get_preset_kwargs("")


def test_minimal_preset_is_subset_of_standard_binary():
    # minimal -> standard relationship: minimal binary ops {+, *} must be a
    # strict subset of standard's binary ops. The presets are deliberately
    # layered; a future edit that drops "+" from standard would silently
    # break the layering.
    """Minimal preset is subset of standard binary."""
    minimal = get_preset_kwargs("minimal")
    standard = get_preset_kwargs("standard")
    assert set(minimal["binary_operators"]).issubset(set(standard["binary_operators"]))


def test_physics_preset_includes_trig_operators():
    """Physics preset includes trig operators."""
    out = get_preset_kwargs("physics")
    unary_names = out["unary_operators"]
    # Trig ops live in physics by docstring contract.
    for trig in ("sin", "cos", "tan"):
        assert trig in unary_names, f"physics preset must include {trig!r}"


def test_standard_preset_includes_safe_log_signature():
    # safe_log MUST be passed as the full Julia signature (not just "safe_log")
    # so PySR registers it correctly. Verify the signature string actually
    # ends up in the unary list.
    """Standard preset includes safe log signature."""
    out = get_preset_kwargs("standard")
    assert OPERATOR_JULIA_SIGNATURES["safe_log"] in out["unary_operators"]


def test_safe_log_sympy_mapping_returns_nan_on_nonpositive_input():
    # The 2026-05 PySR Piecewise fix replaced the legacy sp.log(|x|+eps)
    # form with strict NaN-on-x<=0. Pin the new semantic.
    """Safe log sympy mapping returns nan on nonpositive input."""
    out = get_preset_kwargs("standard")
    safe_log = out["extra_sympy_mappings"]["safe_log"]
    expr = safe_log(sp.Symbol("x"))
    # Substitute x=-1 and evaluate -> nan; x=2 -> log(2).
    nan_result = expr.subs(sp.Symbol("x"), -1.0)
    pos_result = expr.subs(sp.Symbol("x"), 2.0)
    assert nan_result == sp.nan
    assert float(pos_result) == pytest.approx(float(sp.log(2)))


def test_standard_preset_wires_previously_unused_custom_operators():
    """Regression for audits/full_audit_2026-07-21/fe_top_b.md F13: gauss, softplus,
    harmonic_mean, and xlogy were defined + always built into extra_sympy_mappings but
    never referenced by any preset. The standard (default) preset must now include all
    four, and their complexity weights must be present alongside them."""
    out = get_preset_kwargs("standard")
    unary_names = out["unary_operators"]
    binary_names = out["binary_operators"]
    weights = out["complexity_of_operators"]

    for unary_op in ("gauss", "softplus"):
        sig = OPERATOR_JULIA_SIGNATURES[unary_op]
        assert sig in unary_names, f"{unary_op!r} Julia signature missing from standard unary_operators"
        assert unary_op in weights, f"{unary_op!r} missing a complexity_of_operators weight"

    for binary_op in ("harmonic_mean", "xlogy"):
        sig = OPERATOR_JULIA_SIGNATURES[binary_op]
        assert sig in binary_names, f"{binary_op!r} Julia signature missing from standard binary_operators"
        assert binary_op in weights, f"{binary_op!r} missing a complexity_of_operators weight"


def test_gauss_sympy_mapping_matches_julia_semantics():
    """gauss(x) = exp(-x^2); pin the sympy predict-time mapping matches the Julia train-time one."""
    out = get_preset_kwargs("standard")
    gauss = out["extra_sympy_mappings"]["gauss"]
    expr = gauss(sp.Symbol("x"))
    assert float(expr.subs(sp.Symbol("x"), 0.0)) == pytest.approx(1.0)
    assert float(expr.subs(sp.Symbol("x"), 2.0)) == pytest.approx(float(sp.exp(-4)))


def test_harmonic_mean_sympy_mapping_returns_nan_on_nonpositive_sum():
    """harmonic_mean(x, y) is NaN when x+y <= 0 (matches the Julia branch, never throws)."""
    out = get_preset_kwargs("standard")
    harmonic_mean = out["extra_sympy_mappings"]["harmonic_mean"]
    expr = harmonic_mean(sp.Symbol("x"), sp.Symbol("y"))
    nan_result = expr.subs({sp.Symbol("x"): -3.0, sp.Symbol("y"): 3.0})  # x+y == 0
    pos_result = expr.subs({sp.Symbol("x"): 2.0, sp.Symbol("y"): 4.0})
    assert nan_result == sp.nan
    assert float(pos_result) == pytest.approx(2 * 2.0 * 4.0 / (2.0 + 4.0))


def test_xlogy_sympy_mapping_returns_nan_on_nonpositive_y():
    """xlogy(x, y) is NaN when y <= 0 (matches the Julia branch, never throws)."""
    out = get_preset_kwargs("standard")
    xlogy = out["extra_sympy_mappings"]["xlogy"]
    expr = xlogy(sp.Symbol("x"), sp.Symbol("y"))
    nan_result = expr.subs({sp.Symbol("x"): 3.0, sp.Symbol("y"): -1.0})
    pos_result = expr.subs({sp.Symbol("x"): 3.0, sp.Symbol("y"): 2.0})
    assert nan_result == sp.nan
    assert float(pos_result) == pytest.approx(3.0 * float(sp.log(2)))


def test_safe_sqrt_sympy_mapping_handles_negative_inputs():
    """Safe sqrt sympy mapping handles negative inputs."""
    out = get_preset_kwargs("standard")
    safe_sqrt = out["extra_sympy_mappings"]["safe_sqrt"]
    expr = safe_sqrt(sp.Symbol("x"))
    # Negative input -> sqrt(-x), so x=-4 -> sqrt(4)=2 (matches Julia branch).
    neg_result = expr.subs(sp.Symbol("x"), -4.0)
    pos_result = expr.subs(sp.Symbol("x"), 9.0)
    assert float(neg_result) == pytest.approx(2.0)
    assert float(pos_result) == pytest.approx(3.0)
