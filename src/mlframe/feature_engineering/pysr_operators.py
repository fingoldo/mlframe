"""PySR operator presets for symbolic-regression feature engineering.

Three presets curated for tabular FE workloads under
``train_mlframe_models_suite`` -> ``PreprocessingExtensionsConfig.pysr_enabled``:

- ``minimal`` -- legacy-compatible (``[log, inv]``), with ``log`` upgraded to
  ``safe_log`` so the predict-time NaN leak (GA picks ``log(x)`` that turns
  negative on val/test) is fixed.
- ``standard`` -- new default. Adds the operators most often productive on
  numeric tabular targets: ratio (``-, /``), bounding (``max, min``),
  saturating nonlinearity (``tanh``), polynomial (``square``), signed-magnitude
  (``sign``), exponential (``exp``) and safe ``sqrt``. ``complexity_of_operators``
  + ``nested_constraints`` block the GA from wasting budget on runaway
  ``exp(exp(x))`` / ``log(log(x))`` trees.
- ``physics`` -- trig + power identities for oscillatory / wave / ODE-like
  signals (geo / well-log / time-series with seasonality).

Custom operators follow PySR's "return NaN, never throw" rule
([operators.md](https://github.com/MilesCranmer/PySR/blob/master/docs/src/operators.md))
so ``turbo=True`` SIMD eval stays safe. The Julia source strings live in
``OPERATOR_JULIA_SIGNATURES``; ``extra_sympy_mappings`` rebinds them at
predict-time for materialised feature columns.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

# Julia source strings for custom unary operators. Each MUST be total on the
# Float range (return NaN on bad input, never throw) so ``turbo=True`` SIMD
# codegen stays valid. ``T`` is generic so f32 / f64 paths both compile.
OPERATOR_JULIA_SIGNATURES: Dict[str, str] = {
    "safe_log": "safe_log(x::T) where {T} = x > zero(T) ? log(x) : T(NaN)",
    "safe_sqrt": "safe_sqrt(x::T) where {T} = x >= zero(T) ? sqrt(x) : sqrt(-x)",
    "inv": "inv(x::T) where {T} = T(1) / x",
    "gauss": "gauss(x::T) where {T} = exp(-x*x)",
    "softplus": "softplus(x::T) where {T} = log(one(T) + exp(x))",
    "harmonic_mean": "harmonic_mean(x::T, y::T) where {T} = (x + y) > zero(T) ? T(2)*x*y / (x+y) : T(NaN)",
    "xlogy": "xlogy(x::T, y::T) where {T} = y > zero(T) ? x*log(y) : T(NaN)",
}


def _make_extra_sympy_mappings() -> Dict[str, Callable[..., Any]]:
    """Predict-time sympy mappings for the custom operators.

    sympy is imported lazily so importing this module stays free of the
    sympy dep when callers only need the Julia signatures (e.g. preset
    string lookup at config-validation time).
    """
    import sympy as sp

    # Predict-time mappings MUST match the Julia train-time semantics in OPERATOR_JULIA_SIGNATURES bit-for-bit. The legacy ``sp.log(sp.Abs(x) + 1e-9)`` form was chosen for sympy printability but it CHANGES the function PySR fit at train time (Julia returns NaN for x<=0; sympy returned a finite value), so predictions on negative / zero inputs see a value the model never saw during training. Use Piecewise to faithfully replicate the Julia branches.
    return {
        "safe_log": lambda x: sp.Piecewise((sp.log(x), x > 0), (sp.nan, True)),
        "safe_sqrt": lambda x: sp.Piecewise((sp.sqrt(x), x >= 0), (sp.sqrt(-x), True)),
        "inv": lambda x: 1 / x,
        "gauss": lambda x: sp.exp(-(x ** 2)),
        "softplus": lambda x: sp.log(1 + sp.exp(x)),
        "harmonic_mean": lambda x, y: sp.Piecewise(
            (2 * x * y / (x + y), (x + y) > 0), (sp.nan, True)
        ),
        "xlogy": lambda x, y: sp.Piecewise((x * sp.log(y), y > 0), (sp.nan, True)),
    }


def _operators_for_preset(preset: str) -> Tuple[List[str], List[str]]:
    """Return ``(binary_operators, unary_operators)`` for the given preset.

    Unary operators that are PySR built-ins (``square``, ``sign``, ``tanh``,
    ``exp``, ``sin``, ``cos``, ``tan``, ``cube``) are passed by string name;
    custom ones (``safe_log``, ``safe_sqrt``, ``inv``) are passed by the full
    Julia signature so PySR registers them in ``Main`` scope.
    """
    if preset == "minimal":
        binary = ["+", "*"]
        unary = [OPERATOR_JULIA_SIGNATURES["safe_log"], OPERATOR_JULIA_SIGNATURES["inv"]]
    elif preset == "standard":
        binary = ["+", "-", "*", "/", "max", "min"]
        unary = [
            OPERATOR_JULIA_SIGNATURES["safe_log"],
            OPERATOR_JULIA_SIGNATURES["safe_sqrt"],
            "sign",
            "square",
            "tanh",
            "exp",
            OPERATOR_JULIA_SIGNATURES["inv"],
        ]
    elif preset == "physics":
        binary = ["+", "-", "*", "/", "^"]
        unary = [
            OPERATOR_JULIA_SIGNATURES["safe_log"],
            OPERATOR_JULIA_SIGNATURES["safe_sqrt"],
            "sin",
            "cos",
            "tan",
            "exp",
            "square",
            "cube",
            OPERATOR_JULIA_SIGNATURES["inv"],
        ]
    else:
        raise ValueError(
            f"Unknown pysr_operator_preset {preset!r}; expected one of "
            f"{{'minimal', 'standard', 'physics'}}"
        )
    return binary, unary


def _complexity_for_preset(preset: str) -> Dict[str, int]:
    """Per-operator complexity weights. Cheaper ops = 1; expensive (`exp`,
    nested `log/sqrt`) = 2-3 so the GA explores simpler structure first.
    """
    if preset == "minimal":
        return {"safe_log": 2, "inv": 2}
    if preset == "standard":
        return {
            "safe_log": 2,
            "safe_sqrt": 2,
            "exp": 3,
            "tanh": 2,
            "square": 1,
            "sign": 1,
            "inv": 2,
        }
    if preset == "physics":
        return {
            "sin": 2,
            "cos": 2,
            "tan": 3,
            "exp": 3,
            "safe_log": 2,
            "safe_sqrt": 2,
            "cube": 2,
            "square": 1,
            "inv": 2,
        }
    raise ValueError(f"Unknown pysr_operator_preset {preset!r}")


def _nested_constraints_for_preset(preset: str) -> Dict[str, Dict[str, int]]:
    """Block runaway sub-tree patterns (``log(log(x))``, ``exp(exp(x))``,
    ``sin(cos(x))``) by constraining how operators can be nested inside each
    other. 0 = forbidden; default unconstrained.
    """
    if preset == "minimal":
        return {"safe_log": {"safe_log": 0}}
    if preset == "standard":
        return {"exp": {"exp": 0}, "safe_log": {"safe_log": 0}}
    if preset == "physics":
        return {
            "sin": {"sin": 0, "cos": 0},
            "cos": {"sin": 0, "cos": 0},
            "exp": {"exp": 0},
        }
    raise ValueError(f"Unknown pysr_operator_preset {preset!r}")


VALID_PRESETS = ("minimal", "standard", "physics")


def get_preset_kwargs(preset: str) -> Dict[str, Any]:
    """Bundle everything PySR needs for a named preset.

    Returns a dict suitable for splatting into ``PySRRegressor(**preset_kwargs,
    ...)``: ``binary_operators``, ``unary_operators``,
    ``complexity_of_operators``, ``nested_constraints``,
    ``extra_sympy_mappings``.
    """
    binary, unary = _operators_for_preset(preset)
    return {
        "binary_operators": binary,
        "unary_operators": unary,
        "complexity_of_operators": _complexity_for_preset(preset),
        "nested_constraints": _nested_constraints_for_preset(preset),
        "extra_sympy_mappings": _make_extra_sympy_mappings(),
    }


__all__ = [
    "OPERATOR_JULIA_SIGNATURES",
    "VALID_PRESETS",
    "get_preset_kwargs",
]
