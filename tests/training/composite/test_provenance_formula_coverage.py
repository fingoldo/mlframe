"""S12 regression: ``_format_transform_formulas`` covers EVERY registered
transform with a faithful, fitted-param-interpolating formula.

Audit S12 (composite_audit_2026_06_10): the provenance formula table covered
only 4 of ~36 registered transforms (``diff`` / ``ratio`` / ``logratio`` /
``linear_residual``); the other ~32 rendered the opaque generic stub
``T = forward(target, base) [name]`` / ``y_hat = inverse(...) [name]``. That
contradicts :class:`CompositeProvenance`'s documented promise to let a
downstream consumer "reproduce the inverse at serving time without consulting
source code".

The fix makes the renderer registry-driven: a dedicated builder per
registered transform, fitted-param interpolation, full description coverage.
These tests FAIL on the pre-fix logic (28 registered transforms hit the
generic stub) and pin the new contract so a freshly-registered transform that
forgets a formula entry trips this suite instead of silently regressing to the
stub.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.provenance import (
    _TRANSFORM_DESCRIPTIONS,
    _TRANSFORM_FORMULA_BUILDERS,
    _format_transform_formulas,
)
from mlframe.training.composite.transforms import TRANSFORMS_REGISTRY


pytestmark = pytest.mark.sklearn_matrix


_RNG = np.random.default_rng(0)
_N = 200
_BASE = np.linspace(1.0, 10.0, _N)
_Y = 0.5 * _BASE + 1.0 + _RNG.standard_normal(_N) * 0.1
_GROUPS = (np.arange(_N) // 50).astype(np.int64)


def _fit_params(name: str) -> dict:
    """Fit a transform on synthetic data and return its fitted params."""
    t = TRANSFORMS_REGISTRY[name]
    # Groups checked FIRST: a transform may be requires_groups=True AND
    # requires_base=False (target_encoding_residual), needing groups + base=None.
    if t.requires_groups:
        base = _BASE if t.requires_base else None
        dom = t.domain_check(_Y, base)
        base_dom = _BASE[dom] if t.requires_base else None
        return t.fit(_Y[dom], base_dom, groups=_GROUPS[: int(dom.sum())])
    if not t.requires_base:
        dom = t.domain_check(_Y, None)
        return t.fit(_Y[dom], None)
    dom = t.domain_check(_Y, _BASE)
    return t.fit(_Y[dom], _BASE[dom])


_GENERIC_FORWARD_MARK = "= forward("
_GENERIC_INVERSE_MARK = "= inverse("


# ----------------------------------------------------------------------
# Coverage: every registered transform has a builder + a description.
# (Pure structural pin -- FAILS pre-fix because only 4 builders existed.)
# ----------------------------------------------------------------------


def test_every_registered_transform_has_a_formula_builder() -> None:
    registered = set(TRANSFORMS_REGISTRY)
    missing = registered - set(_TRANSFORM_FORMULA_BUILDERS)
    assert not missing, (
        "registered transforms with no provenance formula builder (would "
        f"render the opaque generic stub): {sorted(missing)}"
    )


def test_every_registered_transform_has_a_description() -> None:
    registered = set(TRANSFORMS_REGISTRY)
    missing = registered - set(_TRANSFORM_DESCRIPTIONS)
    assert not missing, (
        f"registered transforms with no stakeholder description: {sorted(missing)}"
    )


@pytest.mark.parametrize("name", sorted(TRANSFORMS_REGISTRY))
def test_registered_transform_never_renders_generic_stub(name: str) -> None:
    """Behavioural pin: a REGISTERED transform must never fall through to the
    ``forward(...)/inverse(...) [name]`` generic stub. This is the exact line
    that fails on the pre-fix code for the 28 uncovered transforms."""
    params = _fit_params(name)
    forward, inverse, description = _format_transform_formulas(
        transform_name=name,
        base_column="base_col",
        target_col="y",
        fitted_params=params,
    )
    assert _GENERIC_FORWARD_MARK not in forward, (
        f"{name!r} rendered the generic forward stub: {forward!r}"
    )
    assert _GENERIC_INVERSE_MARK not in inverse, (
        f"{name!r} rendered the generic inverse stub: {inverse!r}"
    )
    # A real, non-empty, target-mentioning formula on both legs.
    assert forward.startswith("T") and "y" in forward
    assert "y_hat" in inverse or inverse.startswith("T")  # chains spell sub-stages first
    assert description, f"{name!r} rendered an empty description"


# ----------------------------------------------------------------------
# biz_value: the new formulas are SELF-DESCRIBING -- they interpolate the
# actual fitted parameter values, so a serving consumer can reproduce the
# inverse from the string alone. A regression that drops interpolation (or
# reverts to the stub) fails the "value is present in the string" assertion.
# ----------------------------------------------------------------------


def _g(value) -> str:
    return f"{float(value):.4g}"


def test_biz_linear_residual_family_interpolates_alpha_beta() -> None:
    """The alpha/beta family (the most common production transforms) MUST
    spell the fitted coefficients into the formula -- the whole point of
    provenance over the opaque key. Pre-fix only ``linear_residual`` did;
    ``additive_residual`` / ``polynomial_residual_deg2`` / ``asinh_residual``
    were stubs."""
    # linear_residual: alpha and beta both interpolated.
    fwd, inv, _ = _format_transform_formulas(
        "linear_residual", "x", "y", {"alpha": 0.731, "beta": -2.4},
    )
    assert _g(0.731) in fwd and _g(-2.4) in fwd
    assert _g(0.731) in inv and _g(-2.4) in inv

    # additive_residual: beta interpolated (alpha fixed at 1, not shown).
    fwd, inv, _ = _format_transform_formulas(
        "additive_residual", "x", "y", {"beta": 3.14159},
    )
    assert _g(3.14159) in fwd and _g(3.14159) in inv

    # polynomial_residual_deg2: alpha1, alpha2, beta all interpolated.
    fwd, inv, _ = _format_transform_formulas(
        "polynomial_residual_deg2", "x", "y",
        {"alpha1": 0.5, "alpha2": -0.03, "beta": 1.2},
    )
    for v in (0.5, -0.03, 1.2):
        assert _g(v) in fwd, (v, fwd)
        assert _g(v) in inv, (v, inv)

    # asinh_residual: alpha, beta interpolated + arcsinh structure present.
    fwd, inv, _ = _format_transform_formulas(
        "asinh_residual", "x", "y", {"alpha": 0.8, "beta": 0.1},
    )
    assert "asinh(y)" in fwd and "sinh(" in inv
    assert _g(0.8) in fwd and _g(0.1) in fwd


def test_biz_unary_and_chain_formulas_carry_their_fitted_state() -> None:
    """Unary y-transforms and chains were ALL stubs pre-fix. Pin that the
    fitted lambda / offset surface in the rendered string."""
    # log_y: fitted offset interpolated on both legs.
    fwd, inv, _ = _format_transform_formulas("log_y", "x", "y", {"offset": 4.5})
    assert _g(4.5) in fwd and _g(4.5) in inv
    assert "log(y" in fwd and "exp(T_hat)" in inv

    # yeo_johnson_y: fitted lambda interpolated.
    fwd, inv, _ = _format_transform_formulas(
        "yeo_johnson_y", "x", "y", {"lambda": 1.234},
    )
    assert _g(1.234) in fwd and _g(1.234) in inv

    # chain_linres_cbrt: nested bivariate alpha/beta surfaced from the
    # sub-params dict (pre-fix: opaque stub).
    fwd, inv, _ = _format_transform_formulas(
        "chain_linres_cbrt", "x", "y",
        {"bivariate_params": {"alpha": 0.6, "beta": 0.2}, "unary_params": {}},
    )
    assert _g(0.6) in fwd and _g(0.2) in fwd
    assert "^(1/3)" in fwd and "T_hat^3" in inv


def test_unknown_transform_still_falls_back_to_generic_stub() -> None:
    """The generic stub must remain reachable for genuinely-unknown names
    (forward-compat): the function never raises on an unregistered name."""
    fwd, inv, desc = _format_transform_formulas(
        "totally_made_up_v99", "x", "y", {},
    )
    assert "totally_made_up_v99" in fwd
    assert "totally_made_up_v99" in inv
    assert desc  # non-empty fallback description


def test_original_four_branches_are_bit_identical() -> None:
    """The pre-existing 4 transforms must render character-for-character the
    same strings as before the S12 refactor (no behaviour change for them)."""
    assert _format_transform_formulas("diff", "x", "y", {})[:2] == (
        "T = y - x", "y_hat = T_hat + x",
    )
    assert _format_transform_formulas("ratio", "x", "y", {})[:2] == (
        "T = y / x  (with |x| >= 1e-12)", "y_hat = T_hat * x",
    )
    assert _format_transform_formulas(
        "logratio", "x", "y", {"median_t": 0.1, "mad_eff": 0.05},
    )[:2] == (
        "T = log(y) - log(x)  (requires y, x > 0)",
        "y_hat = x * exp(clip(T_hat, 0.1 +/- 10*0.05))",
    )
    assert _format_transform_formulas(
        "linear_residual", "x", "y", {"alpha": 0.95, "beta": -1.5},
    )[:2] == (
        "T = y - 0.95 * x - (-1.5)",
        "y_hat = T_hat + 0.95 * x + (-1.5)",
    )
