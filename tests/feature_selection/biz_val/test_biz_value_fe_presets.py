"""Preset-sanity guard for MRMR FE unary/binary transformation registries.

Pins the contract established by the 2026-06-01 FE fix:

* EVERY preset {minimal, medium, maximal} has >1 unary AND >1 binary member
  (an identity-only "minimal" silently crippled MRMR pair FE).
* tiers grow monotonically: minimal subset of medium subset of maximal, for
  both unary and binary registries.
* div + sub are present in the binary MINIMAL preset (division was
  previously only reachable as reciproc-then-multiply, and subtraction was
  absent from every tier).
* "rich"/"full" aliases resolve to "maximal"; unknown presets raise ValueError.
* every transform in every preset is callable on a sample array without
  crashing (and njit-compiles where the pipeline expects it, i.e. it does not
  raise when invoked).
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.feature_engineering import (
    create_binary_transformations,
    create_unary_transformations,
    _resolve_preset,
)

PRESETS = ["minimal", "medium", "maximal"]


@pytest.mark.parametrize("preset", PRESETS)
def test_every_preset_has_more_than_one_member(preset):
    """Every preset has more than one member."""
    u = create_unary_transformations(preset)
    b = create_binary_transformations(preset)
    assert len(u) > 1, f"unary[{preset}] has {len(u)} member(s); must be > 1"
    assert len(b) > 1, f"binary[{preset}] has {len(b)} member(s); must be > 1"


def test_unary_tiers_monotonically_grow():
    """Unary tiers monotonically grow."""
    mn = set(create_unary_transformations("minimal"))
    md = set(create_unary_transformations("medium"))
    mx = set(create_unary_transformations("maximal"))
    assert mn <= md, f"unary minimal not subset of medium: {mn - md}"
    assert md <= mx, f"unary medium not subset of maximal: {md - mx}"
    # strictly growing (each tier adds something)
    assert len(mn) < len(md) < len(mx)


def test_binary_tiers_monotonically_grow():
    """Binary tiers monotonically grow."""
    mn = set(create_binary_transformations("minimal"))
    md = set(create_binary_transformations("medium"))
    mx = set(create_binary_transformations("maximal"))
    assert mn <= md, f"binary minimal not subset of medium: {mn - md}"
    assert md <= mx, f"binary medium not subset of maximal: {md - mx}"
    assert len(mn) < len(md) < len(mx)


def test_unary_minimal_has_sqr():
    # ``sqr`` (the x**2 op, renamed from ``squared`` 2026-06-01) is the building
    # block for a**2/b targets and MUST be in the minimal unary preset.
    """Unary minimal has sqr."""
    u = create_unary_transformations("minimal")
    assert "sqr" in u, "unary minimal missing 'sqr'"
    assert "squared" not in u, "unary preset still exposes legacy 'squared' name"


def test_binary_minimal_has_sub_and_div():
    """Binary minimal has sub and div."""
    b = create_binary_transformations("minimal")
    assert "sub" in b, "binary minimal missing 'sub'"
    assert "div" in b, "binary minimal missing 'div'"
    # the classic four are still present
    for name in ("mul", "add", "max", "min"):
        assert name in b, f"binary minimal missing '{name}'"


def test_div_handles_divide_by_zero():
    """Div handles divide by zero."""
    b = create_binary_transformations("minimal")
    div = b["div"]
    num = np.array([1.0, 2.0, -3.0, 0.0], dtype=np.float32)
    den = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    out = np.asarray(div(num, den))
    assert np.all(np.isfinite(out)), f"div produced non-finite on /0: {out}"


def test_preset_aliases_and_unknown():
    """Preset aliases and unknown."""
    assert _resolve_preset("rich") == "maximal"
    assert _resolve_preset("full") == "maximal"
    assert _resolve_preset("MINIMAL") == "minimal"  # case-insensitive
    with pytest.raises(ValueError):
        _resolve_preset("bogus")
    # The factory functions must propagate the alias / raise too.
    assert set(create_unary_transformations("rich")) == set(create_unary_transformations("maximal"))
    with pytest.raises(ValueError):
        create_binary_transformations("nonsense")


@pytest.mark.parametrize("preset", PRESETS)
def test_every_transform_callable_without_crash(preset):
    # Sample arrays in a benign domain (positive for sqrt/log; finite for trig).
    """Every transform callable without crash."""
    x = np.linspace(0.5, 4.0, 64).astype(np.float32)
    yv = np.linspace(0.6, 5.0, 64).astype(np.float32)

    for name, fn in create_unary_transformations(preset).items():
        if "poly_" in name:
            continue  # hermite-coefficient keys, applied via hermval not call
        out = np.asarray(fn(x))
        assert out.shape[0] == x.shape[0], f"unary[{preset}].{name} changed length"

    for name, fn in create_binary_transformations(preset).items():
        out = np.asarray(fn(x, yv))
        assert out.shape[0] == x.shape[0], f"binary[{preset}].{name} changed length"
