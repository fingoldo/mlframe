"""GPU-resident replay for ``unary_binary`` engineered recipes (sibling of
``_recipe_unary_binary``).

The CPU ``_apply_unary_binary`` materialises the final engineered column by
applying ``binary(unary_a(X[a]), unary_b(X[b]))`` ELEMENTWISE on the FULL-n host
columns. On the F2 STRICT 300k-1M path this is the dominant FE-replay cost
(~3.4s), and unlike the small sequential FE stages it is embarrassingly parallel
(one large operand, not launch-bound), so a cupy port wins on this card.

``apply_unary_binary_gpu`` uploads the two source columns ONCE, applies the
operator chain with cupy elementwise ops on device, and returns the materialised
column to host (one legitimate bulk D2H -- the engineered output). Each operator
is mapped to its EXACT cupy equivalent so the values match the numpy path within
the f32/f64 discipline of the resident path (selection-equivalent on F2).

GATING + FALLBACK: the caller takes this path only when
``fe_gpu_strict_resident_enabled()`` is True. Operands that are nested-engineered
parents, the ``prewarp`` / ``gate_med`` closed-form pseudo-unaries, or any unary
not in the closed-form cupy map are NOT handled here -- the function returns
``None`` so the caller falls back to the numpy ``_apply_unary_binary`` (which
also recurses for nested parents). Any cupy failure raises and the caller's
try/except logs debug + falls back. NEVER calls
``cp.get_default_memory_pool().free_all_blocks()``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ._recipe_core import EngineeredRecipe
from ._recipe_extract import _extract_column


# Closed-form unary names whose cupy form is byte-faithful to the numpy registry
# (``create_unary_transformations``). EXCLUDES the pseudo-unaries (prewarp /
# gate_med, handled closed-form on CPU from fit-time state) and the scipy.special
# / sinc / arc* / hyperbolic-without-cupy families that the resident batch path
# also leaves on host. ``log`` is handled separately (data-dependent shift).
_GPU_UNARY = frozenset({
    "identity", "neg", "abs", "sign", "rint",
    "sqr", "qubed", "reciproc", "invsquared", "invqubed",
    "cbrt", "sqrt", "invcbrt", "invsqrt",
    "exp", "sin", "cos", "tan", "sinh", "cosh", "tanh",
    "log",
})

# Binary names with an exact cupy elementwise equivalent (mirrors
# ``create_binary_transformations``). ``div`` mirrors ``_safe_div`` exactly.
_GPU_BINARY = frozenset({
    "mul", "add", "sub", "div", "max", "min",
    "abs_diff", "hypot", "signed", "ratio_abs",
    "pow", "logaddexp", "heaviside", "greater", "less", "equal",
})

_PREWARP = "prewarp"
_GATE_MED = "gate_med"
_PSEUDO = (_PREWARP, _GATE_MED)


def _vram_f32() -> bool:
    """Match the rest of the resident FE path's dtype discipline (f32 under
    ``MLFRAME_FE_VRAM_F32``, else f64)."""
    try:
        from .._fe_gpu_batch._devices import fe_gpu_f32_enabled
        return bool(fe_gpu_f32_enabled())
    except Exception:
        return False


def _gpu_unary(cp, name: str, x, recipe: EngineeredRecipe, side: str):
    """Apply one unary operator on device, byte-faithful to the numpy registry.

    Returns the cupy array. ``log`` reproduces the registry ``smart_log`` shift
    (data-dependent ``1e-5 - nanmin``) OR the FROZEN ``log_shift_<side>`` anchor
    when the recipe stored one (BUG2 fit-time freeze), matching the CPU replay."""
    if name == "identity":
        return x
    if name == "neg":
        return -x
    if name == "abs":
        return cp.abs(x)
    if name == "sign":
        return cp.sign(x)
    if name == "rint":
        return cp.rint(x)
    if name == "sqr":
        return cp.power(x, 2)
    if name == "qubed":
        return cp.power(x, 3)
    if name == "reciproc":
        return cp.power(x, -1)
    if name == "invsquared":
        return cp.power(x, -2)
    if name == "invqubed":
        return cp.power(x, -3)
    if name == "cbrt":
        return cp.cbrt(x)
    if name == "sqrt":
        return cp.sqrt(cp.abs(x))
    if name == "invcbrt":
        return cp.power(x, -1.0 / 3.0)
    if name == "invsqrt":
        return cp.power(x, -1.0 / 2.0)
    if name == "exp":
        return cp.exp(x)
    if name == "sin":
        return cp.sin(x)
    if name == "cos":
        return cp.cos(x)
    if name == "tan":
        return cp.tan(x)
    if name == "sinh":
        return cp.sinh(x)
    if name == "cosh":
        return cp.cosh(x)
    if name == "tanh":
        return cp.tanh(x)
    if name == "log":
        _skey = f"log_shift_{side}"
        if _skey in recipe.extra:
            # FROZEN fit-time anchor (BUG2 FIX): replay log from that exact shift,
            # mirroring ``_apply_unary_binary``'s frozen-anchor branch.
            _shift = float(recipe.extra[_skey])
            return cp.log(x + _shift) if _shift != 0.0 else cp.log(x)
        # ``smart_log``: shift non-positive inputs by ``(1e-5 - nanmin(x))``. Branchless + resident: keep the
        # shift as a device 0-dim scalar (cp.where picks 0 when min>0) instead of a float(x_min) sync -- x+0.0
        # is identity, so this is bit-identical to the min>0 -> log(x) branch.
        x_min = cp.nanmin(x)
        shift = cp.where(x_min > 0.0, cp.float64(0.0), 1e-5 - x_min)
        return cp.log(x + shift)
    raise ValueError(f"GPU unary missing for {name!r}")


def _gpu_binary(cp, name: str, a, b):
    """Apply one binary operator on device, byte-faithful to the numpy registry."""
    if name == "mul":
        return a * b
    if name == "add":
        return a + b
    if name == "sub":
        return a - b
    if name == "div":
        # ``_safe_div``: EXACT for every nonzero denominator; eps only substitutes
        # for an exact-zero denominator (matches numpy ``where(y==0, eps, y)``).
        safe_b = cp.where(b == 0.0, cp.asarray(1e-9, dtype=b.dtype), b)
        return a / safe_b
    if name == "max":
        return cp.maximum(a, b)
    if name == "min":
        return cp.minimum(a, b)
    if name == "abs_diff":
        return cp.abs(a - b)
    if name == "hypot":
        return cp.hypot(a, b)
    if name == "signed":
        return cp.sign(a) * cp.abs(b)
    if name == "ratio_abs":
        return a / (cp.abs(b) + 1.0)
    if name == "pow":
        return cp.power(a, b)
    if name == "logaddexp":
        return cp.logaddexp(a, b)
    if name == "heaviside":
        return cp.heaviside(a, b)
    if name == "greater":
        return cp.greater(a, b).astype(a.dtype)
    if name == "less":
        return cp.less(a, b).astype(a.dtype)
    if name == "equal":
        return cp.equal(a, b).astype(a.dtype)
    raise ValueError(f"GPU binary missing for {name!r}")


def _materialise_recipe_gpu(recipe: EngineeredRecipe, X: Any, cp, dt):
    """RECURSIVELY materialise a ``unary_binary`` recipe ON device, returning a RESIDENT cupy ``(n,)`` array, or
    ``None`` when the recipe (or any nested parent) is not GPU-eligible.

    This is the device twin of the host ``_apply_unary_binary`` recursion (``_nested_continuous`` -> replay the
    parent WITHOUT quantization -> continuous operand). A NESTED-engineered operand is materialised device-born
    by recursing here instead of bailing to CPU -- so a compound like ``add(div(sqr(a),b), mul(log(c),d))`` is
    built entirely on device from the resident raw base columns, its intermediate product/ratio operands never
    crossing H2D. A raw operand rides the content-keyed resident-operand cache (uploaded once, shared). The
    pseudo-unaries (prewarp / gate_med) + unmapped unary/binary names still bail (``None``) so the caller keeps
    the exact CPU replay; a nested parent that bails propagates the ``None`` up. Selection-equivalent: the
    device unary/binary kernels mirror the numpy registry op-for-op (see ``_gpu_unary`` / ``_gpu_binary``)."""
    if len(recipe.src_names) != 2 or len(recipe.unary_names) != 2:
        return None
    u_a, u_b = recipe.unary_names
    if u_a in _PSEUDO or u_b in _PSEUDO:
        return None
    if u_a not in _GPU_UNARY or u_b not in _GPU_UNARY:
        return None
    if recipe.binary_name not in _GPU_BINARY:
        return None

    from .._fe_resident_operands import resident_operand

    def _operand_gpu(side_name, nested_parent):
        # Nested-engineered operand: replay the parent WITHOUT quantization ON device (recurse). Raw operand:
        # ride the resident cache (uploaded once per fit, shared across every recipe reusing the column).
        if nested_parent is not None:
            _parent = nested_parent
            if getattr(_parent, "quantization", None) is not None:
                import dataclasses as _dc
                _parent = _dc.replace(_parent, quantization=None)
            return _materialise_recipe_gpu(_parent, X, cp, dt)
        vals = _extract_column(X, side_name)
        try:
            return resident_operand(vals, ("recipe_src", side_name), dtype=dt)
        except Exception:
            return cp.asarray(np.ascontiguousarray(vals), dtype=dt)

    name_a, name_b = recipe.src_names
    a_gpu = _operand_gpu(name_a, recipe.extra.get("nested_parent_a"))
    if a_gpu is None:
        return None
    b_gpu = _operand_gpu(name_b, recipe.extra.get("nested_parent_b"))
    if b_gpu is None:
        return None

    ta = _gpu_unary(cp, u_a, a_gpu, recipe, "a")
    tb = _gpu_unary(cp, u_b, b_gpu, recipe, "b")
    out_gpu = _gpu_binary(cp, recipe.binary_name, ta, tb)
    # Match fit-time NaN/Inf scrubbing in check_prospective_fe_pairs + the CPU replay's nan_to_num.
    # cp.where(cp.isfinite(...)) not cp.nan_to_num(nan=0,posinf=0,neginf=0): cupy's nan_to_num runs
    # cupy.isnan() on each scalar fill arg internally, a blocking D2H sync on every call -- the where form is
    # elementwise and identical (nan / +-inf -> 0.0, finite unchanged).
    return cp.where(cp.isfinite(out_gpu), out_gpu, cp.asarray(0.0, dtype=out_gpu.dtype))


def apply_unary_binary_gpu(recipe: EngineeredRecipe, X: Any) -> Optional[np.ndarray]:
    """GPU-resident replay of a ``unary_binary`` recipe -> host 1-D ``np.ndarray``.

    Returns ``None`` when the recipe is NOT GPU-eligible (a ``prewarp`` / ``gate_med`` pseudo-unary or an
    unmapped unary/binary name, at any nesting level) so the caller falls back to the numpy
    ``_apply_unary_binary``. NESTED-engineered operands are now materialised device-born (recursion in
    ``_materialise_recipe_gpu``) rather than bailing to CPU. Raises on a cupy runtime failure -- the caller's
    try/except logs debug + falls back."""
    if len(recipe.src_names) != 2 or len(recipe.unary_names) != 2:
        return None  # let the numpy path raise its descriptive ValueError

    import cupy as cp

    dt = cp.float32 if _vram_f32() else cp.float64
    out_gpu = _materialise_recipe_gpu(recipe, X, cp, dt)
    if out_gpu is None:
        return None
    # The ONE legitimate output transfer: the materialised engineered column D2H.
    return cp.asnumpy(out_gpu)
