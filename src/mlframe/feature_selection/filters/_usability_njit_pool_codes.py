"""Op-code tables of the njit usability pool and the mappers from preset names to codes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Unary op-code table for the ``medium`` preset (registry order is irrelevant - the caller maps each
# preset name to its code and falls back to Python if any name is absent). Every entry below is
# bit-exact (float32) to ``create_unary_transformations(preset='medium')[name]``; see ``_apply_unary``.
_NJIT_UNARY_OP_CODES: dict = {
    "identity": 0,
    "neg": 1,
    "abs": 2,
    "sqr": 3,
    "reciproc": 4,
    "sqrt": 5,
    "log": 6,
    "sin": 7,
    "sign": 8,
    "rint": 9,
    "qubed": 10,
    "invsquared": 11,
    "invqubed": 12,
    "cbrt": 13,
    "invcbrt": 14,
    "invsqrt": 15,
    "exp": 16,
}

# Binary op-code table for the ``minimal`` preset (the pool's default binary set). Bit-exact to
# ``create_binary_transformations(preset='minimal')`` - see ``_apply_binary``; ``div`` mirrors
# ``_safe_div`` (exact x/y for y!=0, 1e-9 floor only on an exact-zero denominator).
_NJIT_USABILITY_BINARY_OP_CODES: dict = {
    "mul": 0,
    "add": 1,
    "sub": 2,
    "div": 3,
    "max": 4,
    "min": 5,
}


def njit_unary_codes_or_none(unary_names: Sequence[str]) -> "np.ndarray | None":
    """Map each unary name (preset iteration order) to its op-code, or None if ANY name is not
    njit-coded (the caller then uses the per-candidate numpy path - correctness first)."""
    codes = []
    for name in unary_names:
        c = _NJIT_UNARY_OP_CODES.get(name)
        if c is None:
            return None
        codes.append(c)
    return np.asarray(codes, dtype=np.int64)


def njit_binary_codes_or_none(binary_names: Sequence[str]) -> "np.ndarray | None":
    """Map each binary name to its op-code, or None if ANY name is not njit-coded."""
    codes = []
    for name in binary_names:
        c = _NJIT_USABILITY_BINARY_OP_CODES.get(name)
        if c is None:
            return None
        codes.append(c)
    return np.asarray(codes, dtype=np.int64)
