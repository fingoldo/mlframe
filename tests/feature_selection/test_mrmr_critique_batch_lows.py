"""MRMR critique Low-tier fixes.

- EN-4: _coerce_to_int_with_nan_handling rounded floats toward zero, so a fit-time integer code that round-tripped
  through float (2.9999) coded as 2 instead of 3. Now rounds to nearest.
- EN-3: mi_greedy_transform recipe raised a cryptic np.asarray error on a non-numeric source; now a clear one.
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters.engineered_recipes._recipe_extract import _coerce_to_int_with_nan_handling


def test_float_roundtrip_integer_code_rounds_to_nearest():
    # a raw-integer ordinal source arriving as float (int->float round-trip) must recover the fit-time code
    """Float roundtrip integer code rounds to nearest."""
    vals = np.array([0.0, 0.9999999, 2.0000001, 2.9999999, 4.0], dtype=np.float64)
    codes = _coerce_to_int_with_nan_handling(vals, n_bins=10, recipe_name="r", col_name="c", unknown_strategy="clip")
    assert list(codes) == [0, 1, 2, 3, 4], f"float round-trip codes truncated instead of rounded: {list(codes)}"


def test_exact_integer_floats_unchanged():
    """Exact integer floats unchanged."""
    vals = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    codes = _coerce_to_int_with_nan_handling(vals, n_bins=10, recipe_name="r", col_name="c", unknown_strategy="clip")
    assert list(codes) == [0, 1, 2, 3]


def test_mi_greedy_non_numeric_source_clear_error():
    """Mi greedy non numeric source clear error."""
    import pandas as pd
    from mlframe.feature_selection.filters.engineered_recipes._missingness_ratio_recipes import (
        _apply_mi_greedy_transform,
        build_mi_greedy_transform_recipe,
    )

    recipe = build_mi_greedy_transform_recipe(name="mg", transform="identity", src_names=("s",))
    X = pd.DataFrame({"s": ["not", "a", "number"]})
    with pytest.raises(ValueError, match="non-numeric dtype"):
        _apply_mi_greedy_transform(recipe, X)


def test_argmax_gate_serve_nan_propagates_not_spurious_code():
    """Argmax gate serve nan propagates not spurious code."""
    import pandas as pd
    from mlframe.feature_selection.filters._conditional_gate_fe import apply_row_argmax, apply_conditional_gate

    # row 1 has a NaN -> argmax must be NaN, not the spurious first-NaN index (0)
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [2.0, 5.0, 1.0]})
    am = apply_row_argmax(X, ["a", "b"])
    assert am[0] == 1.0 and np.isnan(am[1]) and am[2] == 0.0, f"argmax serve-NaN not propagated: {am}"

    # gate select: NaN gating column c -> NaN (not silently routed to b)
    Xg = pd.DataFrame({"a": [10.0, 10.0], "b": [20.0, 20.0], "c": [1.0, np.nan]})
    sel = apply_conditional_gate(Xg, "select", ["a", "b", "c"], tau=0.5)
    assert sel[0] == 10.0 and np.isnan(sel[1]), f"gate select serve-NaN not propagated: {sel}"

    # gate mask: NaN c -> NaN
    Xm = pd.DataFrame({"a": [7.0, 7.0], "c": [1.0, np.nan]})
    mk = apply_conditional_gate(Xm, "mask", ["a", "c"], tau=0.5)
    assert mk[0] == 7.0 and np.isnan(mk[1]), f"gate mask serve-NaN not propagated: {mk}"
