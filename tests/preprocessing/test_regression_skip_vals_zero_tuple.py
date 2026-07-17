"""Regression: _get_nunique was called with skip_vals=(0.0) (a float, not a tuple),
so the intended skip of value 0.0 was silently dropped, mis-counting integer-part uniques
in is_variable_truly_continuous."""

import numpy as np

from mlframe.preprocessing.cleaning import _get_nunique


def test_get_nunique_skips_zero_as_single_value_tuple():
    """Get nunique skips zero as single value tuple."""
    vals = np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype=float)
    # With 0.0 skipped, three distinct non-zero values remain.
    assert _get_nunique(vals=vals, skip_vals=(0.0,)) == 3


def test_get_nunique_float_arg_does_not_skip_zero():
    # Pre-fix the call site passed (0.0) -- a float -- which _get_nunique treats as "no
    # skip_vals" (falsy / not a tuple): 0.0 is then counted. This pins the difference so a
    # future regression to the float form is caught at the call boundary.
    """Get nunique float arg does not skip zero."""
    vals = np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype=float)
    counted_with_zero = _get_nunique(vals=vals, skip_vals=(0.0))  # float -> behaves as no skip
    skipped_zero = _get_nunique(vals=vals, skip_vals=(0.0,))
    assert counted_with_zero == 4
    assert skipped_zero == 3
    assert counted_with_zero != skipped_zero
