"""The degenerate-column audit reports an all-infinite column as constant.

``np.ptp`` over ``[inf, inf]`` is ``inf - inf = nan``, and ``nan == 0`` is False, so a zero-variance infinite column was reported as
neither constant nor all-NaN, and fell through the audit with no reason at all.
"""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._mrmr_degenerate import _is_constant


def test_all_positive_inf_column_is_constant():
    """A column of +inf has one distinct value."""
    assert _is_constant(np.full(8, np.inf)) is True


def test_all_negative_inf_with_nans_is_constant():
    """NaNs are ignored; the remaining -inf values are a single distinct value."""
    assert _is_constant(np.array([-np.inf, np.nan, -np.inf, -np.inf])) is True


def test_mixed_infinities_and_finite_columns_keep_their_verdicts():
    """Controls: +inf and -inf together are not constant, an ordinary constant is, and varying finite values are not."""
    assert _is_constant(np.array([np.inf, -np.inf, np.inf])) is False
    assert _is_constant(np.full(5, 3.5)) is True
    assert _is_constant(np.array([1.0, 2.0, np.inf])) is False
    assert _is_constant(np.full(4, np.nan)) is False
