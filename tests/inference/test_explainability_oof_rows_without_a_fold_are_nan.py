"""OOF SHAP / base-value assembly turns rows no fold tested on into NaN instead of an object array holding None."""

from __future__ import annotations

import numpy as np

from mlframe.inference.explainability import _stack_oof_rows


def test_untested_rows_become_nan_rows_of_the_same_shape():
    """Rows 0-1 never tested (TimeSeriesSplit warm-up): the stacked matrix is (4, 3) float with NaN rows there."""
    rows = [None, None, np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])]
    out = _stack_oof_rows(rows)
    assert out.shape == (4, 3) and out.dtype == np.float64
    assert np.isnan(out[:2]).all() and out[3].tolist() == [4.0, 5.0, 6.0]


def test_scalar_base_values_stack_to_a_vector():
    """Base values may be scalars per row: None becomes a NaN entry."""
    out = _stack_oof_rows([None, 0.5, 0.25])
    assert out.shape == (3,) and np.isnan(out[0]) and out[2] == 0.25


def test_fully_covered_rows_are_unchanged():
    """No None, plain stack (negative control)."""
    rows = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    np.testing.assert_array_equal(_stack_oof_rows(rows), np.array(rows))
