"""The one-pass ``_finite_std`` equals the numpy form for float32 / float64 columns with non-finite entries and for non-float input (audit-driven, serial tail of the fit)."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._fe_pure_form_retention import _finite_std, _finite_std_numpy


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kernel_equals_numpy_form_with_nonfinite_entries(dtype):
    """Random column with NaN / inf entries, an all-NaN column, a constant column and an empty one."""
    rng = np.random.default_rng(0)
    col = (rng.normal(size=5000) * 7.0 + 100.0).astype(dtype)
    col[::13] = np.nan
    col[5] = np.inf
    col[9] = -np.inf
    assert _finite_std(col) == pytest.approx(_finite_std_numpy(col), rel=1e-9)
    assert _finite_std(np.full(10, np.nan, dtype=dtype)) == 0.0
    assert _finite_std(np.full(10, 3.0, dtype=dtype)) == 0.0
    assert _finite_std(np.empty(0, dtype=dtype)) == 0.0


def test_integer_and_list_input_use_the_numpy_form():
    """Integer arrays and plain lists keep the numpy route."""
    assert _finite_std(np.arange(10)) == pytest.approx(_finite_std_numpy(np.arange(10)))
    assert _finite_std([1.0, 2.0, np.nan]) == pytest.approx(0.5)
