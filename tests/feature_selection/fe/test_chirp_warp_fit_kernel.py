"""The compiled chirp-warp fit gives the parameters of its numpy form (to the accumulation precision) and handles the degenerate columns the same way."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_extra_basis_fe as ebf


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_parameters_equal_the_numpy_form(dtype):
    """mean, std, lo and span agree to 1e-6 relative on a skewed column with missing and infinite values."""
    rng = np.random.default_rng(0)
    x = np.exp(rng.standard_normal(50000)).astype(dtype)
    x[::97] = np.nan
    x[5] = np.inf
    x[9] = -np.inf
    got, want = ebf._fit_chirp_warp_for_col(x), ebf._fit_chirp_warp_numpy(x)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-9)


def test_degenerate_columns_answer_as_before():
    """All missing: the identity warp; constant: std 1, span at the floor; a single finite value; an empty column."""
    for x in (np.full(20, np.nan), np.full(20, 3.0), np.array([np.nan, 2.0, np.nan]), np.empty(0)):
        got, want = ebf._fit_chirp_warp_for_col(x), ebf._fit_chirp_warp_numpy(x)
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-9, err_msg=repr(x[:3]))


def test_the_axis_of_the_fitted_warp_spans_the_unit_interval():
    """Replaying the stored parameters maps the finite values onto [0, 1] (the property the recipes rely on)."""
    x = np.random.default_rng(1).standard_normal(5000).astype(np.float32)
    mean, std, lo, span = ebf._fit_chirp_warp_for_col(x)
    axis = ebf._chirp_axis(x, mean, std, lo, span)
    assert abs(float(axis.min())) < 1e-6 and abs(float(axis.max()) - 1.0) < 1e-6
