"""Unit test for the single-pass finite-min/max/std helper used by the
``output_activation='tanh_train_range'`` auto-derive path.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.neural._neural_numba_kernels import (
    finite_min_max_std,
    _finite_min_max_std_python,
)


class TestFiniteMinMaxStd:
    def test_matches_numpy_on_clean_array(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(1000)
        n, mn, mx, mu, sd = finite_min_max_std(y)
        assert n == 1000
        np.testing.assert_allclose(mn, y.min(), atol=1e-12)
        np.testing.assert_allclose(mx, y.max(), atol=1e-12)
        np.testing.assert_allclose(mu, y.mean(), atol=1e-9)
        np.testing.assert_allclose(sd, y.std(), atol=1e-9)

    def test_skips_nan_inf(self):
        rng = np.random.default_rng(1)
        y = rng.standard_normal(500)
        y[::20] = np.nan
        y[::50] = np.inf
        y[::100] = -np.inf
        finite = y[np.isfinite(y)]
        n, mn, mx, mu, sd = finite_min_max_std(y)
        assert n == finite.size
        np.testing.assert_allclose(mn, finite.min(), atol=1e-12)
        np.testing.assert_allclose(mx, finite.max(), atol=1e-12)
        np.testing.assert_allclose(mu, finite.mean(), atol=1e-9)
        np.testing.assert_allclose(sd, finite.std(), atol=1e-9)

    def test_all_nan_returns_zeros(self):
        y = np.full(10, np.nan)
        n, mn, mx, mu, sd = finite_min_max_std(y)
        assert (n, mn, mx, mu, sd) == (0, 0.0, 0.0, 0.0, 0.0)

    def test_single_finite_value_returns_zero_std(self):
        y = np.array([np.nan, 3.14, np.inf, np.nan])
        n, mn, mx, mu, sd = finite_min_max_std(y)
        assert n == 1
        assert mn == mx == mu == 3.14
        assert sd == 0.0

    def test_high_range_target_no_catastrophic_cancellation(self):
        """Welford's algorithm stays accurate on y values shifted by a
        large offset; the naive ``E[X^2] - E[X]^2`` formula would lose
        precision on this input."""
        rng = np.random.default_rng(42)
        offset = 1e7
        y = (offset + rng.standard_normal(10_000)).astype(np.float64)
        n, mn, mx, mu, sd = finite_min_max_std(y)
        assert n == 10_000
        # Mean and std should be very close to (offset, 1.0) within numerical noise.
        np.testing.assert_allclose(mu, y.mean(), atol=1e-6, rtol=1e-12)
        np.testing.assert_allclose(sd, y.std(), atol=1e-6, rtol=1e-9)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
    def test_dtypes(self, dtype):
        y = np.array([1, 2, 3, 4, 5], dtype=dtype)
        n, mn, mx, mu, sd = finite_min_max_std(y)
        assert n == 5
        assert mn == 1.0
        assert mx == 5.0
        np.testing.assert_allclose(mu, 3.0)
        np.testing.assert_allclose(sd, np.sqrt(2.0))

    def test_python_fallback_matches_njit(self):
        """The pure-Python fallback used on hosts without numba must
        return the SAME tuple as the @njit kernel."""
        rng = np.random.default_rng(7)
        y = rng.standard_normal(2000)
        y[::30] = np.nan
        njit_result = finite_min_max_std(y)
        py_result = _finite_min_max_std_python(y.astype(np.float64))
        assert njit_result[0] == py_result[0]
        np.testing.assert_allclose(njit_result[1:], py_result[1:], atol=1e-12)
