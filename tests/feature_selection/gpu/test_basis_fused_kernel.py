"""The single-kernel orthogonal-polynomial design is bit-identical to the cupy recurrence it replaces."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters.hermite_fe import _hermite_prewarp_gpu_resident as pw
from mlframe.feature_selection.filters.hermite_fe._basis_fused_gpu import MAX_COLUMNS, build_basis_fused


def _both(monkeypatch, basis, z, degree):
    """The design built with the loop and with the kernel."""
    monkeypatch.setenv("MLFRAME_FE_BASIS_FUSED", "0")
    loop = cp.asnumpy(pw._build_basis_matrix_gpu(cp, basis, z, degree))
    monkeypatch.setenv("MLFRAME_FE_BASIS_FUSED", "1")
    return loop, cp.asnumpy(pw._build_basis_matrix_gpu(cp, basis, z, degree))


@pytest.mark.parametrize("basis", ["hermite", "legendre", "chebyshev", "laguerre"])
@pytest.mark.parametrize("degree", [1, 2, 3, 6, 9])
def test_bit_identical_to_the_loop(monkeypatch, basis, degree):
    """Every entry equal, not merely close - the kernel rounds where cupy's separate elementwise kernels do."""
    rng = np.random.default_rng(degree)
    z = cp.asarray(rng.normal(size=20_001))
    loop, fused = _both(monkeypatch, basis, z, degree)
    assert loop.shape == fused.shape == (20_001, degree + 1)
    np.testing.assert_array_equal(fused, loop)


def test_float32_input_is_widened_like_the_loop(monkeypatch):
    """The recurrence runs in float64 whatever the input dtype."""
    z = cp.asarray(np.random.default_rng(1).normal(size=5_000).astype(np.float32))
    loop, fused = _both(monkeypatch, "hermite", z, 5)
    np.testing.assert_array_equal(fused, loop)


def test_requests_outside_the_kernel_fall_back():
    """An unknown basis or a design wider than the unrolled budget returns None so the caller keeps the loop."""
    x = cp.asarray(np.linspace(-1, 1, 100))
    assert build_basis_fused(cp, "wavelet", x, 4) is None
    assert build_basis_fused(cp, "hermite", x, MAX_COLUMNS + 1) is None
    assert build_basis_fused(cp, "hermite", x, 0) is None


def test_extreme_values_overflow_the_same_way(monkeypatch):
    """Large |z| overflows to inf/nan in both implementations at the same entries."""
    z = cp.asarray(np.array([0.0, 1e3, -1e3, 1e100, np.nan, np.inf]))
    loop, fused = _both(monkeypatch, "laguerre", z, 8)
    np.testing.assert_array_equal(np.isnan(fused), np.isnan(loop))
    np.testing.assert_array_equal(fused[np.isfinite(loop)], loop[np.isfinite(loop)])
