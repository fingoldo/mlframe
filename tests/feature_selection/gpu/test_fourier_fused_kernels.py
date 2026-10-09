"""The fused Fourier detector kernels agree with the cupy-op implementation they replace."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_detect_gpu_resident as det
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_fused_gpu as fused

GRID = [float(f) for f in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32)]


def _column(n, seed, freq=3.1):
    """A uniform column and a noisy tone at ``freq`` (centred target)."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(size=n)
    y = np.sin(2 * np.pi * freq * z) + 0.4 * rng.normal(size=n) + 0.2 * z
    return cp.asarray(z), cp.asarray(y - y.mean())


def _set(monkeypatch, on):
    """Select the fused or the cupy-op implementation."""
    monkeypatch.setenv("MLFRAME_FE_FOURIER_FUSED", "1" if on else "0")


@pytest.mark.parametrize("n", [900, 10_000, 66_000])
def test_power_grid_matches_the_cupy_batch(n, monkeypatch):
    """Per-frequency power equals the (F, n) batched cupy implementation, so every argmax is the same."""
    z, yc = _column(n, 1)
    y_ss = float(cp.dot(yc, yc))
    freqs = cp.asarray([*GRID, 3.07, 3.1, 3.12])
    _set(monkeypatch, False)
    want = cp.asnumpy(det._power_grid_centered_gpu(cp, z, yc, y_ss, freqs))
    _set(monkeypatch, True)
    got = cp.asnumpy(det._power_grid_centered_gpu(cp, z, yc, y_ss, freqs))
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)
    assert int(np.argmax(got)) == int(np.argmax(want))


def test_scalar_power_matches(monkeypatch):
    """The held-out confirmation power is the same number."""
    z, yc = _column(5_000, 2)
    y_ss = float(cp.dot(yc, yc))
    _set(monkeypatch, False)
    want = det._power_centered_gpu(cp, z, yc, y_ss, 3.1)
    _set(monkeypatch, True)
    assert det._power_centered_gpu(cp, z, yc, y_ss, 3.1) == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("freq", [1.0, 3.1, 12.5])
def test_deflation_matches(freq, monkeypatch):
    """The residual after removing [1, sin, cos] equals the normal-equations residual."""
    z, y = _column(20_000, 3)
    _set(monkeypatch, False)
    want = cp.asnumpy(det._deflate_sincos_gpu(cp, z, y, freq))
    _set(monkeypatch, True)
    got = cp.asnumpy(det._deflate_sincos_gpu(cp, z, y, freq))
    np.testing.assert_allclose(got, want, rtol=1e-8, atol=1e-10)


def test_singular_deflation_falls_back(monkeypatch):
    """A frequency of 0 makes sin identically 0: the 3x3 system is singular, the fused kernel declines and the original path answers exactly as before."""
    z, y = _column(2_000, 4)
    assert fused.deflate_sincos(cp, z, y, 0.0) is None
    _set(monkeypatch, False)
    want = cp.asnumpy(det._deflate_sincos_gpu(cp, z, y, 0.0))
    _set(monkeypatch, True)
    np.testing.assert_array_equal(cp.asnumpy(det._deflate_sincos_gpu(cp, z, y, 0.0)), want)


@pytest.mark.parametrize("freq", [3.1, 5.0, 7.3, 11.0])
def test_detected_frequencies_are_the_same(freq, monkeypatch):
    """End to end: the detector returns the same frequency list (possibly empty) with and without the fused kernels."""
    rng = np.random.default_rng(int(freq * 10))
    n = 20_000
    z = rng.uniform(size=n)
    y = np.sin(2 * np.pi * freq * z) + 0.5 * np.sin(2 * np.pi * 2.0 * freq * z) + 0.3 * rng.normal(size=n)
    _set(monkeypatch, False)
    want = det.detect_fourier_freqs_for_col_gpu(z, y, f_grid=GRID, min_rows=800, fourier_detect_max_n=30_000)
    _set(monkeypatch, True)
    got = det.detect_fourier_freqs_for_col_gpu(z, y, f_grid=GRID, min_rows=800, fourier_detect_max_n=30_000)
    assert len(got) == len(want)
    np.testing.assert_allclose(got, want, atol=0.02)
