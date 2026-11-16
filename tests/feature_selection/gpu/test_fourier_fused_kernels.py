"""The fused Fourier detector kernels agree with the cupy-op implementation they replace."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._fe_usability_signal import _crit_np_dtype
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_detect_gpu_resident as det
from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_fused_gpu as fused

# both implementations read the samples in the criterion dtype and accumulate in double: they agree to a few units of that dtype's epsilon, not to float64 round-off
_RTOL = 4 * float(np.finfo(_crit_np_dtype()).eps)

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
    np.testing.assert_allclose(got, want, rtol=_RTOL, atol=_RTOL * float(np.max(np.abs(want))))
    assert int(np.argmax(got)) == int(np.argmax(want))


def test_scalar_power_matches(monkeypatch):
    """The held-out confirmation power is the same number."""
    z, yc = _column(5_000, 2)
    y_ss = float(cp.dot(yc, yc))
    _set(monkeypatch, False)
    want = det._power_centered_gpu(cp, z, yc, y_ss, 3.1)
    _set(monkeypatch, True)
    assert det._power_centered_gpu(cp, z, yc, y_ss, 3.1) == pytest.approx(want, rel=_RTOL)


@pytest.mark.parametrize("freq", [1.0, 3.1, 12.5])
def test_deflation_matches(freq, monkeypatch):
    """The residual after removing [1, sin, cos] equals the normal-equations residual."""
    z, y = _column(20_000, 3)
    _set(monkeypatch, False)
    want = cp.asnumpy(det._deflate_sincos_gpu(cp, z, y, freq))
    _set(monkeypatch, True)
    got = cp.asnumpy(det._deflate_sincos_gpu(cp, z, y, freq))
    np.testing.assert_allclose(got, want, rtol=_RTOL, atol=_RTOL * float(np.max(np.abs(want))))


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


@pytest.mark.parametrize("n", [900, 66_000])
def test_float32_trig_keeps_the_power_grid_and_its_argmax(n):
    """float32 samples take the float sine / cosine of a double-reduced argument: the powers agree with the float64 run to 1e-5 of the largest and every argmax is the same."""
    z, yc = _column(n, 3)
    y_ss = float(cp.dot(yc, yc))
    freqs = cp.asarray([*GRID, 3.07, 3.1, 3.12, 40.0, 77.3])
    want = cp.asnumpy(fused.power_sincos(cp, z, yc, y_ss, freqs))
    got = cp.asnumpy(fused.power_sincos(cp, z.astype(cp.float32), yc.astype(cp.float32), y_ss, freqs))
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5 * float(np.max(np.abs(want))))
    assert int(np.argmax(got)) == int(np.argmax(want))


def test_short_frequency_lists_get_wide_blocks_and_the_same_powers(monkeypatch):
    """With fewer than 8 blocks per SM (a few frequencies) the power kernel launches blocks of at least 256 threads; a long list keeps the occupancy rule's block. The powers agree between the widths to summation order."""
    cp = pytest.importorskip("cupy")
    import numpy as np

    from mlframe.feature_selection.filters._orthogonal_univariate_fe import _fourier_fused_gpu as G

    rng = np.random.default_rng(1)
    n = 6000
    z = cp.asarray(rng.uniform(0, 1, n))
    y = rng.normal(size=n)
    y -= y.mean()
    yc = cp.asarray(y)
    y_ss = float((y**2).sum())
    real_kernel = G.kernel_power(cp, z.dtype)
    launches = []

    class _Spy:
        """Delegates to the real kernel and records the launch geometry."""

        def __call__(self, grid, block, args):
            launches.append((grid[0], block[0]))
            return real_kernel(grid, block, args)

        def __getattr__(self, name):
            return getattr(real_kernel, name)

    monkeypatch.setattr(G, "kernel_power", lambda cp_, real: _Spy())
    sms = int(cp.cuda.Device().attributes["MultiProcessorCount"])
    few = cp.asarray(np.linspace(1.0, 20.0, 4))
    many = cp.asarray(np.linspace(1.0, 20.0, 8 * sms))
    p_few = cp.asnumpy(G.power_sincos(cp, z, yc, y_ss, few))
    G.power_sincos(cp, z, yc, y_ss, many)
    assert launches[0][1] >= 256
    from pyutilz.performance.gpu_hw_launch import block_size_for

    assert launches[1][1] == block_size_for(real_kernel, work_items=n, max_width=1024)  # a long list keeps the occupancy rule's block
    ref = []
    for f in cp.asnumpy(few):
        s, c = np.sin(2 * np.pi * f * cp.asnumpy(z)), np.cos(2 * np.pi * f * cp.asnumpy(z))
        ref.append(sum((np.dot(v, y) ** 2) / ((np.dot(v, v) - v.sum() ** 2 / n) * y_ss) for v in (s, c)))
    np.testing.assert_allclose(p_few, ref, rtol=1e-6)
