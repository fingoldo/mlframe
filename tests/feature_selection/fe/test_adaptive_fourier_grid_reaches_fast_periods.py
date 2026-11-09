"""The adaptive Fourier detector finds a 20-cycle signal on enough rows: its coarse grid used to stop at 16 cycles."""

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe import _detect_fourier_freqs_for_col
from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe_generate import adaptive_fourier_grid


def test_grid_grows_with_the_rows_and_keeps_the_old_range_at_small_n():
    """0.5 steps to 16 always; the 1.0-step tail up to clip(n / 250, 16, 64) only when the rows support it."""
    small, mid, big = adaptive_fourier_grid(500), adaptive_fourier_grid(14000), adaptive_fourier_grid(10**6)
    assert small == tuple(0.5 * k for k in range(1, 33))
    assert mid[: len(small)] == small and mid[-1] == 56.0 and big[-1] == 64.0
    assert list(mid) == sorted(set(mid))


@pytest.mark.parametrize("freq", [20.0, 30.0])
def test_detector_locks_a_fast_oscillation_with_the_row_dependent_grid(freq):
    """A hidden periodic target at 20 or 30 cycles over the column's range is detected; the old 16-cycle grid returned nothing."""
    rng = np.random.default_rng(0)
    n = 14000
    z = rng.uniform(0.0, 1.0, n)
    y = (np.sin(2 * np.pi * freq * z) + 0.5 * rng.standard_normal(n) > 0).astype(float)
    got = _detect_fourier_freqs_for_col(z, y, f_grid=adaptive_fourier_grid(n), min_val_corr=0.15)
    assert any(abs(f - freq) < 0.3 for f in got)
    assert _detect_fourier_freqs_for_col(z, y, f_grid=tuple(0.5 * k for k in range(1, 33)), min_val_corr=0.15) == []


def test_pure_noise_gets_no_frequency_on_the_wider_grid():
    """The held-out gate keeps the extra grid points from admitting chance peaks."""
    rng = np.random.default_rng(1)
    n = 14000
    z = rng.uniform(0.0, 1.0, n)
    for seed in range(5):
        y = np.random.default_rng(seed).standard_normal(n)
        assert _detect_fourier_freqs_for_col(z, y, f_grid=adaptive_fourier_grid(n), min_val_corr=0.15) == []
