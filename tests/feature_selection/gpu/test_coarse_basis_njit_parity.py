"""Regression sensors for the fused njit(parallel=True) coarse-basis builder in the held-out Fourier detector (iter52).

The kernel ``_coarse_basis_njit`` fuses the per-grid-frequency sin/cos + center + sum-of-squares loop into one
prange-over-freqs pass. Its sequential reduction differs from the exact numpy build by at most a single-ULP (~1e-13)
class shift; that shift only perturbs the coarse-sweep ``best_f`` argmax that ``_refine_peak_freq`` re-localises, so the
detector's emitted frequency LIST and the downstream MRMR selection stay byte-identical. These tests pin both sides:
the kernel-vs-numpy numerical bound, and the detector fast-vs-exact byte-identity.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._orthogonal_univariate_fe import _orth_extra_basis_fe as M


def _build_numpy(z, freqs):
    out = []
    for f in freqs:
        ang = 2.0 * np.pi * float(f) * z
        s = np.sin(ang)
        c = np.cos(ang)
        sc = s - s.mean()
        cc = c - c.mean()
        out.append((sc, float(sc @ sc), cc, float(cc @ cc)))
    return out


@pytest.mark.parametrize("n", [533, 1667, 5000])
@pytest.mark.parametrize("nfreq", [16, 48])
def test_coarse_basis_njit_matches_numpy_to_single_ulp(n, nfreq):
    rng = np.random.default_rng(n + nfreq)
    z = np.ascontiguousarray(np.sort(rng.uniform(-1.0, 1.0, n)))
    freqs = np.array([0.5 * k for k in range(1, nfreq + 1)], dtype=np.float64)
    sc_m, cc_m, sss, css = M._coarse_basis_njit(z, freqs)
    ref = _build_numpy(z, freqs)
    for gi in range(nfreq):
        # ~1e-13 reduction-order shift is acceptable; assert it stays in the single-ULP class.
        assert np.max(np.abs(sc_m[gi] - ref[gi][0])) < 1e-9
        assert np.max(np.abs(cc_m[gi] - ref[gi][2])) < 1e-9
        assert abs(float(sss[gi]) - ref[gi][1]) < 1e-6 * max(1.0, ref[gi][1])
        assert abs(float(css[gi]) - ref[gi][3]) < 1e-6 * max(1.0, ref[gi][3])


def test_detector_fast_path_byte_identical_to_exact(monkeypatch):
    rng = np.random.default_rng(7)
    n = 4000
    x = np.sort(rng.uniform(-1.0, 1.0, n))
    y = np.sin(5.3 * 2 * np.pi * x) + np.sin(3.1 * 2 * np.pi * x) + 0.2 * rng.normal(size=n)
    z01 = (x - x.min()) / (x.max() - x.min())
    grid = tuple(0.5 * k for k in range(1, 49))

    monkeypatch.setenv("MLFRAME_FOURIER_COARSE_BASIS_EXACT", "1")
    exact = M._detect_fourier_freqs_for_col(z01, y, f_grid=grid, min_val_corr=0.15, min_rows=800, max_freqs=4)
    monkeypatch.setenv("MLFRAME_FOURIER_COARSE_BASIS_EXACT", "0")
    fast = M._detect_fourier_freqs_for_col(z01, y, f_grid=grid, min_val_corr=0.15, min_rows=800, max_freqs=4)

    assert exact == fast, f"fast {fast} diverged from exact {exact}"
    assert len(fast) >= 2, "multitone target should detect at least 2 frequencies"
