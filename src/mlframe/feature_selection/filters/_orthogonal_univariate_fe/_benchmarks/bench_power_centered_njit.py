"""Microbench: fused njit `_power_centered` vs the numpy baseline.

`_power_centered(z, yc, y_ss, freq)` is the hottest leaf in the multi-frequency
Fourier detector (`_orth_extra_basis_fe._detect_fourier_freqs_for_col` ->
`_refine_peak_freq`): ~13860 calls in the scene-2500 MRMR profile, 0.69s tottime.
Per call it allocates `ang`, `sin`, `cos`, two centered copies, and walks the
length-n_train array ~6 times. A single fused njit pass computes the two squared
correlations in ONE loop with no temporaries.

Run:
  PYTHONPATH=<worktree>/src python bench_power_centered_njit.py
"""
from __future__ import annotations
import time

import numba
import numpy as np

from mlframe.feature_selection.filters._orthogonal_univariate_fe._orth_extra_basis_fe import (
    _power_centered,
)


@numba.njit(cache=True, fastmath=True)
def _power_centered_njit(z, yc, y_ss, freq):
    """Rejected fused candidate (kept here per REJECTED != DELETED)."""
    n = z.shape[0]
    w = 2.0 * np.pi * freq
    s_sum = 0.0
    c_sum = 0.0
    for i in range(n):
        a = w * z[i]
        s_sum += np.sin(a)
        c_sum += np.cos(a)
    s_mean = s_sum / n
    c_mean = c_sum / n
    s_ss = 0.0
    c_ss = 0.0
    num_s = 0.0
    num_c = 0.0
    for i in range(n):
        a = w * z[i]
        sc = np.sin(a) - s_mean
        cc = np.cos(a) - c_mean
        s_ss += sc * sc
        c_ss += cc * cc
        num_s += sc * yc[i]
        num_c += cc * yc[i]
    p = 0.0
    if s_ss >= 1e-24 and y_ss >= 1e-24:
        p += (num_s * num_s) / (s_ss * y_ss)
    if c_ss >= 1e-24 and y_ss >= 1e-24:
        p += (num_c * num_c) / (c_ss * y_ss)
    return p


def main():
    rng = np.random.default_rng(0)
    for n in (800, 1667, 5000):  # scene-2500 train slice ~1667
        z = np.sort(rng.random(n))
        y = np.sin(2 * np.pi * 3.3 * z) + 0.3 * rng.standard_normal(n)
        yc = y - y.mean()
        y_ss = float(yc @ yc)
        freqs = [0.05 + 0.0125 * k for k in range(40)]

        # warm + correctness
        for f in freqs[:3]:
            a = _power_centered(z, yc, y_ss, f)
            b = _power_centered_njit(z, yc, y_ss, f)
            assert abs(a - b) < 1e-10, (n, f, a, b)

        reps = 3000
        t0 = time.perf_counter()
        for _ in range(reps):
            for f in freqs:
                _power_centered(z, yc, y_ss, f)
        t_np = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(reps):
            for f in freqs:
                _power_centered_njit(z, yc, y_ss, f)
        t_nj = time.perf_counter() - t0

        per_np = t_np / (reps * len(freqs)) * 1e6
        per_nj = t_nj / (reps * len(freqs)) * 1e6
        print(f"n={n:5d}  numpy={per_np:7.2f}us  njit={per_nj:7.2f}us  speedup={per_np/per_nj:.2f}x")


if __name__ == "__main__":
    main()
