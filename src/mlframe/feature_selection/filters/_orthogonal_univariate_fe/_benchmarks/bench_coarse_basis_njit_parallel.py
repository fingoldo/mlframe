"""SHIPPED (iter52, 2026-06-13): njit(parallel=True) prange-over-freqs coarse-basis build for ``_detect_fourier_freqs_for_col``.

The detector's own-frame cProfile cost (scene n=12000 MRMR.fit: 1.050s tottime / 68 calls) is dominated by the
per-grid-frequency coarse-basis build: for each f in the f_grid (16 adaptive / 48 chirp) it does ``np.sin(2*pi*f*z)``,
``np.cos(...)``, mean-subtract, and two dot products. ``_coarse_basis_njit`` fuses that build into ONE njit(parallel=True)
kernel with a ``prange`` over the frequencies (each iteration computes the n-length sin/cos pass + centered SS for one
frequency, the freqs spread across cores).

VERDICT: SHIPPED as default. Warm steady-state measurement (this bench) wins consistently across all sizes:
       chirp   nf=48  n=533  6.79x | n=1667  8.22x | n=5000  2.45x | n=8000  3.25x
       adaptive nf=16 n=533 9.07x | n=1667  7.36x | n=5000  9.20x | n=8000  7.18x
(An earlier run under heavy machine contention + cold cache mis-read this as noisy/0.6x at n=5000 -- a single cold shot
is not a measurement; the warm bench is the verdict.) The sequential per-element accumulation of the mean / SS in the
njit loop differs from numpy's pairwise summation by ~1e-13 (maxd 5.7e-14 .. 9.1e-13), so it is NOT bit-identical -- but
the shift only perturbs the coarse-sweep periodogram-power ``argmax`` (which picks ``best_f`` BEFORE ``_refine_peak_freq``
re-localises it), and end-to-end MRMR scene selection is BYTE-IDENTICAL (verified: identical engineered-feature list incl.
the Fourier ``f12__qsin5.45`` / ``f12__qcos5.45`` / ``f13__qcos7.95`` columns vs the exact numpy path). Gated to the FE
detector; ``MLFRAME_FOURIER_COARSE_BASIS_EXACT=1`` forces the exact numpy build.

Run: python -m mlframe.feature_selection.filters._orthogonal_univariate_fe._benchmarks.bench_coarse_basis_njit_parallel
"""
from __future__ import annotations

import time

import numpy as np

from .._orth_extra_basis_fe import _coarse_basis_njit as _build_basis_njit


def _build_numpy(z, freqs):
    cb = []
    for f in freqs:
        ang = 2.0 * np.pi * float(f) * z
        s = np.sin(ang)
        c = np.cos(ang)
        scen = s - s.mean()
        ccen = c - c.mean()
        cb.append((scen, float(scen @ scen), ccen, float(ccen @ ccen)))
    return cb


def main():
    for nfreq, label in ((48, "chirp"), (16, "adaptive")):
        for n in (533, 1667, 5000, 8000):
            rng = np.random.default_rng(0)
            z = np.sort(rng.uniform(-1, 1, n))
            freqs = np.array([0.5 * k for k in range(1, nfreq + 1)])
            _build_basis_njit(z, freqs)
            cbn = _build_numpy(z, freqs)
            sc, _cc, sss, _css = _build_basis_njit(z, freqs)
            maxd = 0.0
            for fi in range(len(freqs)):
                maxd = max(maxd, float(np.max(np.abs(sc[fi] - cbn[fi][0]))), abs(sss[fi] - cbn[fi][1]))
            reps = 50
            t = time.perf_counter()
            for _ in range(reps):
                _build_numpy(z, freqs)
            to = time.perf_counter() - t
            t = time.perf_counter()
            for _ in range(reps):
                _build_basis_njit(z, freqs)
            tn = time.perf_counter() - t
            print(f"{label} nf={nfreq} n={n:5d} numpy {to / reps * 1e3:6.3f}ms njit {tn / reps * 1e3:6.3f}ms speedup {to / tn:.2f}x maxd={maxd:.2e}")


if __name__ == "__main__":
    main()
