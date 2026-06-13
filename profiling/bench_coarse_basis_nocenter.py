"""Microbench: coarse-basis precompute build in _detect_fourier_freqs_for_col without the mean-subtract temporaries.

The grid loop's numerator ``basis_centered @ yc`` is identity-equal to ``basis_raw @ yc`` because ``yc`` sums to zero (the
``mean*sum(yc)`` cross-term vanishes), and the centered sum-of-squares is ``raw@raw - sum(raw)**2/n``. So the build can store the
RAW sin/cos arrays plus their centered SS and skip the two length-n ``s - s.mean()`` / ``c - c.mean()`` temporaries per grid freq.
Same no-alloc identity already shipped in ``_corr_sq_centered`` (see profiling/bench_corr_sq_noalloc.py); selection is bit-identical
to <1e-12 on the periodogram power.

Run: PYTHONPATH=src python profiling/bench_coarse_basis_nocenter.py
"""
from __future__ import annotations
import time
import numpy as np


def build_centered(grid, z_tr):
    out = []
    for f in grid:
        ang = 2.0 * np.pi * f * z_tr
        s = np.sin(ang); c = np.cos(ang)
        sc = s - s.mean(); cc = c - c.mean()
        out.append((sc, float(sc @ sc), cc, float(cc @ cc)))
    return out


def build_nocenter(grid, z_tr):
    m = z_tr.shape[0]
    out = []
    for f in grid:
        ang = 2.0 * np.pi * f * z_tr
        s = np.sin(ang); c = np.cos(ang)
        s_sum = float(s.sum()); c_sum = float(c.sum())
        out.append((s, float(s @ s) - s_sum * s_sum / m, c, float(c @ c) - c_sum * c_sum / m))
    return out


def main():
    rng = np.random.default_rng(0)
    adaptive = tuple(0.5 * k for k in range(1, 17))
    chirp = tuple(0.5 * k for k in range(1, 49))
    for gn, grid in (("adaptive16", adaptive), ("chirp48", chirp)):
        for n in (800, 1100, 1667, 3333):
            z = np.sort(rng.random(n))
            z_tr = z[(np.arange(n) % 3) != 0]
            # identity check on the periodogram numerator/power
            yc = np.sin(2 * np.pi * 3.5 * z_tr); yc = yc - yc.mean(); y_ss = float(yc @ yc)
            bo = build_centered(grid, z_tr); bn = build_nocenter(grid, z_tr)
            md = 0.0
            for (sc, sss, cc, css), (s, sss2, c, css2) in zip(bo, bn):
                po = (float(sc @ yc) ** 2) / (sss * y_ss) + (float(cc @ yc) ** 2) / (css * y_ss)
                pn = (float(s @ yc) ** 2) / (sss2 * y_ss) + (float(c @ yc) ** 2) / (css2 * y_ss)
                md = max(md, abs(po - pn))
            reps = 400
            t0 = time.perf_counter()
            for _ in range(reps):
                build_centered(grid, z_tr)
            t_o = time.perf_counter() - t0
            t0 = time.perf_counter()
            for _ in range(reps):
                build_nocenter(grid, z_tr)
            t_n = time.perf_counter() - t0
            print(f"{gn:11s} n={n:5d}  centered={t_o/reps*1e3:7.3f}ms  nocenter={t_n/reps*1e3:7.3f}ms  "
                  f"speedup={t_o/t_n:.2f}x  max|dpower|={md:.2e}")


if __name__ == "__main__":
    main()
