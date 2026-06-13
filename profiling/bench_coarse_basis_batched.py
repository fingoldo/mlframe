"""Microbench: batched coarse-basis build vs the per-freq Python loop.

``_detect_fourier_freqs_for_col`` builds the coarse sin/cos basis once per column
via a Python loop over the grid (each iter: np.sin(ang), np.cos(ang), two means,
two SS). The grid is ~16-22 fixed frequencies. Batching the angle into one outer
product + a single matrix sin/cos eval keeps the SAME per-freq numerics (centered
arrays + SS) but pays one big sin/cos C call instead of m small ones.

Run: PYTHONPATH=src python profiling/bench_coarse_basis_batched.py
"""
from __future__ import annotations
import time
import numpy as np


def build_loop(grid, z_tr):
    out = []
    for f in grid:
        ang = 2.0 * np.pi * f * z_tr
        s = np.sin(ang); c = np.cos(ang)
        sc = s - s.mean(); cc = c - c.mean()
        out.append((sc, float(sc @ sc), cc, float(cc @ cc)))
    return out


def build_batched(grid, z_tr):
    gf = np.asarray(grid, dtype=np.float64)
    ang = (2.0 * np.pi) * np.outer(gf, z_tr)  # (m, n)
    S = np.sin(ang); C = np.cos(ang)
    Sc = S - S.mean(axis=1, keepdims=True)
    Cc = C - C.mean(axis=1, keepdims=True)
    s_ss = np.einsum("ij,ij->i", Sc, Sc)
    c_ss = np.einsum("ij,ij->i", Cc, Cc)
    out = []
    for i in range(len(grid)):
        out.append((Sc[i], float(s_ss[i]), Cc[i], float(c_ss[i])))
    return out


def main():
    rng = np.random.default_rng(0)
    # typical f_grid: arange step 0.25 over ~0.25..5 => ~20 freqs
    grid = list(np.round(np.arange(0.25, 5.01, 0.25), 4))
    for n in (533, 1100, 1667, 3333):  # train-slice sizes for 800/1650/2500/5000 rows
        z = np.sort(rng.random(n))
        a = build_loop(grid, z)
        b = build_batched(grid, z)
        md = 0.0
        for (sc1, ss1, cc1, cs1), (sc2, ss2, cc2, cs2) in zip(a, b):
            md = max(md, float(np.max(np.abs(sc1 - sc2))), abs(ss1 - ss2),
                     float(np.max(np.abs(cc1 - cc2))), abs(cs1 - cs2))
        reps = 400
        t0 = time.perf_counter()
        for _ in range(reps):
            build_loop(grid, z)
        t_l = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(reps):
            build_batched(grid, z)
        t_b = time.perf_counter() - t0
        print(f"n={n:5d} m={len(grid)}  loop={t_l*1e3:7.2f}ms batched={t_b*1e3:7.2f}ms "
              f"speedup={t_l/t_b:.2f}x  max|df|={md:.2e}")


if __name__ == "__main__":
    main()
