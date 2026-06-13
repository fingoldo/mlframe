"""Microbench: _corr_sq_centered without the vc temporary.

Since y_centered sums to ~0, num = vc @ yc == v @ yc exactly (the v.mean()*sum(yc)
cross term vanishes). And v_ss = vc@vc = v@v - sum(v)**2/n. So we can drop the
length-n ``vc = v - v.mean()`` allocation and compute everything from raw v dots.

This changes FP reduction order slightly, so it is NOT bit-identical -- gate the
decision on the measured divergence + the periodogram-power impact.

Run: PYTHONPATH=src python profiling/bench_corr_sq_noalloc.py
"""
from __future__ import annotations
import time
import numpy as np


def corr_sq_orig(v, yc, y_ss):
    vc = v - v.mean()
    v_ss = float(vc @ vc)
    if v_ss < 1e-24 or y_ss < 1e-24:
        return 0.0
    num = float(vc @ yc)
    return (num * num) / (v_ss * y_ss)


def corr_sq_noalloc(v, yc, y_ss):
    n = v.shape[0]
    sv = float(v.sum())
    v_ss = float(v @ v) - sv * sv / n
    if v_ss < 1e-24 or y_ss < 1e-24:
        return 0.0
    num = float(v @ yc)  # yc sums to 0 -> identical to vc @ yc
    return (num * num) / (v_ss * y_ss)


def main():
    rng = np.random.default_rng(0)
    for n in (533, 1100, 1667, 3333):
        z = np.sort(rng.random(n))
        cases = []
        for _ in range(80):
            f = 0.25 + 5 * rng.random()
            v = np.sin(2 * np.pi * f * z)
            y = np.sin(2 * np.pi * (0.25 + 5 * rng.random()) * z) + 0.3 * rng.standard_normal(n)
            yc = y - y.mean()
            cases.append((v, yc, float(yc @ yc)))
        md = max(abs(corr_sq_orig(v, yc, y) - corr_sq_noalloc(v, yc, y)) for v, yc, y in cases)
        reps = 2000
        t0 = time.perf_counter()
        for _ in range(reps):
            for v, yc, y_ss in cases:
                corr_sq_orig(v, yc, y_ss)
        t_o = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(reps):
            for v, yc, y_ss in cases:
                corr_sq_noalloc(v, yc, y_ss)
        t_n = time.perf_counter() - t0
        print(f"n={n:5d}  orig={t_o*1e3:8.2f}ms  noalloc={t_n*1e3:8.2f}ms  "
              f"speedup={t_o/t_n:.2f}x  max|df|={md:.2e}")


if __name__ == "__main__":
    main()
