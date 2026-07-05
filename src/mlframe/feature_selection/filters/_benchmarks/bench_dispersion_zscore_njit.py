"""Bench + identity for Family-D _zscore_from_bins: fuse the multi-pass numpy
(clip + 2 gathers + where-floor + isfinite + masked divide) into a single njit
loop. _zscore_from_bins is the top tottime frame in
generate_conditional_dispersion_features (0.71s/2.6s at n=200k, p=6, called once
per ordered (x_i,x_j) pair).

bit-identity: the njit loop computes per row exactly
  s = bin_std[codes[i]]; if s < FLOOR: s = 1.0
  z[i] = (xi[i] - bin_mean[codes[i]]) / s   when isfinite(xi[i]) else 0.0
which is the SAME float64 op sequence (subtract, single divide) as the numpy
path -- no reduction-order concern (per-element independent).

Run: python src/mlframe/feature_selection/filters/_benchmarks/bench_dispersion_zscore_njit.py
"""
import time
import numpy as np
from numba import njit

_FLOOR = 1e-9


def old_zscore(xi, codes_j, bin_mean, bin_std):
    codes_j = np.clip(codes_j, 0, bin_mean.size - 1)
    per_row_mean = bin_mean[codes_j]
    per_row_std = bin_std[codes_j]
    per_row_std = np.where(per_row_std >= _FLOOR, per_row_std, 1.0)
    finite_i = np.isfinite(xi)
    z = np.zeros_like(xi, dtype=np.float64)
    z[finite_i] = (xi[finite_i] - per_row_mean[finite_i]) / per_row_std[finite_i]
    return z


@njit(cache=True, nogil=True)
def _zscore_njit(xi, codes_j, bin_mean, bin_std, floor):
    n = xi.shape[0]
    nb = bin_mean.shape[0]
    z = np.zeros(n, dtype=np.float64)
    for i in range(n):
        v = xi[i]
        if not np.isfinite(v):
            continue
        c = codes_j[i]
        if c < 0:
            c = 0
        elif c >= nb:
            c = nb - 1
        s = bin_std[c]
        if s < floor:
            s = 1.0
        z[i] = (v - bin_mean[c]) / s
    return z


def new_zscore(xi, codes_j, bin_mean, bin_std):
    return _zscore_njit(xi, np.ascontiguousarray(codes_j), bin_mean, bin_std, _FLOOR)


def main():
    rng = np.random.default_rng(0)
    nb = 10
    bin_mean = rng.standard_normal(nb)
    bin_std = np.abs(rng.standard_normal(nb)) + 0.1
    bin_std[2] = 0.0  # force a floored bin
    for n in (10_000, 100_000, 1_000_000):
        xi = rng.standard_normal(n)
        xi[rng.random(n) < 0.05] = np.nan
        codes = rng.integers(0, nb, size=n).astype(np.int64)
        o = old_zscore(xi, codes, bin_mean, bin_std)
        ne = new_zscore(xi, codes, bin_mean, bin_std)
        md = np.max(np.abs(o - ne))

        def timeit(fn, reps=300 if n <= 100_000 else 60):
            for _ in range(3):
                fn(xi, codes, bin_mean, bin_std)
            best = 1e9
            for _ in range(reps):
                t = time.perf_counter()
                fn(xi, codes, bin_mean, bin_std)
                best = min(best, time.perf_counter() - t)
            return best
        to = timeit(old_zscore)
        tn = timeit(new_zscore)
        print(f"n={n:>9}: OLD {to*1e6:9.1f}us  NEW {tn*1e6:9.1f}us  " f"speedup {to/tn:5.2f}x  max|diff|={md:.2e}")


if __name__ == "__main__":
    main()
