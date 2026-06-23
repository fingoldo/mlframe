"""Bench + identity check for Family-D _per_bin_mean_std: replace the 3 np.add.at
scatter-adds (sum / count / centred-SS) with np.bincount.

Mirrors the already-shipped Family-B fix (generate_conditional_residual_features
lines 481-482: "np.bincount accumulates per bin in element order exactly as
np.add.at does -- bit-identical sum/count, but a single C pass instead of the
unbuffered scatter"). Family D's _per_bin_mean_std was missed and still uses
np.add.at for all three reductions.

bench-attempt-rejected (2026-06-23): bit-identical (max|diff|=0.0) but NO win --
1.05x / 0.97x / 0.99x at n=10k/100k/1M. With only n_bins (~10) scatter targets
the np.add.at is already cheap vs the surrounding full-n float math; bincount does
not help here (unlike Family-B's larger-scatter case). Kept as a negative result.

Run: python src/mlframe/feature_selection/filters/_benchmarks/bench_dispersion_per_bin_bincount.py
"""
import time
import numpy as np

_FLOOR = 1e-9
_MIN_ROWS = 2


def old_per_bin(xi, codes_j, n_bins_eff):
    finite_i = np.isfinite(xi)
    xi_f = xi[finite_i]
    codes_f = codes_j[finite_i]
    global_mean = float(xi_f.mean())
    global_std = float(xi_f.std())
    if not np.isfinite(global_std) or global_std < _FLOOR:
        global_std = 1.0
    bin_sum = np.zeros(n_bins_eff, dtype=np.float64)
    bin_cnt = np.zeros(n_bins_eff, dtype=np.float64)
    np.add.at(bin_sum, codes_f, xi_f)
    np.add.at(bin_cnt, codes_f, 1.0)
    bin_mean = np.where(bin_cnt > 0.0, bin_sum / np.maximum(bin_cnt, 1.0), global_mean)
    centred = xi_f - bin_mean[codes_f]
    bin_css = np.zeros(n_bins_eff, dtype=np.float64)
    np.add.at(bin_css, codes_f, centred * centred)
    with np.errstate(invalid="ignore", divide="ignore"):
        bin_std = np.sqrt(np.where(bin_cnt > 0.0, bin_css / np.maximum(bin_cnt, 1.0), 0.0))
    usable = (bin_cnt >= float(_MIN_ROWS)) & (bin_std >= _FLOOR)
    bin_std = np.where(usable, bin_std, global_std)
    return bin_mean, bin_std, global_mean, global_std


def new_per_bin(xi, codes_j, n_bins_eff):
    finite_i = np.isfinite(xi)
    xi_f = xi[finite_i]
    codes_f = codes_j[finite_i]
    global_mean = float(xi_f.mean())
    global_std = float(xi_f.std())
    if not np.isfinite(global_std) or global_std < _FLOOR:
        global_std = 1.0
    bin_sum = np.bincount(codes_f, weights=xi_f, minlength=n_bins_eff).astype(np.float64)
    bin_cnt = np.bincount(codes_f, minlength=n_bins_eff).astype(np.float64)
    bin_mean = np.where(bin_cnt > 0.0, bin_sum / np.maximum(bin_cnt, 1.0), global_mean)
    centred = xi_f - bin_mean[codes_f]
    bin_css = np.bincount(codes_f, weights=centred * centred, minlength=n_bins_eff).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        bin_std = np.sqrt(np.where(bin_cnt > 0.0, bin_css / np.maximum(bin_cnt, 1.0), 0.0))
    usable = (bin_cnt >= float(_MIN_ROWS)) & (bin_std >= _FLOOR)
    bin_std = np.where(usable, bin_std, global_std)
    return bin_mean, bin_std, global_mean, global_std


def main():
    rng = np.random.default_rng(0)
    for n in (10_000, 100_000, 1_000_000):
        for nb in (10,):
            xi = rng.standard_normal(n)
            xi[rng.random(n) < 0.05] = np.nan  # some NaNs
            codes = rng.integers(0, nb, size=n).astype(np.int64)
            o = old_per_bin(xi, codes, nb)
            ne = new_per_bin(xi, codes, nb)
            md = max(np.max(np.abs(o[0] - ne[0])), np.max(np.abs(o[1] - ne[1])))
            # warm + best-of-N
            def timeit(fn, reps=200 if n <= 100_000 else 30):
                best = 1e9
                for _ in range(3):
                    fn(xi, codes, nb)
                for _ in range(reps):
                    t = time.perf_counter()
                    fn(xi, codes, nb)
                    best = min(best, time.perf_counter() - t)
                return best
            to = timeit(old_per_bin)
            tn = timeit(new_per_bin)
            print(f"n={n:>9} nb={nb}: OLD {to*1e6:9.1f}us  NEW {tn*1e6:9.1f}us  "
                  f"speedup {to/tn:5.2f}x  max|diff|={md:.2e}")


if __name__ == "__main__":
    main()
