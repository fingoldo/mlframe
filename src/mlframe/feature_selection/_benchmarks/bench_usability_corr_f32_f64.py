"""Micro-benchmark: f32 vs f64 for the MRMR usability float hot paths (usability_form_corrs / abs_pearson).

Times both dtypes at a representative production shape (n=250k operands, ~500 pair calls -> the module runs ~8 full-n Pearson passes
per usability_form_corrs) and reports the MAX |corr| divergence between the f32 and f64 form materialisation, so a maintainer can
confirm f32 stays inside the selection-safe band (< 1e-3) while quantifying the speed/memory trade.

Run:  CUDA_VISIBLE_DEVICES="" python -m mlframe.feature_selection._benchmarks.bench_usability_corr_f32_f64
"""
import time
import numpy as np

from mlframe.feature_selection.filters import _fe_usability_signal as U


def _forms(y, x0, x1, dtype):
    eps = 1e-12
    _y = np.asarray(y, dtype=dtype).ravel(); _x0 = np.asarray(x0, dtype=dtype).ravel(); _x1 = np.asarray(x1, dtype=dtype).ravel()

    def sd(n, d):
        return n / np.where(np.abs(d) < eps, np.nan, d)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pf = [sd(_x0, _x1), sd(_x1, _x0), sd(_x0 * _x0, _x1), sd(_x1 * _x1, _x0), _x0 * _x1]
        sf = [_x0, _x1, _x0 * _x0, _x1 * _x1]
        cp = max((U.abs_pearson(_y, f) for f in pf), default=0.0)
        cs = max((U.abs_pearson(_y, f) for f in sf), default=0.0)
    return cp, cs


def main(n=250_000, n_calls=500, seed=0):
    rng = np.random.default_rng(seed)
    # Heavy-tailed operands + a tail-concentrated a**2/b target (the regime the module exists for).
    pairs = []
    for _ in range(n_calls):
        scale = 10.0 ** rng.integers(2, 7)
        x0 = rng.standard_t(3, n) * scale
        x1 = rng.standard_t(3, n) * scale
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            sig = (x0 * x0) / np.where(np.abs(x1) < 1e-12, np.nan, x1)
        y = 0.6 * np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0) + 0.4 * rng.normal(0, np.std(sig) + 1e-9, n)
        pairs.append((y, x0, x1))

    # Warm njit.
    _forms(*pairs[0], np.float64); _forms(*pairs[0], np.float32)

    results = {}
    max_div = 0.0
    for dt in (np.float64, np.float32):
        t0 = time.perf_counter()
        for y, x0, x1 in pairs:
            cp, cs = _forms(y, x0, x1, dt)
            if dt is np.float32:
                cp64, cs64 = _f64_cache[id(y)]
                max_div = max(max_div, abs(cp - cp64), abs(cs - cs64))
            else:
                _f64_cache[id(y)] = (cp, cs)
        results[dt.__name__] = time.perf_counter() - t0

    print(f"\nusability float f32-vs-f64 microbench  (n={n}, calls={n_calls})")
    print("-" * 56)
    print(f"{'dtype':<10}{'wall (s)':>14}{'per-call (ms)':>18}")
    for name, t in results.items():
        print(f"{name:<10}{t:>14.3f}{t / n_calls * 1e3:>18.3f}")
    speedup = results["float64"] / results["float32"] if results["float32"] else float("nan")
    print("-" * 56)
    print(f"f32 speedup vs f64 : {speedup:.2f}x")
    print(f"max |corr| f32-vs-f64 divergence : {max_div:.3e}  (selection-safe if < 1e-3)")


_f64_cache = {}

if __name__ == "__main__":
    main()
