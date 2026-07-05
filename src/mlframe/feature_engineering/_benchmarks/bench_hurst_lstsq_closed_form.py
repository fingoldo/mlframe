"""Bench: replace the SVD-based ``np.linalg.lstsq`` log-log R/S fit in
``compute_hurst_exponent`` (hurst.py:~150) with the closed-form 2-parameter OLS
(slope/intercept via the covariance normal equations).

The fit is a single-regressor linear model ``y = h*xv + c`` (xv=log10(window_sizes),
y=log10(rs)). ``np.linalg.lstsq`` solves it via a full SVD of an (M x 2) design matrix
plus a vstack/transpose to assemble that matrix -- vastly over-powered for 2 unknowns.
The closed form ``h = cov(xv,y)/var(xv); c = mean(y)-h*mean(xv)`` is the SAME OLS
solution, with no SVD, no design-matrix allocation, no ones-column.

Run:
    python -m mlframe.feature_engineering._benchmarks.bench_hurst_lstsq_closed_form
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
import numpy as np


def old_fit(window_sizes_arr, rs_arr):
    x = np.vstack([np.log10(window_sizes_arr), np.ones(len(rs_arr))]).T
    h, c = np.linalg.lstsq(x, np.log10(rs_arr), rcond=None)[0]
    c = 10**c
    return h, c


def new_fit(window_sizes_arr, rs_arr):
    xv = np.log10(window_sizes_arr)
    yv = np.log10(rs_arr)
    xm = xv.mean()
    ym = yv.mean()
    dx = xv - xm
    var_x = (dx * dx).sum()
    h = (dx * (yv - ym)).sum() / var_x
    c = 10 ** (ym - h * xm)
    return h, c


def _make_ladder(rng):
    # Realistic R/S ladder: ~10-18 window sizes log-spaced; rs grows ~n**0.5 with noise.
    k = rng.integers(8, 18)
    ws = np.unique(np.round(np.logspace(np.log10(5), np.log10(500), k)).astype(int)).astype(float)
    rs = (ws ** (0.4 + 0.3 * rng.random())) * (1.0 + 0.05 * rng.standard_normal(len(ws)))
    rs = np.abs(rs) + 1e-6
    return ws, rs


def main():
    rng = np.random.default_rng(0)
    ladders = [_make_ladder(rng) for _ in range(5000)]

    # Identity gate
    max_h, max_c = 0.0, 0.0
    for ws, rs in ladders:
        oh, oc = old_fit(ws, rs)
        nh, nc = new_fit(ws, rs)
        max_h = max(max_h, abs(oh - nh))
        max_c = max(max_c, abs(oc - nc) / max(abs(oc), 1e-12))
    print(f"identity: max |dh|={max_h:.3e}  max rel|dc|={max_c:.3e}")

    def bench(fn, reps=5):
        best = float("inf")
        for _ in range(reps):
            t = time.perf_counter()
            for ws, rs in ladders:
                fn(ws, rs)
            best = min(best, time.perf_counter() - t)
        return best

    bench(old_fit, 2); bench(new_fit, 2)  # warm
    to = bench(old_fit)
    tn = bench(new_fit)
    print(f"OLD lstsq:   {to*1e3:8.2f} ms / {len(ladders)} fits")
    print(f"NEW closed:  {tn*1e3:8.2f} ms / {len(ladders)} fits")
    print(f"speedup: {to/tn:.2f}x")


if __name__ == "__main__":
    main()
