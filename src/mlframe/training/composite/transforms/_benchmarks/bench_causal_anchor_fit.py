"""cProfile + njit A/B bench for ``causal_anchor_residual`` (fit + forward + inverse).

The fit is a two-pass trimmed-LS slope (two ``np.median`` sorts inside the MAD
trim) + a scalar shrink/clamp; forward / inverse are single fused AXPY passes.
This bench profiles the three ops at a representative shape and A/Bs an ``@njit``
robust-slope rewrite against the numpy path to justify the default backend.

Run::

    CUDA_VISIBLE_DEVICES="" python -m mlframe.training.composite.transforms._benchmarks.bench_causal_anchor_fit
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

from mlframe.training.composite.transforms._causal_anchor import (
    _CAUSAL_ANCHOR_MAD_K,
    _CAUSAL_ANCHOR_MIN_KEEP_FRAC,
    _causal_anchor_residual_fit,
    _causal_anchor_residual_forward,
    _causal_anchor_residual_inverse,
    _ols_slope_intercept,
    _robust_slope,
)

try:
    import numba as _numba

    @_numba.njit(cache=True)
    def _median_sorted_njit(a):
        # Sort-based median (bit-identical to np.median); np.median support in numba nopython is environment-fragile.
        s = np.sort(a)
        m = s.size // 2
        if s.size & 1:
            return s[m]
        return 0.5 * (s[m - 1] + s[m])

    @_numba.njit(cache=True)
    def _robust_slope_njit(base_f, y_f, k, min_keep_frac):
        n = base_f.size
        # Pass 1 OLS.
        mx = base_f.mean()
        my = y_f.mean()
        dx = base_f - mx
        vx = np.dot(dx, dx)
        if n < 2 or vx <= 0.0:
            return 0.0
        a1 = np.dot(dx, y_f - my) / vx
        b1 = my - a1 * mx
        resid = y_f - a1 * base_f - b1
        med = _median_sorted_njit(resid)
        mad = _median_sorted_njit(np.abs(resid - med))
        sigma = mad * 1.4826
        if sigma <= 0.0:
            return a1
        keep = np.abs(resid - med) <= k * sigma
        n_kept = keep.sum()
        thresh = 2 if 2 > int(min_keep_frac * n) else int(min_keep_frac * n)
        if n_kept < thresh or n_kept == n:
            return a1
        bk = base_f[keep]
        yk = y_f[keep]
        mxk = bk.mean()
        myk = yk.mean()
        dxk = bk - mxk
        vxk = np.dot(dxk, dxk)
        if vxk <= 0.0:
            return 0.0
        return np.dot(dxk, yk - myk) / vxk

    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False


def _make(n, seed=0, outlier_frac=0.02):
    rng = np.random.default_rng(seed)
    anchor = rng.standard_normal(n) * 3.0
    mr = rng.standard_normal(n)
    y = 0.6 * anchor + mr + rng.standard_normal(n) * 0.5
    n_out = int(outlier_frac * n)
    idx = rng.choice(n, size=n_out, replace=False)
    y[idx] += rng.standard_normal(n_out) * 80.0 + 60.0
    anchor[idx] += rng.standard_normal(n_out) * 20.0
    return anchor.astype(np.float64), y.astype(np.float64)


def _best_of(fn, *args, reps=15):
    best = float("inf")
    r = None
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best, r


def main():
    n = 200_000
    base, y = _make(n)
    p = _causal_anchor_residual_fit(y, base)
    T = _causal_anchor_residual_forward(y, base, p)

    print(f"=== cProfile: fit + forward + inverse, n={n} (x50) ===")
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(50):
        pp = _causal_anchor_residual_fit(y, base)
        TT = _causal_anchor_residual_forward(y, base, pp)
        _causal_anchor_residual_inverse(TT, base, pp)
    pr.disable()
    s = StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(10)
    print(s.getvalue())

    print("=== njit vs numpy robust-slope A/B ===")
    t_np, a_np = _best_of(_robust_slope, base, y)
    if _HAS_NUMBA:
        _robust_slope_njit(base, y, _CAUSAL_ANCHOR_MAD_K, _CAUSAL_ANCHOR_MIN_KEEP_FRAC)  # warm
        t_nb, a_nb = _best_of(
            _robust_slope_njit, base, y, _CAUSAL_ANCHOR_MAD_K, _CAUSAL_ANCHOR_MIN_KEEP_FRAC,
        )
        print(f"numpy  robust_slope: {t_np*1e3:8.3f} ms  alpha={a_np:.6f}")
        print(f"njit   robust_slope: {t_nb*1e3:8.3f} ms  alpha={a_nb:.6f}")
        print(f"speedup (numpy/njit): {t_np/t_nb:.2f}x   |alpha diff|={abs(a_np-a_nb):.2e}")
    else:
        print(f"numpy robust_slope: {t_np*1e3:8.3f} ms  (numba unavailable)")

    t_fwd, _ = _best_of(_causal_anchor_residual_forward, y, base, p)
    t_inv, _ = _best_of(_causal_anchor_residual_inverse, T, base, p)
    print(f"forward: {t_fwd*1e3:8.3f} ms   inverse: {t_inv*1e3:8.3f} ms  (n={n})")


if __name__ == "__main__":
    main()
