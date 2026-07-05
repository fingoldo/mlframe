"""Microbench + identity check for the loop-invariant np.arange hoist in hurst DFA kernels.

OLD = pre-hoist kernels (np.arange(s) + derived invariants recomputed inside the per-segment j-loop),
copied verbatim from the pre-edit source. NEW = the current (hoisted) kernels imported from mlframe.

Run: python -m mlframe.feature_engineering._benchmarks.bench_hurst_arange_hoist
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit

from mlframe.feature_engineering.hurst import (
    dfa_alpha as NEW_dfa_alpha,
    dfa_alpha2_quadratic as NEW_dfa_alpha2,
    multifractal_dfa as NEW_mfdfa,
)


@njit(cache=True, fastmath=True)
def OLD_dfa_alpha(x):
    n = x.size
    if n < 20:
        return np.nan
    mu = x.mean()
    y = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc += x[i] - mu
        y[i] = acc
    sizes = np.array([10, 20, 40, 80])
    sizes = sizes[sizes < n // 2]
    if sizes.size < 2:
        return np.nan
    log_s = np.empty(sizes.size)
    log_f = np.empty(sizes.size)
    for k in range(sizes.size):
        s = sizes[k]
        m = n // s
        var_sum = 0.0
        for j in range(m):
            seg = y[j * s : (j + 1) * s]
            t = np.arange(s).astype(np.float64)
            tm = t.mean()
            sm = seg.mean()
            num = 0.0
            den = 0.0
            for i in range(s):
                num += (t[i] - tm) * (seg[i] - sm)
                den += (t[i] - tm) ** 2
            slope = num / (den + 1e-12)
            intercept = sm - slope * tm
            resid_sq = 0.0
            for i in range(s):
                fit = intercept + slope * t[i]
                resid_sq += (seg[i] - fit) ** 2
            var_sum += resid_sq / s
        f_s = np.sqrt(var_sum / m)
        log_s[k] = np.log(s)
        log_f[k] = np.log(f_s + 1e-12)
    lm = log_s.mean()
    fm = log_f.mean()
    num = 0.0
    den = 0.0
    for k in range(log_s.size):
        num += (log_s[k] - lm) * (log_f[k] - fm)
        den += (log_s[k] - lm) ** 2
    return num / (den + 1e-12)


@njit(cache=True, fastmath=True)
def OLD_dfa_alpha2(x):
    n = x.size
    if n < 50:
        return np.nan
    mu = x.mean()
    y = np.empty(n)
    acc = 0.0
    for i in range(n):
        acc += x[i] - mu
        y[i] = acc
    sizes = np.array([10, 20, 40, 80])
    sizes = sizes[sizes < n // 2]
    if sizes.size < 2:
        return np.nan
    log_s = np.empty(sizes.size)
    log_f = np.empty(sizes.size)
    for k in range(sizes.size):
        s = sizes[k]
        m = n // s
        var_sum = 0.0
        for j in range(m):
            seg = y[j * s : (j + 1) * s]
            t = np.arange(s).astype(np.float64)
            S0 = float(s)
            S1 = t.sum()
            S2 = (t * t).sum()
            S3 = (t * t * t).sum()
            S4 = (t * t * t * t).sum()
            Sy = seg.sum()
            Sty = (t * seg).sum()
            St2y = (t * t * seg).sum()
            M00 = S0; M01 = S1; M02 = S2
            M11 = S2; M12 = S3
            M22 = S4
            det = M00 * (M11 * M22 - M12 * M12) - M01 * (M01 * M22 - M12 * M02) + M02 * (M01 * M12 - M11 * M02)
            if abs(det) < 1e-12:
                continue
            inv_det = 1.0 / det
            c00 = (M11 * M22 - M12 * M12) * inv_det
            c01 = -(M01 * M22 - M12 * M02) * inv_det
            c02 = (M01 * M12 - M11 * M02) * inv_det
            c11 = (M00 * M22 - M02 * M02) * inv_det
            c12 = -(M00 * M12 - M01 * M02) * inv_det
            c22 = (M00 * M11 - M01 * M01) * inv_det
            a = c00 * Sy + c01 * Sty + c02 * St2y
            b = c01 * Sy + c11 * Sty + c12 * St2y
            cq = c02 * Sy + c12 * Sty + c22 * St2y
            resid_sq = 0.0
            for i in range(s):
                fit = a + b * t[i] + cq * t[i] * t[i]
                d = seg[i] - fit
                resid_sq += d * d
            var_sum += resid_sq / s
        f_s = np.sqrt(var_sum / m)
        log_s[k] = np.log(s)
        log_f[k] = np.log(f_s + 1e-12)
    lm = log_s.mean()
    fm = log_f.mean()
    num = 0.0
    den = 0.0
    for k in range(log_s.size):
        num += (log_s[k] - lm) * (log_f[k] - fm)
        den += (log_s[k] - lm) ** 2
    return num / (den + 1e-12)


def _bestof(fn, arg, n=200):
    fn(arg)  # warm
    best = 1e9
    for _ in range(n):
        t0 = time.perf_counter()
        fn(arg)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.standard_normal(2000)).astype(np.float64)
    q_values = np.array([-5.0, -3.0, -1.0, 2.0, 3.0, 5.0])
    scales = np.array([10, 20, 40, 80, 160])

    # identity
    assert OLD_dfa_alpha(x) == NEW_dfa_alpha(x), "dfa_alpha diverged"
    assert OLD_dfa_alpha2(x) == NEW_dfa_alpha2(x), "dfa_alpha2_quadratic diverged"
    print("identity: dfa_alpha OK, dfa_alpha2_quadratic OK (exact ==)")

    for name, old, new in (
        ("dfa_alpha", OLD_dfa_alpha, NEW_dfa_alpha),
        ("dfa_alpha2_quadratic", OLD_dfa_alpha2, NEW_dfa_alpha2),
    ):
        bo = _bestof(old, x)
        bn = _bestof(new, x)
        print(f"{name:24s} OLD {bo*1e6:8.2f}us  NEW {bn*1e6:8.2f}us  speedup {bo/bn:.3f}x")

    # mfdfa: just time NEW (no inline OLD; the hoist is identical-by-construction).
    NEW_mfdfa(x, q_values, scales)
    print(f"multifractal_dfa NEW {_bestof(lambda a: NEW_mfdfa(a, q_values, scales), x)*1e6:.2f}us")


if __name__ == "__main__":
    main()
