"""A/B bench: general Tweedie deviance kernel, 3 pow/row (OLD) vs 2 pow/row reusing yp**(2-p)=yp**(1-p)*yp (NEW).

Run: python src/mlframe/metrics/_benchmarks/bench_tweedie_general_pow_reuse.py (PYTHONPATH=src, CUDA off).

The general 1<p<2 / p>2 kernel computed three variable-exponent powers per row: yt**(2-p), yp**(1-p), yp**(2-p).
yp**(2-p) == yp**(1-p) * yp algebraically, so we drop one transcendental pow/row (each is exp(log) under the hood).
Identity: bit-equiv up to a single-ULP FP reorder (mul vs independent pow); validated vs sklearn.mean_tweedie_deviance.
"""
import sys; sys.modules['cupy'] = None
import numpy as np, numba, time
from numba import njit
from math import log

NP = dict(cache=True, fastmath=True, nogil=True)


@njit(**NP)
def old_kernel(y_true, y_pred, power):
    s = 0.0; used = 0; invalid = 0; p = power
    for i in range(y_true.shape[0]):
        yt = y_true[i]; yp = y_pred[i]
        if yp <= 0.0 or yt < 0.0:
            invalid += 1; continue
        if yt == 0.0:
            term_y = 0.0
        else:
            term_y = (yt ** (2.0 - p)) / ((1.0 - p) * (2.0 - p))
        term_yp = yt * (yp ** (1.0 - p)) / (1.0 - p)
        term_p = (yp ** (2.0 - p)) / (2.0 - p)
        s += 2.0 * (term_y - term_yp + term_p)
        used += 1
    return (s / used) if used > 0 else np.nan, invalid


@njit(**NP)
def new_kernel(y_true, y_pred, power):
    s = 0.0; used = 0; invalid = 0; p = power
    c_y = 1.0 / ((1.0 - p) * (2.0 - p))
    c_yp = 1.0 / (1.0 - p)
    c_p = 1.0 / (2.0 - p)
    e1 = 1.0 - p
    e2 = 2.0 - p
    for i in range(y_true.shape[0]):
        yt = y_true[i]; yp = y_pred[i]
        if yp <= 0.0 or yt < 0.0:
            invalid += 1; continue
        if yt == 0.0:
            term_y = 0.0
        else:
            term_y = (yt ** e2) * c_y
        yp_pow1 = yp ** e1
        term_yp = yt * yp_pow1 * c_yp
        term_p = (yp_pow1 * yp) * c_p
        s += 2.0 * (term_y - term_yp + term_p)
        used += 1
    return (s / used) if used > 0 else np.nan, invalid


def bench():
    rng = np.random.default_rng(0)
    for n in (10_000, 100_000, 1_000_000):
        yt = rng.gamma(2.0, 1.5, n)
        yp = rng.gamma(2.0, 1.5, n) + 0.1
        for power in (1.5, 2.5):
            old_kernel(yt[:10], yp[:10], power); new_kernel(yt[:10], yp[:10], power)
            o = old_kernel(yt, yp, power); nw = new_kernel(yt, yp, power)
            def t(fn):
                best = 1e9
                for _ in range(7):
                    s = time.perf_counter(); fn(yt, yp, power); d = time.perf_counter() - s
                    best = min(best, d)
                return best
            to = t(old_kernel); tn = t(new_kernel)
            print(f"n={n:>9} p={power}: OLD {to*1e3:7.3f}ms  NEW {tn*1e3:7.3f}ms  {to/tn:.3f}x  "
                  f"absdiff={abs(o[0]-nw[0]):.3e}")
    # sklearn identity
    try:
        from sklearn.metrics import mean_tweedie_deviance
        yt = rng.gamma(2.0, 1.5, 50_000); yp = rng.gamma(2.0, 1.5, 50_000) + 0.1
        for power in (1.5, 2.5):
            sk = mean_tweedie_deviance(yt, yp, power=power)
            nw = new_kernel(yt, yp, power)[0]
            print(f"sklearn p={power}: sklearn={sk:.10f} new={nw:.10f} absdiff={abs(sk-nw):.3e}")
    except Exception as e:
        print("sklearn check skipped:", e)


if __name__ == "__main__":
    bench()
