"""Bench replacing ``z ** k`` with chained multiplication in _excess_kurtosis / _skewness.

np.power dispatches through a general-purpose path that's ~7x slower
than the inline ``z * z * z`` / ``(z*z) * (z*z)`` form for integer
exponents 3 / 4 (already validated in iter129 for audit_residuals;
this is the same antipattern at two more sites).

c0125 iter138 profile attributed ~32ms each to _skewness and
_excess_kurtosis (called once per regression fit during the mini-HPT
target-distribution-analyzer pass). The fix removes the np.power
dispatch on the (N,) z-score array.

Run: ``python profiling/bench_target_moments_no_power.py``
"""

import time
import math
import numpy as np


def kurtosis_power(y):
    n = y.size
    if n < 4:
        return 0.0
    mu = float(np.mean(y))
    sigma = float(np.std(y))
    if sigma <= 0.0 or not math.isfinite(sigma):
        return 0.0
    z = (y - mu) / sigma
    return float(np.mean(z ** 4)) - 3.0


def kurtosis_chain(y):
    n = y.size
    if n < 4:
        return 0.0
    mu = float(np.mean(y))
    sigma = float(np.std(y))
    if sigma <= 0.0 or not math.isfinite(sigma):
        return 0.0
    z = (y - mu) / sigma
    z2 = z * z
    return float(np.mean(z2 * z2)) - 3.0


def skewness_power(y):
    n = y.size
    if n < 3:
        return 0.0
    mu = float(np.mean(y))
    sigma = float(np.std(y))
    if sigma <= 0.0 or not math.isfinite(sigma):
        return 0.0
    z = (y - mu) / sigma
    return float(np.mean(z ** 3))


def skewness_chain(y):
    n = y.size
    if n < 3:
        return 0.0
    mu = float(np.mean(y))
    sigma = float(np.std(y))
    if sigma <= 0.0 or not math.isfinite(sigma):
        return 0.0
    z = (y - mu) / sigma
    return float(np.mean(z * z * z))


def bench(label, fn, y, n_iter=50):
    for _ in range(5):
        fn(y)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n_iter):
            fn(y)
        times.append((time.perf_counter() - t) / n_iter)
    return min(times) * 1e3, label


if __name__ == "__main__":
    np.random.seed(0)
    for n in [50_000, 200_000, 1_000_000, 5_000_000]:
        y = np.random.randn(n).astype(np.float64)
        tp, _ = bench("kurt power", kurtosis_power, y)
        tc, _ = bench("kurt chain", kurtosis_chain, y)
        sp, _ = bench("skew power", skewness_power, y)
        sc, _ = bench("skew chain", skewness_chain, y)

        # Numerical equivalence
        kp = kurtosis_power(y); kc = kurtosis_chain(y)
        sp_v = skewness_power(y); sc_v = skewness_chain(y)
        print(f"n={n:>8}: kurt power={tp:6.3f}ms chain={tc:6.3f}ms ({tp/tc:.2f}x) | "
              f"skew power={sp:6.3f}ms chain={sc:6.3f}ms ({sp/sc:.2f}x)  "
              f"|kdif|={abs(kp-kc):.2e} |sdif|={abs(sp_v-sc_v):.2e}")
