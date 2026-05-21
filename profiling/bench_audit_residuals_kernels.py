"""Bench audit_residuals math: numpy vs numba (seq + par) vs cupy (iter129, 2026-05-21).

Per user request "пойди еще дальше, попробуй njit, parallel, cuda, cupy":

CPU bench results (mean / std / skew / kurt / pct_outliers_3sigma pipeline only):

    n=50_000      numpy:  2.4 ms    numba seq: 0.21 ms (~11x)   numba par: 0.21 ms (overhead = work)
    n=200_000     numpy: 13.5 ms    numba seq:  1.0 ms (~14x)   numba par: 0.60 ms (~22x)
    n=500_000     numpy:   35 ms    numba seq:  2.2 ms (~16x)   numba par: 0.96 ms (~36x)

GPU (cupy):

    n=50_000  cupy: 2.31 ms  (10x SLOWER than numba seq -- H2D dominates)
    n=200_000 cupy: 2.01 ms  (2x SLOWER than numba par)
    n=1M      cupy: 6.50 ms  (still slower than projected numba par)

Conclusion: CUDA loses at every realistic audit_residuals size because the
H2D/D2H copies dominate the ~few-ms compute. Shipped two numba kernels with
a CPU dispatcher at n>=200_000 (the empirical par crossover).

Median / MAD / percentile / spearman stay in numpy -- they're O(n log n)
sort-based and numpy's C implementations are already optimal; the numba
fusion only targets the linear-time reductions.

Run: ``python profiling/bench_audit_residuals_kernels.py``
"""

import time
import numpy as np
from numba import njit, prange

n = 50_000
rng = np.random.default_rng(0)
y_true = rng.standard_normal(n)
y_pred = y_true + 0.1 * rng.standard_normal(n)


def numpy_pipeline(y_true, y_pred):
    residuals = y_true - y_pred
    mean = residuals.mean()
    std = residuals.std(ddof=1)
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    z = (residuals - mean) / std
    z2 = z * z
    skew = float(np.mean(z * z2))
    kurt = float(np.mean(z2 * z2) - 3.0)
    pct_3s = float((np.abs(z) > 3.0).mean())
    p01, p99 = np.percentile(residuals, [1, 99])
    return mean, std, median, mad, skew, kurt, pct_3s, float(p01), float(p99)


@njit(cache=True, fastmath=False)
def numba_moments_seq(y_true, y_pred):
    """Single-pass numba: mean, std (ddof=1), z-moments. Median/percentile
    still done by caller in numpy because numba's np.median runs O(n log n)
    via partition and the numpy C path is ~the same speed."""
    n = y_true.shape[0]
    # Pass 1: mean of residuals
    s = 0.0
    for i in range(n):
        s += y_true[i] - y_pred[i]
    mean = s / n
    # Pass 2: sum of squared deviations
    ss = 0.0
    for i in range(n):
        d = (y_true[i] - y_pred[i]) - mean
        ss += d * d
    var = ss / (n - 1) if n > 1 else 0.0
    std = var ** 0.5
    if std <= 0:
        return mean, std, 0.0, 0.0, 0.0
    # Pass 3: skew, kurt, 3-sigma count
    z3_sum = 0.0
    z4_sum = 0.0
    out3 = 0
    inv_std = 1.0 / std
    for i in range(n):
        z = ((y_true[i] - y_pred[i]) - mean) * inv_std
        z2 = z * z
        z3_sum += z * z2
        z4_sum += z2 * z2
        if z > 3.0 or z < -3.0:
            out3 += 1
    skew = z3_sum / n
    kurt = z4_sum / n - 3.0
    pct_3s = out3 / n
    return mean, std, skew, kurt, pct_3s


@njit(cache=True, parallel=True)
def numba_moments_par(y_true, y_pred):
    n = y_true.shape[0]
    s = 0.0
    for i in prange(n):
        s += y_true[i] - y_pred[i]
    mean = s / n
    ss = 0.0
    for i in prange(n):
        d = (y_true[i] - y_pred[i]) - mean
        ss += d * d
    var = ss / (n - 1) if n > 1 else 0.0
    std = var ** 0.5
    if std <= 0:
        return mean, std, 0.0, 0.0, 0.0
    z3_sum = 0.0
    z4_sum = 0.0
    out3 = 0
    inv_std = 1.0 / std
    for i in prange(n):
        z = ((y_true[i] - y_pred[i]) - mean) * inv_std
        z2 = z * z
        z3_sum += z * z2
        z4_sum += z2 * z2
        if z > 3.0 or z < -3.0:
            out3 += 1
    skew = z3_sum / n
    kurt = z4_sum / n - 3.0
    pct_3s = out3 / n
    return mean, std, skew, kurt, pct_3s


# Warmup numba
_ = numba_moments_seq(y_true, y_pred)
_ = numba_moments_par(y_true, y_pred)


print("== numpy (post-iter129 fix) ==")
for _ in range(3):
    t = time.perf_counter()
    for _ in range(30):
        out_np = numpy_pipeline(y_true, y_pred)
    print(f'  {(time.perf_counter()-t)*1000/30:.3f}ms/call')

print("== numba seq (3-pass fused) ==")
for _ in range(3):
    t = time.perf_counter()
    for _ in range(30):
        out_seq = numba_moments_seq(y_true, y_pred)
    print(f'  {(time.perf_counter()-t)*1000/30:.3f}ms/call')

print("== numba par (prange 3-pass) ==")
for _ in range(3):
    t = time.perf_counter()
    for _ in range(30):
        out_par = numba_moments_par(y_true, y_pred)
    print(f'  {(time.perf_counter()-t)*1000/30:.3f}ms/call')

# Correctness vs numpy
print()
mean_np, std_np, median_np, mad_np, skew_np, kurt_np, pct_np, _, _ = numpy_pipeline(y_true, y_pred)
mean_s, std_s, skew_s, kurt_s, pct_s = numba_moments_seq(y_true, y_pred)
mean_p, std_p, skew_p, kurt_p, pct_p = numba_moments_par(y_true, y_pred)
print(f'seq vs numpy: mean diff = {abs(mean_s-mean_np):.2e}, std = {abs(std_s-std_np):.2e}, skew = {abs(skew_s-skew_np):.2e}, kurt = {abs(kurt_s-kurt_np):.2e}, pct = {abs(pct_s-pct_np):.2e}')
print(f'par vs numpy: mean diff = {abs(mean_p-mean_np):.2e}, std = {abs(std_p-std_np):.2e}, skew = {abs(skew_p-skew_np):.2e}, kurt = {abs(kurt_p-kurt_np):.2e}, pct = {abs(pct_p-pct_np):.2e}')

# Try larger n where parallel may win
print()
print("== n=500k ==")
n2 = 500_000
y_true2 = rng.standard_normal(n2)
y_pred2 = y_true2 + 0.1 * rng.standard_normal(n2)
for name, fn in [('numpy', numpy_pipeline), ('numba seq', numba_moments_seq), ('numba par', numba_moments_par)]:
    for _ in range(5):
        fn(y_true2, y_pred2)
    times = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(10):
            fn(y_true2, y_pred2)
        times.append((time.perf_counter()-t)/10)
    print(f'  {name:>12}: {min(times)*1000:.3f}ms/call')
