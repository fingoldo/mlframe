"""Host cost per call of the cupy operations the FE code uses most, on a 30k-row array.

The numbers (30-140 us per op on the dev box, ``std`` ~280 us, ``bincount`` ~500 us) are the reason a stage with thousands of tiny operations is launch-bound whatever its GPU time.
"""
import time, warnings
warnings.simplefilter("ignore")
import numpy as np, cupy as cp
n = 30000
a = cp.random.rand(n); b = cp.random.rand(n); m = a > 0.5; f32 = a.astype(cp.float32); idx = cp.arange(n)[::-1].copy()
M = cp.random.rand(n, 8)
def bench(label, fn, reps=400):
    fn(); cp.cuda.Device().synchronize()
    t = time.perf_counter()
    for _ in range(reps): fn()
    t_launch = (time.perf_counter() - t) / reps * 1e6
    cp.cuda.Device().synchronize()
    t_all = (time.perf_counter() - t) / reps * 1e6
    print(f"{label:28s} host {t_launch:7.1f} us/call   total {t_all:7.1f}")
bench("a + b", lambda: a + b)
bench("a * 2.0", lambda: a * 2.0)
bench("cp.where(m, a, b)", lambda: cp.where(m, a, b))
bench("cp.where(m, a, 0.0)", lambda: cp.where(m, a, 0.0))
bench("a.astype(float32)", lambda: a.astype(cp.float32))
bench("f32.astype(float64)", lambda: f32.astype(cp.float64))
bench("a.sum()", lambda: a.sum())
bench("a.std()", lambda: a.std())
bench("a.mean()", lambda: a.mean())
bench("cp.abs(a).max()", lambda: cp.abs(a).max())
bench("cp.dot(a, b)", lambda: cp.dot(a, b))
bench("a[idx] (take)", lambda: a[idx])
bench("cp.empty(n)", lambda: cp.empty(n))
bench("cp.stack([a,b])", lambda: cp.stack([a, b]))
bench("cp.isfinite(a)", lambda: cp.isfinite(a))
bench("m & m", lambda: m & m)
bench("M.sum(axis=0)", lambda: M.sum(axis=0))
bench("M.T @ a", lambda: M.T @ a)
bench("cp.asnumpy(a[:4])", lambda: cp.asnumpy(a[:4]))
bench("float(a.sum())", lambda: float(a.sum()))
bench("cp.sin(a)", lambda: cp.sin(a))
bench("cp.sort(a)", lambda: cp.sort(a))
bench("cp.argmax(a)", lambda: cp.argmax(a))
bench("cp.bincount(i32,minlength)", lambda: cp.bincount(cp.arange(n, dtype=cp.int32) % 10, minlength=10))
bench("cp.ascontiguousarray(a)", lambda: cp.ascontiguousarray(a))
bench("cp.concatenate([a,b])", lambda: cp.concatenate([a, b]))
